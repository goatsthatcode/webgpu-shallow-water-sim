import renderWGSL from './shaders/render.wgsl?raw';
import tracerWGSL from './shaders/tracer.wgsl?raw';
import { createLaxFriedrichs } from './schemes/lax-friedrichs.js';
import { createRK4 }           from './schemes/rk4.js';

const GRID_NX = 256;
const GRID_NY = 256;
const WG_SIZE = 8;
const DT      = 0.4;

// Params buffer layout (shared by all scheme shaders — must stay in sync):
//   [0]  width u32   [1]  height u32  [2]  dx f32    [3]  dt f32
//   [4]  step u32    [5]  f f32       [6]  H f32(H1)  [7]  nu4 f32
//   [8]  A_eq f32    [9]  tau f32     [10] beta f32   [11] gain f32
//   [12] mode u32    [13] H2 f32      [14] g_prime f32 [15] layerMode u32
const PARAMS_BYTES       = 80; // 20 slots; [16] = q_sat (render only)
const TRACER_PARAMS_BYTES = 16;

const SCHEMES = {
  'lax-friedrichs': createLaxFriedrichs,
  'rk4':            createRK4,
};

export function createSimulation(device, canvasFormat, initialScheme = 'lax-friedrichs') {
  const C = GPUShaderStage.COMPUTE;

  // ── Shared SWE state buffers ─────────────────────────────────────────────
  const cellCount  = GRID_NX * GRID_NY;
  const fieldBytes = cellCount * 6 * 4; // [h1 | u1 | v1 | h2 | u2 | v2]

  const bufs = [0, 1].map(i => device.createBuffer({
    label: `state-${i}`, size: fieldBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  }));

  const paramsBuf = device.createBuffer({
    label: 'params', size: PARAMS_BYTES,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // ── Tracer buffers ────────────────────────────────────────────────────────
  const tracerBytes = cellCount * 4; // one f32 per cell

  const tracerBufs = [0, 1].map(i => device.createBuffer({
    label: `tracer-${i}`, size: tracerBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  }));

  const tracerParamsBuf = device.createBuffer({
    label: 'tracer-params', size: TRACER_PARAMS_BYTES,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // ── Mutable physics state ────────────────────────────────────────────────
  let step      = 0;
  let f         = 0.0;
  let H         = 0.25;
  let H2        = 0.25;
  let g_prime   = 0.02;
  let sigma     = 0.10;
  let nu4       = 0.01;
  let A_eq      = 0.3;
  let tau       = 300.0;
  let beta      = 0.0;
  let gain      = 5.0;
  let mode      = 0;
  let layerMode = 1; // 1 = two-layer (default), 0 = single-layer

  let e_evap = 0.005;
  let k_prec = 2.0;
  let q_sat  = 1.0;

  const paramsScratch = new ArrayBuffer(PARAMS_BYTES);
  const paramsU32     = new Uint32Array(paramsScratch);
  const paramsF32     = new Float32Array(paramsScratch);
  paramsU32[0] = GRID_NX;
  paramsU32[1] = GRID_NY;
  paramsF32[2] = 1.0; // dx
  paramsF32[3] = DT;

  const tracerScratch = new Float32Array(4);

  function uploadTracerParams() {
    tracerScratch[0] = e_evap;
    tracerScratch[1] = k_prec;
    tracerScratch[2] = q_sat;
    tracerScratch[3] = 0;
    device.queue.writeBuffer(tracerParamsBuf, 0, tracerScratch);
  }
  uploadTracerParams();

  // ── Render pipeline (scheme-agnostic) ─────────────────────────────────────
  const renderModule   = device.createShaderModule({ label: 'render', code: renderWGSL });
  const renderPipeline = device.createRenderPipeline({
    label: 'render', layout: 'auto',
    vertex:    { module: renderModule, entryPoint: 'vs' },
    fragment:  { module: renderModule, entryPoint: 'fs', targets: [{ format: canvasFormat }] },
    primitive: { topology: 'triangle-list' },
  });
  const renderLayout = renderPipeline.getBindGroupLayout(0);

  // renderBG[sweBuf][tracerBuf] — 4 combinations for independent ping-pong
  const renderBG = [0, 1].map(s => [0, 1].map(t => device.createBindGroup({
    label: `render-s${s}-t${t}`, layout: renderLayout, entries: [
      { binding: 0, resource: { buffer: paramsBuf } },
      { binding: 1, resource: { buffer: bufs[s] } },
      { binding: 2, resource: { buffer: tracerBufs[t] } },
    ],
  })));

  // ── Tracer compute pipeline ──────────────────────────────────────────────
  const tracerBGL = device.createBindGroupLayout({ entries: [
    { binding: 0, visibility: C, buffer: { type: 'uniform' } },
    { binding: 1, visibility: C, buffer: { type: 'read-only-storage' } },
    { binding: 2, visibility: C, buffer: { type: 'read-only-storage' } },
    { binding: 3, visibility: C, buffer: { type: 'storage' } },
    { binding: 4, visibility: C, buffer: { type: 'uniform' } },
  ]});

  const tracerPipeline = device.createComputePipeline({
    label: 'tracer',
    layout: device.createPipelineLayout({ bindGroupLayouts: [tracerBGL] }),
    compute: {
      module: device.createShaderModule({ label: 'tracer', code: tracerWGSL }),
      entryPoint: 'cs_main',
    },
  });

  // tracerBGs[tracerCur][sweCur]: reads q from tracerBufs[tc], velocity from bufs[sc],
  // writes q to tracerBufs[1-tc].
  const tracerBGs = [0, 1].map(tc => [0, 1].map(sc => device.createBindGroup({
    label: `tracer-tc${tc}-sc${sc}`, layout: tracerBGL, entries: [
      { binding: 0, resource: { buffer: paramsBuf } },
      { binding: 1, resource: { buffer: bufs[sc] } },
      { binding: 2, resource: { buffer: tracerBufs[tc] } },
      { binding: 3, resource: { buffer: tracerBufs[1 - tc] } },
      { binding: 4, resource: { buffer: tracerParamsBuf } },
    ],
  })));

  let tracerCur = 0;

  function tickTracer(encoder, sweCur) {
    const pass = encoder.beginComputePass({ label: 'tracer' });
    pass.setPipeline(tracerPipeline);
    pass.setBindGroup(0, tracerBGs[tracerCur][sweCur]);
    pass.dispatchWorkgroups(GRID_NX / WG_SIZE, GRID_NY / WG_SIZE);
    pass.end();
    tracerCur = 1 - tracerCur;
  }

  // ── IC ─────────────────────────────────────────────────────────────────────
  function uploadIC() {
    device.queue.writeBuffer(bufs[0], 0, makeIC(GRID_NX, GRID_NY, sigma));
    device.queue.writeBuffer(bufs[1], 0, new Float32Array(cellCount * 6));
  }

  function resetTracer() {
    device.queue.writeBuffer(tracerBufs[0], 0, new Float32Array(cellCount));
    device.queue.writeBuffer(tracerBufs[1], 0, new Float32Array(cellCount));
    tracerCur = 0;
  }

  uploadIC();
  resetTracer();

  // ── Active SWE scheme ─────────────────────────────────────────────────────
  let scheme = SCHEMES[initialScheme](device, bufs, paramsBuf, GRID_NX, GRID_NY);

  // ── Tick / render ─────────────────────────────────────────────────────────
  function tick(encoder, substeps = 1) {
    paramsU32[4]  = step;
    paramsF32[5]  = f;
    paramsF32[6]  = H;
    paramsF32[7]  = nu4;
    paramsF32[8]  = A_eq;
    paramsF32[9]  = tau;
    paramsF32[10] = beta;
    paramsF32[11] = gain;
    paramsU32[12] = mode;
    paramsF32[13] = H2;
    paramsF32[14] = g_prime;
    paramsU32[15] = layerMode;
    paramsF32[16] = q_sat;
    device.queue.writeBuffer(paramsBuf, 0, paramsScratch);

    // Run SWE + tracer one substep at a time so the tracer stays synchronized
    // with the velocity field and the per-step CFL condition is respected.
    for (let s = 0; s < substeps; s++) {
      scheme.tick(encoder, 1);
      tickTracer(encoder, scheme.readBuf);
    }
    step += substeps;
  }

  function render(encoder, view) {
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view,
        clearValue: { r: 0.03, g: 0.18, b: 0.42, a: 1.0 },
        loadOp: 'clear', storeOp: 'store',
      }],
    });
    pass.setPipeline(renderPipeline);
    pass.setBindGroup(0, renderBG[scheme.readBuf][tracerCur]);
    pass.draw(6);
    pass.end();
  }

  function reset() {
    step = 0;
    uploadIC();
    scheme.setCur(0);
    resetTracer();
  }

  function setScheme(name) {
    const prevCur = scheme.readBuf;
    scheme.destroy();
    scheme = SCHEMES[name](device, bufs, paramsBuf, GRID_NX, GRID_NY);
    scheme.setCur(prevCur);
  }

  return {
    tick, render, reset, setScheme,
    setF(v)          { f = v; },
    setH(v)          { H = v; },
    setH2(v)         { H2 = v; },
    setGPrime(v)     { g_prime = v; },
    setSigma(v)      { sigma = v; },
    setNu4(v)        { nu4 = v; },
    setAEq(v)        { A_eq = v; },
    setTau(v)        { tau = v; },
    setBeta(v)       { beta = v; },
    setGain(v)       { gain = v; },
    setMode(v)       { mode = v; },
    setLayerMode(v)  { layerMode = v; },
    setEEvap(v)      { e_evap = v; uploadTracerParams(); },
    setKPrec(v)      { k_prec = v; uploadTracerParams(); },
    setQSat(v)       { q_sat  = v; uploadTracerParams(); },
    get step()       { return step; },
    get simTime()    { return step * DT; },
    get f()          { return f; },
    get H()          { return H; },
    get H2()         { return H2; },
    get g_prime()    { return g_prime; },
    get sigma()      { return sigma; },
    get nu4()        { return nu4; },
    get A_eq()       { return A_eq; },
    get tau()        { return tau; },
    get beta()       { return beta; },
    get gain()       { return gain; },
    get mode()       { return mode; },
    get layerMode()  { return layerMode; },
    get e_evap()     { return e_evap; },
    get k_prec()     { return k_prec; },
    get q_sat()      { return q_sat; },
  };
}

function makeIC(nx, ny, sigma) {
  const data = new Float32Array(nx * ny * 6);
  const cx = nx / 2, cy = ny / 2;
  const sx = sigma * nx, sy = sigma * ny;
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const dx = (i - cx) / sx;
      const dy = (j - cy) / sy;
      // Gaussian perturbation in upper layer h1 only; h2, velocities = 0
      data[j * nx + i] = Math.exp(-(dx * dx + dy * dy));
    }
  }
  return data;
}
