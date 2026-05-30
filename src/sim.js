import renderWGSL from './shaders/render.wgsl?raw';
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
//   [12] mode u32    [13] H2 f32      [14] g_prime f32 [15] pad
const PARAMS_BYTES = 64;

const SCHEMES = {
  'lax-friedrichs': createLaxFriedrichs,
  'rk4':            createRK4,
};

export function createSimulation(device, canvasFormat, initialScheme = 'lax-friedrichs') {
  // ── Shared state buffers ─────────────────────────────────────────────────
  // Both scheme types use the same two ping-pong state buffers.
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

  // ── Render pipeline (scheme-agnostic) ────────────────────────────────────
  const renderModule = device.createShaderModule({ label: 'render', code: renderWGSL });
  const renderPipeline = device.createRenderPipeline({
    label: 'render', layout: 'auto',
    vertex:    { module: renderModule, entryPoint: 'vs' },
    fragment:  { module: renderModule, entryPoint: 'fs', targets: [{ format: canvasFormat }] },
    primitive: { topology: 'triangle-list' },
  });
  const renderLayout = renderPipeline.getBindGroupLayout(0);
  const renderBG = bufs.map((buf, i) => device.createBindGroup({
    label: `render-${i}`, layout: renderLayout, entries: [
      { binding: 0, resource: { buffer: paramsBuf } },
      { binding: 1, resource: { buffer: buf } },
    ],
  }));

  // ── Mutable physics state ────────────────────────────────────────────────
  let step    = 0;
  let f       = 0.0;
  let H       = 0.25;
  let H2      = 0.25;
  let g_prime = 0.02;
  let sigma   = 0.10;
  let nu4     = 0.01;
  let A_eq    = 0.3;
  let tau     = 300.0;
  let beta    = 0.0;
  let gain    = 5.0;
  let mode    = 0;

  const paramsScratch = new ArrayBuffer(PARAMS_BYTES);
  const paramsU32     = new Uint32Array(paramsScratch);
  const paramsF32     = new Float32Array(paramsScratch);
  paramsU32[0] = GRID_NX;
  paramsU32[1] = GRID_NY;
  paramsF32[2] = 1.0; // dx
  paramsF32[3] = DT;

  function uploadIC() {
    device.queue.writeBuffer(bufs[0], 0, makeIC(GRID_NX, GRID_NY, sigma));
    device.queue.writeBuffer(bufs[1], 0, new Float32Array(cellCount * 6));
  }
  uploadIC();

  // ── Active scheme ────────────────────────────────────────────────────────
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
    device.queue.writeBuffer(paramsBuf, 0, paramsScratch);
    scheme.tick(encoder, substeps);
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
    pass.setBindGroup(0, renderBG[scheme.readBuf]);
    pass.draw(6);
    pass.end();
  }

  function reset() {
    step = 0;
    uploadIC();
    scheme.setCur(0); // IC is in bufs[0]
  }

  function setScheme(name) {
    const prevCur = scheme.readBuf;
    scheme.destroy();
    scheme = SCHEMES[name](device, bufs, paramsBuf, GRID_NX, GRID_NY);
    scheme.setCur(prevCur); // preserve current state buffer
  }

  return {
    tick, render, reset, setScheme,
    setF(v)       { f = v; },
    setH(v)       { H = v; },
    setH2(v)      { H2 = v; },
    setGPrime(v)  { g_prime = v; },
    setSigma(v)   { sigma = v; },
    setNu4(v)     { nu4 = v; },
    setAEq(v)     { A_eq = v; },
    setTau(v)     { tau = v; },
    setBeta(v)    { beta = v; },
    setGain(v)    { gain = v; },
    setMode(v)    { mode = v; },
    get step()    { return step; },
    get simTime() { return step * DT; },
    get f()       { return f; },
    get H()       { return H; },
    get H2()      { return H2; },
    get g_prime() { return g_prime; },
    get sigma()   { return sigma; },
    get nu4()     { return nu4; },
    get A_eq()    { return A_eq; },
    get tau()     { return tau; },
    get beta()    { return beta; },
    get gain()    { return gain; },
    get mode()    { return mode; },
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
      // Gaussian perturbation in upper layer h1 only; h2, u1, v1, u2, v2 = 0
      data[j * nx + i] = Math.exp(-(dx * dx + dy * dy));
    }
  }
  return data;
}
