import computeWGSL from './shaders/compute.wgsl?raw';
import renderWGSL  from './shaders/render.wgsl?raw';

const GRID_NX = 256;
const GRID_NY = 256;
const WG_SIZE = 8;

// Params struct layout (matches WGSL): width u32, height u32, dx f32, dt f32, frame u32 + pad.
const PARAMS_BYTES = 32;

export function createSimulation(device, canvasFormat) {
  // ── Buffers ─────────────────────────────────────────────────────────────
  const cellCount = GRID_NX * GRID_NY;
  const fieldBytes = cellCount * 4;

  const bufA = device.createBuffer({
    label: 'field-A',
    size: fieldBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });
  const bufB = device.createBuffer({
    label: 'field-B',
    size: fieldBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });

  const paramsBuf = device.createBuffer({
    label: 'params',
    size: PARAMS_BYTES,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // Initial condition: Gaussian bump at the center of A.
  device.queue.writeBuffer(bufA, 0, makeGaussianIC(GRID_NX, GRID_NY, 0.04));

  // ── Pipelines ───────────────────────────────────────────────────────────
  const computeModule = device.createShaderModule({ label: 'compute', code: computeWGSL });
  const renderModule  = device.createShaderModule({ label: 'render',  code: renderWGSL });

  const computePipeline = device.createComputePipeline({
    label: 'compute',
    layout: 'auto',
    compute: { module: computeModule, entryPoint: 'cs_main' },
  });

  const renderPipeline = device.createRenderPipeline({
    label: 'render',
    layout: 'auto',
    vertex:   { module: renderModule, entryPoint: 'vs' },
    fragment: { module: renderModule, entryPoint: 'fs', targets: [{ format: canvasFormat }] },
    primitive: { topology: 'triangle-list' },
  });

  // ── Bind groups ─────────────────────────────────────────────────────────
  // Two compute bind groups (A→B and B→A); two render bind groups (read A, read B).
  const computeLayout = computePipeline.getBindGroupLayout(0);
  const renderLayout  = renderPipeline.getBindGroupLayout(0);

  const computeBG = [
    device.createBindGroup({
      label: 'compute A→B',
      layout: computeLayout,
      entries: [
        { binding: 0, resource: { buffer: paramsBuf } },
        { binding: 1, resource: { buffer: bufA } },
        { binding: 2, resource: { buffer: bufB } },
      ],
    }),
    device.createBindGroup({
      label: 'compute B→A',
      layout: computeLayout,
      entries: [
        { binding: 0, resource: { buffer: paramsBuf } },
        { binding: 1, resource: { buffer: bufB } },
        { binding: 2, resource: { buffer: bufA } },
      ],
    }),
  ];

  const renderBG = [
    device.createBindGroup({
      label: 'render read-A',
      layout: renderLayout,
      entries: [
        { binding: 0, resource: { buffer: paramsBuf } },
        { binding: 1, resource: { buffer: bufA } },
      ],
    }),
    device.createBindGroup({
      label: 'render read-B',
      layout: renderLayout,
      entries: [
        { binding: 0, resource: { buffer: paramsBuf } },
        { binding: 1, resource: { buffer: bufB } },
      ],
    }),
  ];

  // ── State ───────────────────────────────────────────────────────────────
  let frame = 0;
  const paramsScratch = new ArrayBuffer(PARAMS_BYTES);
  const paramsU32 = new Uint32Array(paramsScratch);
  const paramsF32 = new Float32Array(paramsScratch);
  paramsU32[0] = GRID_NX;
  paramsU32[1] = GRID_NY;
  paramsF32[2] = 1.0;  // dx (placeholder until SWE stage)
  paramsF32[3] = 0.01; // dt (placeholder)

  function step(encoder) {
    paramsU32[4] = frame;
    device.queue.writeBuffer(paramsBuf, 0, paramsScratch);

    const pass = encoder.beginComputePass({ label: 'sim-step' });
    pass.setPipeline(computePipeline);
    pass.setBindGroup(0, computeBG[frame % 2]);
    pass.dispatchWorkgroups(GRID_NX / WG_SIZE, GRID_NY / WG_SIZE);
    pass.end();
  }

  function render(encoder, view) {
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view,
        clearValue: { r: 0.04, g: 0.04, b: 0.10, a: 1.0 },
        loadOp:  'clear',
        storeOp: 'store',
      }],
    });
    pass.setPipeline(renderPipeline);
    // After step N, the freshly-written buffer is B if frame was even, A if odd.
    pass.setBindGroup(0, renderBG[(frame + 1) % 2]);
    pass.draw(6);
    pass.end();
  }

  function advance() { frame += 1; }

  return { step, render, advance, get frame() { return frame; } };
}

function makeGaussianIC(nx, ny, sigma) {
  const arr = new Float32Array(nx * ny);
  const cx = nx / 2;
  const cy = ny / 2;
  const sx = sigma * nx;
  const sy = sigma * ny;
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const dx = (i - cx) / sx;
      const dy = (j - cy) / sy;
      arr[j * nx + i] = Math.exp(-(dx * dx + dy * dy));
    }
  }
  return arr;
}
