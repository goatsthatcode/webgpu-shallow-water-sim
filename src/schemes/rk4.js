import rhsWGSL   from '../shaders/rhs.wgsl?raw';
import blendWGSL from '../shaders/blend.wgsl?raw';

const WG_RHS   = 8;
const WG_BLEND = 256;

export function createRK4(device, bufs, paramsBuf, nx, ny) {
  const fieldBytes  = nx * ny * 6 * 4;
  const totalCells  = nx * ny * 6;
  const blendGroups = Math.ceil(totalCells / WG_BLEND);

  // Temp buffers — only live while RK4 is the active scheme.
  const buf_k     = device.createBuffer({ label: 'rk4-k',     size: fieldBytes, usage: GPUBufferUsage.STORAGE });
  const buf_stage = device.createBuffer({ label: 'rk4-stage', size: fieldBytes, usage: GPUBufferUsage.STORAGE });
  const buf_acc   = device.createBuffer({ label: 'rk4-acc',   size: fieldBytes, usage: GPUBufferUsage.STORAGE });

  // ── Explicit bind group layouts ────────────────────────────────────────
  // Shared across all pipelines so bind groups can be reused between stages.
  const C = GPUShaderStage.COMPUTE;

  const rhsBGL = device.createBindGroupLayout({ entries: [
    { binding: 0, visibility: C, buffer: { type: 'uniform' } },
    { binding: 1, visibility: C, buffer: { type: 'read-only-storage' } },
    { binding: 2, visibility: C, buffer: { type: 'storage' } },
  ]});

  const blendBGL = device.createBindGroupLayout({ entries: [
    { binding: 0, visibility: C, buffer: { type: 'uniform' } },
    { binding: 1, visibility: C, buffer: { type: 'read-only-storage' } },
    { binding: 2, visibility: C, buffer: { type: 'read-only-storage' } },
    { binding: 3, visibility: C, buffer: { type: 'storage' } },
    { binding: 4, visibility: C, buffer: { type: 'storage' } },
  ]});

  const rhsLayout   = device.createPipelineLayout({ bindGroupLayouts: [rhsBGL] });
  const blendLayout = device.createPipelineLayout({ bindGroupLayouts: [blendBGL] });

  // ── Pipelines ────────────────────────────────────────────────────────
  const rhsMod   = device.createShaderModule({ label: 'rhs',   code: rhsWGSL });
  const blendMod = device.createShaderModule({ label: 'blend', code: blendWGSL });

  const rhsPipeline  = device.createComputePipeline({ label: 'rhs',  layout: rhsLayout,   compute: { module: rhsMod,   entryPoint: 'cs_main'    } });
  const s1Pipeline   = device.createComputePipeline({ label: 'rk4-s1', layout: blendLayout, compute: { module: blendMod, entryPoint: 'cs_stage1'  } });
  const s2Pipeline   = device.createComputePipeline({ label: 'rk4-s2', layout: blendLayout, compute: { module: blendMod, entryPoint: 'cs_stage2'  } });
  const s3Pipeline   = device.createComputePipeline({ label: 'rk4-s3', layout: blendLayout, compute: { module: blendMod, entryPoint: 'cs_stage3'  } });
  const finPipeline  = device.createComputePipeline({ label: 'rk4-fin', layout: blendLayout, compute: { module: blendMod, entryPoint: 'cs_finalize'} });

  // ── Bind groups ───────────────────────────────────────────────────────
  // RHS: one per state buffer (k1 reads current state; k2/k3/k4 read buf_stage).
  const rhs_bg = [
    device.createBindGroup({ label: 'rhs-0', layout: rhsBGL, entries: [
      { binding: 0, resource: { buffer: paramsBuf } },
      { binding: 1, resource: { buffer: bufs[0] } },
      { binding: 2, resource: { buffer: buf_k } },
    ]}),
    device.createBindGroup({ label: 'rhs-1', layout: rhsBGL, entries: [
      { binding: 0, resource: { buffer: paramsBuf } },
      { binding: 1, resource: { buffer: bufs[1] } },
      { binding: 2, resource: { buffer: buf_k } },
    ]}),
  ];
  const rhs_stage_bg = device.createBindGroup({ label: 'rhs-stage', layout: rhsBGL, entries: [
    { binding: 0, resource: { buffer: paramsBuf } },
    { binding: 1, resource: { buffer: buf_stage } },
    { binding: 2, resource: { buffer: buf_k } },
  ]});

  // Blend stages 1-3: buf_n varies with cur; buf_stage and buf_acc are always the same.
  const stage_bg = [
    device.createBindGroup({ label: 'blend-0', layout: blendBGL, entries: [
      { binding: 0, resource: { buffer: paramsBuf } },
      { binding: 1, resource: { buffer: bufs[0] } },
      { binding: 2, resource: { buffer: buf_k } },
      { binding: 3, resource: { buffer: buf_stage } },
      { binding: 4, resource: { buffer: buf_acc } },
    ]}),
    device.createBindGroup({ label: 'blend-1', layout: blendBGL, entries: [
      { binding: 0, resource: { buffer: paramsBuf } },
      { binding: 1, resource: { buffer: bufs[1] } },
      { binding: 2, resource: { buffer: buf_k } },
      { binding: 3, resource: { buffer: buf_stage } },
      { binding: 4, resource: { buffer: buf_acc } },
    ]}),
  ];

  // Finalize: reads uⁿ from bufs[cur], writes u^{n+1} to bufs[1-cur].
  const fin_bg = [
    device.createBindGroup({ label: 'fin-0', layout: blendBGL, entries: [
      { binding: 0, resource: { buffer: paramsBuf } },
      { binding: 1, resource: { buffer: bufs[0] } },
      { binding: 2, resource: { buffer: buf_k } },
      { binding: 3, resource: { buffer: bufs[1] } },
      { binding: 4, resource: { buffer: buf_acc } },
    ]}),
    device.createBindGroup({ label: 'fin-1', layout: blendBGL, entries: [
      { binding: 0, resource: { buffer: paramsBuf } },
      { binding: 1, resource: { buffer: bufs[1] } },
      { binding: 2, resource: { buffer: buf_k } },
      { binding: 3, resource: { buffer: bufs[0] } },
      { binding: 4, resource: { buffer: buf_acc } },
    ]}),
  ];

  let cur = 0;
  const d2 = [nx / WG_RHS, ny / WG_RHS]; // 2D dispatch for RHS passes

  function step(encoder) {
    const c = cur;

    // k1 = F(uⁿ)
    let p = encoder.beginComputePass({ label: 'k1' });
    p.setPipeline(rhsPipeline); p.setBindGroup(0, rhs_bg[c]);
    p.dispatchWorkgroups(d2[0], d2[1]); p.end();

    // stage = uⁿ + dt/2·k1;  acc = k1
    p = encoder.beginComputePass({ label: 's1' });
    p.setPipeline(s1Pipeline); p.setBindGroup(0, stage_bg[c]);
    p.dispatchWorkgroups(blendGroups); p.end();

    // k2 = F(stage)
    p = encoder.beginComputePass({ label: 'k2' });
    p.setPipeline(rhsPipeline); p.setBindGroup(0, rhs_stage_bg);
    p.dispatchWorkgroups(d2[0], d2[1]); p.end();

    // stage = uⁿ + dt/2·k2;  acc += 2·k2
    p = encoder.beginComputePass({ label: 's2' });
    p.setPipeline(s2Pipeline); p.setBindGroup(0, stage_bg[c]);
    p.dispatchWorkgroups(blendGroups); p.end();

    // k3 = F(stage)
    p = encoder.beginComputePass({ label: 'k3' });
    p.setPipeline(rhsPipeline); p.setBindGroup(0, rhs_stage_bg);
    p.dispatchWorkgroups(d2[0], d2[1]); p.end();

    // stage = uⁿ + dt·k3;  acc += 2·k3
    p = encoder.beginComputePass({ label: 's3' });
    p.setPipeline(s3Pipeline); p.setBindGroup(0, stage_bg[c]);
    p.dispatchWorkgroups(blendGroups); p.end();

    // k4 = F(stage)
    p = encoder.beginComputePass({ label: 'k4' });
    p.setPipeline(rhsPipeline); p.setBindGroup(0, rhs_stage_bg);
    p.dispatchWorkgroups(d2[0], d2[1]); p.end();

    // u^{n+1} = uⁿ + dt/6·(acc + k4)
    p = encoder.beginComputePass({ label: 'rk4-fin' });
    p.setPipeline(finPipeline); p.setBindGroup(0, fin_bg[c]);
    p.dispatchWorkgroups(blendGroups); p.end();

    cur = 1 - cur;
  }

  return {
    tick(encoder, substeps) {
      for (let s = 0; s < substeps; s++) step(encoder);
    },
    get readBuf() { return cur; },
    setCur(c)    { cur = c; },
    destroy() {
      buf_k.destroy();
      buf_stage.destroy();
      buf_acc.destroy();
    },
  };
}
