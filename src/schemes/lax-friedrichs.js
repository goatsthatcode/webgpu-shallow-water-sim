import computeWGSL from '../shaders/compute.wgsl?raw';

const WG = 8;

export function createLaxFriedrichs(device, bufs, paramsBuf, nx, ny) {
  const pipeline = device.createComputePipeline({
    label: 'lf', layout: 'auto',
    compute: {
      module: device.createShaderModule({ label: 'lf-compute', code: computeWGSL }),
      entryPoint: 'cs_main',
    },
  });
  const layout = pipeline.getBindGroupLayout(0);

  const bg = [
    device.createBindGroup({ label: 'lf A→B', layout, entries: [
      { binding: 0, resource: { buffer: paramsBuf } },
      { binding: 1, resource: { buffer: bufs[0] } },
      { binding: 2, resource: { buffer: bufs[1] } },
    ]}),
    device.createBindGroup({ label: 'lf B→A', layout, entries: [
      { binding: 0, resource: { buffer: paramsBuf } },
      { binding: 1, resource: { buffer: bufs[1] } },
      { binding: 2, resource: { buffer: bufs[0] } },
    ]}),
  ];

  let cur = 0;

  return {
    tick(encoder, substeps) {
      const pass = encoder.beginComputePass({ label: 'lf' });
      pass.setPipeline(pipeline);
      for (let s = 0; s < substeps; s++) {
        pass.setBindGroup(0, bg[cur]);
        pass.dispatchWorkgroups(nx / WG, ny / WG);
        cur = 1 - cur;
      }
      pass.end();
    },
    get readBuf() { return cur; },
    setCur(c)    { cur = c; },
    destroy()    {},
  };
}
