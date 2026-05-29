import { createSimulation } from './sim.js';

const status = document.getElementById('status');

async function main() {
  if (!navigator.gpu) {
    status.textContent = 'WebGPU not supported — use Chrome 113+ or Safari 18+';
    return;
  }

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    status.textContent = 'No GPU adapter found';
    return;
  }

  const device = await adapter.requestDevice();
  const canvas = document.getElementById('canvas');
  const context = canvas.getContext('webgpu');
  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: 'opaque' });

  const sim = createSimulation(device, format);

  function loop() {
    const encoder = device.createCommandEncoder({ label: 'frame' });
    sim.step(encoder);
    sim.render(encoder, context.getCurrentTexture().createView());
    device.queue.submit([encoder.finish()]);
    sim.advance();
    requestAnimationFrame(loop);
  }

  requestAnimationFrame(loop);

  const info = adapter.info ?? (await adapter.requestAdapterInfo?.()) ?? {};
  status.textContent = `adapter: ${info.description || info.vendor || 'gpu'} | grid 256×256 | ping-pong active`;
}

main().catch(err => {
  console.error(err);
  status.textContent = `Error: ${err.message}`;
});
