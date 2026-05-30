import { createSimulation } from './sim.js';

const status      = document.getElementById('status');
const pauseBtn    = document.getElementById('pause-btn');
const resetBtn      = document.getElementById('reset-btn');
const schemeSelect  = document.getElementById('scheme-select');
const speedSlider = document.getElementById('speed-slider');
const speedVal    = document.getElementById('speed-val');
const sigmaSlider = document.getElementById('sigma-slider');
const sigmaVal    = document.getElementById('sigma-val');
const HSlider     = document.getElementById('H-slider');
const HVal        = document.getElementById('H-val');
const cVal        = document.getElementById('c-val');
const fSlider     = document.getElementById('f-slider');
const fVal        = document.getElementById('f-val');
const lrVal       = document.getElementById('lr-val');
const nu4Slider   = document.getElementById('nu4-slider');
const nu4Val      = document.getElementById('nu4-val');
const AeqSlider   = document.getElementById('Aeq-slider');
const AeqVal      = document.getElementById('Aeq-val');
const tauSlider   = document.getElementById('tau-slider');
const tauVal      = document.getElementById('tau-val');
const betaSlider    = document.getElementById('beta-slider');
const betaVal       = document.getElementById('beta-val');
const displaySelect = document.getElementById('display-select');
const gainSlider    = document.getElementById('gain-slider');
const gainVal       = document.getElementById('gain-val');

let paused   = false;
let substeps = parseInt(speedSlider.value);

function updateHDisplay(H) {
  HVal.textContent = H.toFixed(2);
  cVal.textContent = Math.sqrt(H).toFixed(2);
}

function updateFDisplay(f, H) {
  const c = Math.sqrt(H);
  fVal.textContent = f.toFixed(3);
  lrVal.textContent = f > 0 ? (c / f).toFixed(1) : '∞';
}

async function main() {
  if (!navigator.gpu) {
    status.textContent = 'WebGPU not supported — use Chrome 113+ or Safari 18+';
    return;
  }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) { status.textContent = 'No GPU adapter found'; return; }

  const device  = await adapter.requestDevice();
  const canvas  = document.getElementById('canvas');
  const context = canvas.getContext('webgpu');
  const format  = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: 'opaque' });

  const sim = createSimulation(device, format);

  const info     = adapter.info ?? (await adapter.requestAdapterInfo?.()) ?? {};
  const gpuLabel = info.description || info.vendor || 'gpu';

  // Init displays to match sim defaults
  sigmaVal.textContent = sim.sigma.toFixed(2);
  sigmaSlider.value    = sim.sigma;
  HSlider.value        = sim.H;
  updateHDisplay(sim.H);
  updateFDisplay(sim.f, sim.H);
  nu4Slider.value      = sim.nu4;
  nu4Val.textContent   = sim.nu4.toFixed(3);
  AeqSlider.value      = sim.A_eq;
  AeqVal.textContent   = sim.A_eq.toFixed(2);
  tauSlider.value      = sim.tau;
  tauVal.textContent   = sim.tau.toFixed(0);
  betaSlider.value     = sim.beta;
  betaVal.textContent  = sim.beta.toFixed(3);
  gainSlider.value     = sim.gain;
  gainVal.textContent  = sim.gain.toFixed(0);
  displaySelect.value  = sim.mode.toString();

  pauseBtn.addEventListener('click', () => {
    paused = !paused;
    pauseBtn.textContent = paused ? 'play' : 'pause';
  });

  speedSlider.addEventListener('input', () => {
    substeps = parseInt(speedSlider.value);
    speedVal.textContent = substeps;
  });

  // Sigma only affects the IC — auto-reset so the change is immediately visible.
  sigmaSlider.addEventListener('input', () => {
    const sigma = parseFloat(sigmaSlider.value);
    sigmaVal.textContent = sigma.toFixed(2);
    sim.setSigma(sigma);
    sim.reset();
  });

  HSlider.addEventListener('input', () => {
    const H = parseFloat(HSlider.value);
    sim.setH(H);
    updateHDisplay(H);
    updateFDisplay(parseFloat(fSlider.value), H);
  });

  fSlider.addEventListener('input', () => {
    const f = parseFloat(fSlider.value);
    sim.setF(f);
    updateFDisplay(f, parseFloat(HSlider.value));
  });

  nu4Slider.addEventListener('input', () => {
    const v = parseFloat(nu4Slider.value);
    sim.setNu4(v);
    nu4Val.textContent = v.toFixed(3);
  });

  AeqSlider.addEventListener('input', () => {
    const v = parseFloat(AeqSlider.value);
    sim.setAEq(v);
    AeqVal.textContent = v.toFixed(2);
  });

  tauSlider.addEventListener('input', () => {
    const v = parseFloat(tauSlider.value);
    sim.setTau(v);
    tauVal.textContent = v.toFixed(0);
  });

  betaSlider.addEventListener('input', () => {
    const v = parseFloat(betaSlider.value);
    sim.setBeta(v);
    betaVal.textContent = v.toFixed(3);
  });

  gainSlider.addEventListener('input', () => {
    const v = parseFloat(gainSlider.value);
    sim.setGain(v);
    gainVal.textContent = v.toFixed(0);
  });

  displaySelect.addEventListener('change', () => {
    sim.setMode(parseInt(displaySelect.value));
  });

  resetBtn.addEventListener('click', () => sim.reset());

  schemeSelect.addEventListener('change', () => {
    sim.setScheme(schemeSelect.value);
  });

  function loop() {
    if (!paused) {
      const encoder = device.createCommandEncoder({ label: 'frame' });
      sim.tick(encoder, substeps);
      sim.render(encoder, context.getCurrentTexture().createView());
      device.queue.submit([encoder.finish()]);
    }
    status.textContent = `${gpuLabel} | 256×256 | t = ${sim.simTime.toFixed(1)}`;
    requestAnimationFrame(loop);
  }

  requestAnimationFrame(loop);
}

main().catch(err => {
  console.error(err);
  status.textContent = `Error: ${err.message}`;
});
