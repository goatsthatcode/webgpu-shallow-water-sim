# webgpu-shallow-water

A browser-based shallow water simulation built on WebGPU compute shaders.
The long-term goal is a toy aquaplanet / weather simulator that's playable
in the browser, growing one stage at a time from a single scalar field to
multi-layer SWE with tracers.

## Run it

```bash
npm install
npm run dev
```

Open the URL Vite prints (usually `http://localhost:5173`).

Requires Chrome 113+ or Safari 18+ (with WebGPU enabled in
**Develop → Feature Flags**).

## Architecture

- `src/main.js` — WebGPU init, frame loop
- `src/sim.js` — buffers, pipelines, bind groups
- `src/shaders/compute.wgsl` — simulation step (compute shader)
- `src/shaders/render.wgsl` — fullscreen quad + colormap

Each frame runs a **compute pass** that updates one of two ping-pong
storage buffers, then a **render pass** that reads the just-written buffer
and colormaps it directly to the canvas. No CPU readback.

## Stages

- [x] **0** — WebGPU bootstrap (hello triangle)
- [x] **1** — Fullscreen quad + ping-pong compute buffers (identity copy)
- [ ] **2** — Periodic boundary conditions + diffusion sanity check
- [ ] **3** — Linearized shallow water (gravity waves)
- [ ] **4** — Coriolis force
- [ ] **5** — Forcing + dissipation
- [ ] **6** — Multi-layer SWE
- [ ] **7** — Tracer advection + precipitation
- [ ] **8** — Sphere projection
