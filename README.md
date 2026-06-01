# WebGPU Shallow Water

A browser-based interactive fluid simulator built on the linearized shallow water equations (SWE). Runs entirely on the GPU via WebGPU compute shaders — no server, no dependencies beyond Vite.

**Requires:** Chrome 113+ or Safari 18+ (WebGPU support).

```bash
npm install
npm run dev
```

---

## What it simulates

The **linearized shallow water equations** on a doubly-periodic flat domain (a mathematical torus — think of a global atmosphere with no poles or boundaries):

```
∂h/∂t = −H(∂u/∂x + ∂v/∂y)  −  (h − h_eq)/τ  −  ν₄∇⁴h
∂u/∂t = −g∂h/∂x  +  f_eff·v  −  ν₄∇⁴u
∂v/∂t = −g∂h/∂y  −  f_eff·u  −  ν₄∇⁴v
```

where:
- `h(x,y,t)` — height perturbation around mean depth H (proxy for geopotential / temperature via the hypsometric equation)
- `u, v` — depth-averaged horizontal velocity components
- `g = 1` — non-dimensionalized gravity
- `f_eff(j) = f₀ + β·(j − ny/2)` — effective Coriolis parameter (beta-plane)
- `h_eq(j) = −A_eq·cos(2π·j/ny)` — thermal equilibrium profile (warm equator at j = ny/2, cold poles at j = 0, ny)
- `ν₄` — hyperdiffusion coefficient (∇⁴, RK4 only)

The **gravity wave speed** is `c = √(g·H)`. The **Rossby deformation radius** is `L_R = c/f`, the scale at which rotation and gravity waves are in balance.

---

## Grid and numerics

| Property | Value |
|---|---|
| Grid | 256 × 256, doubly-periodic |
| Grid spacing | dx = dy = 1 (non-dimensional) |
| Time step | dt = 0.4 |
| State layout | `[h block | u block | v block]` packed sequentially in GPU storage buffers |
| Ping-pong | Two state buffers; scheme alternates read/write each step |

### Numerical schemes

**Lax-Friedrichs (1st order)**

Replaces the cell-center value with its spatial average before subtracting the flux divergence:

```
h_new = ¼(h_r + h_l + h_u + h_d) − (dt/2dx)·H·(u_r − u_l + v_u − v_d)
```

The averaging is equivalent to baking in numerical diffusion of `dx²/(4dt)·∇²` at every step. This keeps the scheme stable for the purely imaginary eigenvalues of the SWE (unlike RK2/forward Euler, which are only stable on the real axis). The tradeoff is that LF diffusion dominates over any physical diffusion at short wavelengths and smears sharp features.

Because LF sits exactly at `|G(k=π)| = 1` at the Nyquist frequency, adding any explicit hyperdiffusion (ν₄) would push `|G| > 1` and immediately blow up. ν₄ is silently ignored in LF mode.

Stability limit: `c·dt·(1/dx + 1/dy) ≤ 1` → safe for H ≤ 1.56 with dx = 1, dt = 0.4. UI caps H at 1.0.

**RK4 (4th order)**

Classical 4-stage Runge-Kutta applied to the centered-difference SWE tendency. No spatial averaging — no built-in numerical diffusion. RK4's stability region covers the imaginary axis up to `|λ·dt| ≤ 2√2 ≈ 2.83`. For 2D SWE this gives `c·π√2·dt ≤ 2√2`, or `c·π·dt ≤ 2`, which holds safely for all H ≤ 1 with dt = 0.4 (worst case: `c·π·dt ≈ 1.26`).

Each step uses 8 compute passes (k1 → stage → k2 → stage → k3 → stage → k4 → finalize) and three temporary GPU buffers (k, stage, accumulator). Use RK4 for any run where small-scale structure matters: jets, Rossby waves, geostrophic eddies.

**Why not RK2 (Heun's)?**

Heun's stability region covers only the real axis. SWE eigenvalues are purely imaginary `±ic|k|`, so RK2 is unconditionally unstable for the SWE regardless of time step size.

**Why not implicit schemes?**

Fully implicit methods require solving a large sparse linear system each step — straightforward on a CPU but awkward on a GPU, which has no native sparse solver. IMEX splitting (implicit only for the stiff part) is feasible but adds significant complexity. For our linearized problem with controllable wave speed, explicit RK4 is stable and efficient.

---

## Controls reference

### Row 1 — Playback

| Control | Range | Default | Notes |
|---|---|---|---|
| pause / play | — | playing | |
| speed | 1–30 steps/frame | 2 | Higher = faster simulation clock |
| reset | — | — | Reloads Gaussian IC; σ change takes effect here |
| scheme | LF / RK4 | LF | State is preserved when switching |

### Row 2 — Physics

| Control | Range | Default | Notes |
|---|---|---|---|
| σ (IC width) | 0.03–0.20 | 0.10 | Gaussian half-width as fraction of domain; takes effect on reset |
| H (mean depth) | 0.02–1.0 | 0.25 | Wave speed `c = √H`; display shows c |
| f (Coriolis) | 0–0.20 | 0 | f-plane base Coriolis; display shows L_R = c/f |

### Row 3 — Display

| Control | Options / Range | Default | Notes |
|---|---|---|---|
| display | h anomaly / vorticity / raw h | h anomaly | See below |
| gain | 1–40 | 5 | Colormap amplification; increase to see low-amplitude features |

**Display modes:**
- **h anomaly** — `(h − h_eq)·gain`. Subtracts the thermal background so only the dynamical deviation is visible. The right default for forced runs.
- **vorticity** — `(∂v/∂x − ∂u/∂y)·gain`. Relative vorticity ζ, computed with centered differences directly in the fragment shader. The clearest view for Rossby waves and geostrophic eddies: cyclones (negative ζ) appear indigo, anticyclones (positive ζ) appear seafoam.
- **raw h** — `h·gain`. Unmodified height field. Useful for inspecting the unforced gravity wave problem or verifying that the thermal gradient is building correctly.

### Row 4 — Forcing / dissipation

| Control | Range | Default | Notes |
|---|---|---|---|
| ν₄ (hyperdiff) | 0–0.10 | 0.01 | ∇⁴ hyperdiffusion; **RK4 only**. 13-point biharmonic stencil, k⁴ damping — kills grid-scale noise while leaving large-scale dynamics untouched |
| A_eq (thermal) | 0–1.0 | 0.30 | Amplitude of the h_eq equilibrium profile. 0 = no thermal forcing |
| τ (relax) | 20–2000 | 300 | Relaxation timescale in time units. Smaller = stronger nudging toward h_eq |
| β (beta) | 0–0.05 | 0 | Beta-plane parameter. Required for Rossby waves |

---

## Parameter recipes

### 1 — Pure gravity waves

The unforced, non-rotating SWE. A Gaussian bump radiates an outward ring; with periodic BCs the ring wraps and interferes with itself.

```
scheme: Lax-Friedrichs or RK4
H:      0.25    (c = 0.50)
f:      0
A_eq:   0
β:      0
display: raw h, gain 3
speed:  2–5
```

Run LF and RK4 back-to-back to see the effect of numerical diffusion: the ring stays sharp in RK4 and visibly smears in LF.

### 2 — Geostrophic adjustment (f-plane)

With Coriolis active, the gravity wave ring radiates outward but a geostrophically balanced vortex is left behind at the origin. Its size is the Rossby deformation radius L_R = c/f. Varying f lets you directly observe the scale selection.

```
scheme: RK4
H:      0.25    (c = 0.50)
f:      0.05    (L_R ≈ 10 grid cells)
A_eq:   0
β:      0
display: vorticity, gain 8
speed:  5
```

Try f = 0.10 (L_R ≈ 5, tight compact vortex) and f = 0.02 (L_R ≈ 25, vortex nearly fills domain).

### 3 — Thermal jets

Newtonian relaxation drives h toward h_eq (warm equator, cold poles). With Coriolis, the meridional h gradient is in geostrophic balance with a zonal wind — the **thermal wind relation**: `f·∂u/∂z ∝ −∂T/∂y`. Zonal jets emerge near the latitude bands of maximum meridional gradient (the ±45° analogs at j ≈ 64 and j ≈ 192).

Takes hundreds of time units to spin up; use speed 10–15.

```
scheme: RK4
H:      0.25
f:      0.05–0.08
A_eq:   0.40
τ:      200–300
β:      0
ν₄:     0.01
display: vorticity, gain 15–20
speed:  10–15
```

Run for ~2000–5000 time units (a few minutes at speed 10). Zonal banding should emerge in the vorticity field.

### 4 — Rossby waves

Rossby waves are large-scale, low-frequency waves that exist because of the planetary vorticity gradient β = ∂f/∂y. Their dispersion relation is:

```
ω = −β·kx / (kx² + ky² + L_R⁻²)
```

Phase propagates **westward** at all wavelengths; longer waves propagate faster. They appear as a meandering wave train along the jet latitudes.

Start from the thermal jet state and add β once jets are established:

```
scheme: RK4
H:      0.25
f:      0.05
A_eq:   0.40
τ:      200
β:      0.010–0.020
ν₄:     0.01
display: vorticity, gain 20–30
speed:  10
```

After ~1000–2000 t, the jets go wavy. Look for trains of cyclone/anticyclone pairs propagating westward in the vorticity display. If β is too small relative to the jet's vorticity curvature, the flow is barotropically unstable (Rayleigh-Kuo criterion) and the waves amplify instead of propagate — this is also interesting to watch.

---

## Colormap

All display modes share the same diverging colormap:

```
negative → deep indigo  (depressed surface / cyclonic)
zero     → ocean blue   (mean state)
positive → bright seafoam  (raised surface / anticyclonic)
```

---

## Architecture

```
src/
  main.js              UI wiring, WebGPU init, RAF loop
  sim.js               Scheme-agnostic core: buffers, params uniform, tick/render/reset
  schemes/
    lax-friedrichs.js  LF: 1 compute pipeline, 2 bind groups, no temp buffers
    rk4.js             RK4: 5 pipelines, 3 temp buffers, 8 compute passes per step
  shaders/
    compute.wgsl       Lax-Friedrichs full step (flux + Coriolis + relaxation)
    rhs.wgsl           Centered-difference SWE tendency for RK4
    blend.wgsl         RK4 stage blending / accumulation (4 entry points, 1D dispatch)
    render.wgsl        Fullscreen quad + colormap (3 display modes)
```

The params uniform buffer (64 bytes) is shared by all shaders:

| index | type | field |
|---|---|---|
| [0] | u32 | width |
| [1] | u32 | height |
| [2] | f32 | dx |
| [3] | f32 | dt |
| [4] | u32 | step |
| [5] | f32 | f (base Coriolis) |
| [6] | f32 | H (mean depth) |
| [7] | f32 | ν₄ (hyperdiffusion, RK4 only) |
| [8] | f32 | A_eq (thermal forcing amplitude) |
| [9] | f32 | τ (relaxation timescale) |
| [10] | f32 | β (beta-plane) |
| [11] | f32 | gain (colormap scale) |
| [12] | u32 | mode (display: 0=h anomaly, 1=vorticity, 2=raw h) |
| [13–15] | — | padding |

Schemes are modular: `createLaxFriedrichs` and `createRK4` each return `{ tick, readBuf, setCur, destroy }`. Switching schemes at runtime preserves the current state buffer. Adding new schemes (leapfrog, Adams-Bashforth, IMEX) requires only a new file in `src/schemes/`.

---

## Roadmap

- [x] Stage 0–2 — WebGPU bootstrap, ping-pong buffers, periodic BCs
- [x] Stage 3 — Linearized SWE, gravity waves, Lax-Friedrichs
- [x] Stage 4 — Coriolis, geostrophic adjustment, Rossby radius display
- [x] Stage 5 — RK4, thermal forcing, beta-plane, hyperdiffusion, vorticity display
- [x] Stage 6 — Two-layer SWE (baroclinic instability, storm tracks)
- [x] Stage 7 — Tracer advection + precipitation
- [ ] Stage 8 — Sphere projection (globe visualization)
- [ ] Stage 9+ — Interactive forcing, topography, primitive equations
