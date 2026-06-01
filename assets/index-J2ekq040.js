(function(){const a=document.createElement("link").relList;if(a&&a.supports&&a.supports("modulepreload"))return;for(const o of document.querySelectorAll('link[rel="modulepreload"]'))u(o);new MutationObserver(o=>{for(const n of o)if(n.type==="childList")for(const d of n.addedNodes)d.tagName==="LINK"&&d.rel==="modulepreload"&&u(d)}).observe(document,{childList:!0,subtree:!0});function s(o){const n={};return o.integrity&&(n.integrity=o.integrity),o.referrerPolicy&&(n.referrerPolicy=o.referrerPolicy),o.crossOrigin==="use-credentials"?n.credentials="include":o.crossOrigin==="anonymous"?n.credentials="omit":n.credentials="same-origin",n}function u(o){if(o.ep)return;o.ep=!0;const n=s(o);fetch(o.href,n)}})();const Qe=`// Fullscreen quad renderer. Seven display modes:
//   0 — h1 anomaly:            (h1 − h_eq)·gain
//   1 — vorticity layer 1:     ζ1 = (∂v1/∂x − ∂u1/∂y)·gain
//   2 — raw h1:                h1·gain
//   3 — h2 interface:          h2·gain
//   4 — barotropic vorticity:  (H·ζ1 + H2·ζ2)/(H+H2)·gain
//   5 — baroclinic vorticity:  (ζ1 − ζ2)·gain
//   6 — tracer q:              q·gain  (moisture)

struct Params {
  width:   u32,  // [0]
  height:  u32,  // [1]
  dx:      f32,  // [2]
  dt:      f32,  // [3]
  step:    u32,  // [4]
  f:       f32,  // [5]
  H:       f32,  // [6]  upper layer mean depth H1
  nu4:     f32,  // [7]
  A_eq:    f32,  // [8]
  tau:     f32,  // [9]
  beta:    f32,  // [10]
  gain:    f32,  // [11]
  mode:    u32,  // [12]
  H2:      f32,  // [13] lower layer mean depth
  g_prime: f32,  // [14]
  layerMode: u32,  // [15]
  q_sat:     f32,  // [16] saturation threshold — used to normalize tracer display
}

@group(0) @binding(0) var<uniform>        params: Params;
@group(0) @binding(1) var<storage, read>  field:  array<f32>;
@group(0) @binding(2) var<storage, read>  tracer: array<f32>;

const PI: f32 = 3.14159265358979;

struct VertOut {
  @builtin(position) pos: vec4f,
  @location(0)       uv:  vec2f,
}

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VertOut {
  var corners = array<vec2f, 6>(
    vec2f(-1.0, -1.0), vec2f( 1.0, -1.0), vec2f(-1.0,  1.0),
    vec2f(-1.0,  1.0), vec2f( 1.0, -1.0), vec2f( 1.0,  1.0),
  );
  let p = corners[vi];
  var out: VertOut;
  out.pos = vec4f(p, 0.0, 1.0);
  out.uv  = vec2f((p.x + 1.0) * 0.5, (1.0 - p.y) * 0.5);
  return out;
}

fn colormap(t: f32) -> vec3f {
  let trough = vec3f(0.08, 0.02, 0.35);
  let base   = vec3f(0.03, 0.18, 0.42);
  let crest  = vec3f(0.72, 0.96, 0.98);
  if (t < 0.0) { return mix(base, trough, clamp(-t, 0.0, 1.0)); }
  return mix(base, crest, clamp(t, 0.0, 1.0));
}

@fragment
fn fs(in: VertOut) -> @location(0) vec4f {
  let nx = params.width;
  let ny = params.height;
  let i  = u32(clamp(in.uv.x, 0.0, 0.9999) * f32(nx));
  let j  = u32(clamp(in.uv.y, 0.0, 0.9999) * f32(ny));

  let il = (i + nx - 1u) % nx;  let ir = (i + 1u)      % nx;
  let jd = (j + ny - 1u) % ny;  let ju = (j + 1u)      % ny;

  let h1o = 0u;
  let u1o = nx * ny;
  let v1o = 2u * nx * ny;
  let h2o = 3u * nx * ny;
  let u2o = 4u * nx * ny;
  let v2o = 5u * nx * ny;

  var val: f32;

  if params.mode == 1u {
    // Layer 1 vorticity
    let dv_dx = field[v1o + j*nx + ir] - field[v1o + j*nx + il];
    let du_dy = field[u1o + ju*nx + i] - field[u1o + jd*nx + i];
    val = (dv_dx - du_dy) * 0.5 * params.gain;
  } else if params.mode == 2u {
    val = field[h1o + j*nx + i] * params.gain;
  } else if params.mode == 3u {
    val = field[h2o + j*nx + i] * params.gain;
  } else if params.mode == 4u {
    // Barotropic vorticity: depth-weighted mean of ζ1 and ζ2
    let z1_dv = field[v1o + j*nx + ir] - field[v1o + j*nx + il];
    let z1_du = field[u1o + ju*nx + i] - field[u1o + jd*nx + i];
    let z1    = (z1_dv - z1_du) * 0.5;
    let z2_dv = field[v2o + j*nx + ir] - field[v2o + j*nx + il];
    let z2_du = field[u2o + ju*nx + i] - field[u2o + jd*nx + i];
    let z2    = (z2_dv - z2_du) * 0.5;
    val = (params.H * z1 + params.H2 * z2) / (params.H + params.H2) * params.gain;
  } else if params.mode == 5u {
    // Baroclinic vorticity: ζ1 − ζ2
    let z1_dv = field[v1o + j*nx + ir] - field[v1o + j*nx + il];
    let z1_du = field[u1o + ju*nx + i] - field[u1o + jd*nx + i];
    let z1    = (z1_dv - z1_du) * 0.5;
    let z2_dv = field[v2o + j*nx + ir] - field[v2o + j*nx + il];
    let z2_du = field[u2o + ju*nx + i] - field[u2o + jd*nx + i];
    let z2    = (z2_dv - z2_du) * 0.5;
    val = (z1 - z2) * params.gain;
  } else if params.mode == 6u {
    // Tracer / moisture: normalize by q_sat so the full colormap is used.
    // q=0 (dry) → −1 → deep indigo
    // q=q_sat/2 → 0 → ocean blue
    // q=q_sat (saturated) → +1 → seafoam
    let q_norm = tracer[j*nx + i] / max(params.q_sat, 0.001);
    val = (q_norm - 0.5) * 2.0 * params.gain;
  } else {
    // Mode 0: h1 anomaly relative to thermal equilibrium
    let h    = field[h1o + j*nx + i];
    let h_eq = -params.A_eq * cos(2.0 * PI * f32(j) / f32(ny));
    val = (h - h_eq) * params.gain;
  }

  return vec4f(colormap(clamp(val, -1.0, 1.0)), 1.0);
}
`,Xe=`// Passive moisture tracer advected by layer-1 velocity using first-order upwind.
//
// State layout of SWE buffer: [h1 | u1 | v1 | h2 | u2 | v2], each block nx·ny floats.
//
// Source: evaporation at equator (j = ny/2) following E·(1−cos(2πj/ny))/2.
// Sink  : Newtonian precipitation when q > q_sat, rate k_prec·(q − q_sat).
//
// Upwind (donor-cell) rather than LF: monotone, so the checkerboard mode that
// plagues 2D LF advection cannot grow. CFL bound: (|u|+|v|)·dt/dx ≤ 1.

struct Params {
  width:     u32,
  height:    u32,
  dx:        f32,
  dt:        f32,
  step:      u32,
  f:         f32,
  H:         f32,
  nu4:       f32,
  A_eq:      f32,
  tau:       f32,
  beta:      f32,
  _pad0:     u32,
  _pad1:     u32,
  H2:        f32,
  g_prime:   f32,
  layerMode: u32,
}

struct TracerParams {
  e_evap: f32,  // evaporation rate at equator
  k_prec: f32,  // precipitation rate coefficient
  q_sat:  f32,  // saturation specific humidity
  _pad:   f32,
}

@group(0) @binding(0) var<uniform>             params:  Params;
@group(0) @binding(1) var<storage, read>       swe:     array<f32>;
@group(0) @binding(2) var<storage, read>       q_src:   array<f32>;
@group(0) @binding(3) var<storage, read_write> q_dst:   array<f32>;
@group(0) @binding(4) var<uniform>             tparams: TracerParams;

const PI: f32 = 3.14159265358979;

@compute @workgroup_size(8, 8)
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
  let i  = gid.x;
  let j  = gid.y;
  let nx = params.width;
  let ny = params.height;
  if (i >= nx || j >= ny) { return; }

  let il = (i + nx - 1u) % nx;  let ir = (i + 1u) % nx;
  let jd = (j + ny - 1u) % ny;  let ju = (j + 1u) % ny;

  let u1o = nx * ny;
  let v1o = 2u * nx * ny;

  let q_c = q_src[j  * nx + i ];
  let q_r = q_src[j  * nx + ir];
  let q_l = q_src[j  * nx + il];
  let q_u = q_src[ju * nx + i ];
  let q_d = q_src[jd * nx + i ];

  let u_c = swe[u1o + j * nx + i];
  let v_c = swe[v1o + j * nx + i];

  // CFL safety: scale velocity if (|u|+|v|)·dt/dx > 0.9
  let dt_dx = params.dt / params.dx;
  let cfl   = (abs(u_c) + abs(v_c)) * dt_dx;
  let safe  = select(1.0, 0.9 / cfl, cfl > 0.9);
  let u_s   = u_c * safe;
  let v_s   = v_c * safe;

  // First-order upwind advection.
  // LF (centered + 4-neighbor average) develops a checkerboard instability in 2D
  // because even- and odd-parity grid points decouple. Upwind is monotone — it
  // cannot create new extrema — so the checkerboard mode cannot grow.
  // Tradeoff: more numerical diffusion than LF, but the field stays smooth.
  //
  // Direction convention: if u > 0 flow moves right, so information comes from
  // the left cell; if u < 0 information comes from the right cell.
  let dqx = select(q_c - q_l, q_r - q_c, u_s < 0.0);
  let dqy = select(q_c - q_d, q_u - q_c, v_s < 0.0);

  // Evaporation: confined to the tropical belt (j near ny/2).
  // max(0, -cos(2πj/ny)) is positive only in the inner half of the domain,
  // zero at j=0, ny/4, 3ny/4, ny (poles and mid-latitudes).
  // The extratropics are a dry sink — moisture only arrives there via jet advection,
  // which is what creates filaments and storm-track signatures.
  let evap = tparams.e_evap * max(0.0, -cos(2.0 * PI * f32(j) / f32(ny)));

  // Precipitation: Newtonian relaxation toward q_sat from above
  let prec = tparams.k_prec * max(0.0, q_c - tparams.q_sat);

  let tendency = -(u_s * dqx + v_s * dqy) / params.dx + evap - prec;
  q_dst[j * nx + i] = max(0.0, q_c + params.dt * tendency);
}
`,Ze=`// Two-layer Lax-Friedrichs SWE with Coriolis, beta-plane, and Newtonian relaxation.
//
// State layout: [h1 | u1 | v1 | h2 | u2 | v2] — each block is nx·ny floats.
//
// LF spatial averaging replaces center values with 4-neighbor averages before
// subtracting flux divergence. Numerical diffusion dx²/(4dt)·∇² is baked in;
// explicit ν₄ is NOT applied (would push |G|>1 at Nyquist → blowup).
//
// Stability limit: c_bt·dt·(1/dx+1/dy) ≤ 1  where  c_bt = √(H+H2).

struct Params {
  width:   u32,  // [0]
  height:  u32,  // [1]
  dx:      f32,  // [2]
  dt:      f32,  // [3]
  step:    u32,  // [4]
  f:       f32,  // [5]  base Coriolis
  H:       f32,  // [6]  upper layer mean depth H1
  nu4:     f32,  // [7]  (unused in LF)
  A_eq:    f32,  // [8]  thermal forcing amplitude
  tau:     f32,  // [9]  Newtonian relaxation timescale
  beta:    f32,  // [10] beta-plane parameter
  _pad0:   u32,  // [11] (gain in render shader)
  _pad1:   u32,  // [12] (mode in render shader)
  H2:      f32,  // [13] lower layer mean depth
  g_prime: f32,  // [14] reduced gravity
  layerMode: u32,  // [15] 0=single-layer, 1=two-layer
}

@group(0) @binding(0) var<uniform>             params: Params;
@group(0) @binding(1) var<storage, read>       src:    array<f32>;
@group(0) @binding(2) var<storage, read_write> dst:    array<f32>;

const g:  f32 = 1.0;
const PI: f32 = 3.14159265358979;

@compute @workgroup_size(8, 8)
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
  let i  = gid.x;
  let j  = gid.y;
  let nx = params.width;
  let ny = params.height;
  if (i >= nx || j >= ny) { return; }

  let il = (i + nx - 1u) % nx;  let ir = (i + 1u)      % nx;
  let jd = (j + ny - 1u) % ny;  let ju = (j + 1u)      % ny;

  let h1o = 0u;
  let u1o = nx * ny;
  let v1o = 2u * nx * ny;
  let h2o = 3u * nx * ny;
  let u2o = 4u * nx * ny;
  let v2o = 5u * nx * ny;

  // Center values (for Coriolis and relaxation)
  let h1_c = src[h1o + j*nx + i];
  let u1_c = src[u1o + j*nx + i];
  let v1_c = src[v1o + j*nx + i];
  let u2_c = src[u2o + j*nx + i];
  let v2_c = src[v2o + j*nx + i];

  // All neighbors
  let h1_r = src[h1o + j*nx + ir];  let h1_l = src[h1o + j*nx + il];
  let h1_u = src[h1o + ju*nx + i];  let h1_d = src[h1o + jd*nx + i];

  let u1_r = src[u1o + j*nx + ir];  let u1_l = src[u1o + j*nx + il];
  let u1_u = src[u1o + ju*nx + i];  let u1_d = src[u1o + jd*nx + i];

  let v1_r = src[v1o + j*nx + ir];  let v1_l = src[v1o + j*nx + il];
  let v1_u = src[v1o + ju*nx + i];  let v1_d = src[v1o + jd*nx + i];

  let h2_r = src[h2o + j*nx + ir];  let h2_l = src[h2o + j*nx + il];
  let h2_u = src[h2o + ju*nx + i];  let h2_d = src[h2o + jd*nx + i];

  let u2_r = src[u2o + j*nx + ir];  let u2_l = src[u2o + j*nx + il];
  let u2_u = src[u2o + ju*nx + i];  let u2_d = src[u2o + jd*nx + i];

  let v2_r = src[v2o + j*nx + ir];  let v2_l = src[v2o + j*nx + il];
  let v2_u = src[v2o + ju*nx + i];  let v2_d = src[v2o + jd*nx + i];

  // LF spatial averages
  let h1_avg = 0.25 * (h1_r + h1_l + h1_u + h1_d);
  let u1_avg = 0.25 * (u1_r + u1_l + u1_u + u1_d);
  let v1_avg = 0.25 * (v1_r + v1_l + v1_u + v1_d);
  let h2_avg = 0.25 * (h2_r + h2_l + h2_u + h2_d);
  let u2_avg = 0.25 * (u2_r + u2_l + u2_u + u2_d);
  let v2_avg = 0.25 * (v2_r + v2_l + v2_u + v2_d);

  let c = params.dt / (2.0 * params.dx);

  let f_eff     = params.f + params.beta * (f32(j) - f32(ny) * 0.5);
  let h1_eq     = -params.A_eq * cos(2.0 * PI * f32(j) / f32(ny));
  let dh1_relax = select(0.0, -(h1_c - h1_eq) / params.tau, params.tau > 0.0);

  // ── Layer 1 ────────────────────────────────────────────────────────────
  dst[h1o + j*nx + i] = h1_avg
                       - params.H * c * ((u1_r - u1_l) + (v1_u - v1_d))
                       + params.dt * dh1_relax;
  dst[u1o + j*nx + i] = u1_avg
                       - g * c * ((h1_r + h2_r) - (h1_l + h2_l))
                       + params.dt * f_eff * v1_c;
  dst[v1o + j*nx + i] = v1_avg
                       - g * c * ((h1_u + h2_u) - (h1_d + h2_d))
                       - params.dt * f_eff * u1_c;

  // ── Layer 2 ────────────────────────────────────────────────────────────
  if params.layerMode == 0u {
    dst[h2o + j*nx + i] = 0.0;
    dst[u2o + j*nx + i] = 0.0;
    dst[v2o + j*nx + i] = 0.0;
  } else {
    dst[h2o + j*nx + i] = h2_avg
                         - params.H2 * c * ((u2_r - u2_l) + (v2_u - v2_d));
    dst[u2o + j*nx + i] = u2_avg
                         - g * c * ((h1_r + h2_r) - (h1_l + h2_l))
                         - params.g_prime * c * (h2_r - h2_l)
                         + params.dt * f_eff * v2_c;
    dst[v2o + j*nx + i] = v2_avg
                         - g * c * ((h1_u + h2_u) - (h1_d + h2_d))
                         - params.g_prime * c * (h2_u - h2_d)
                         - params.dt * f_eff * u2_c;
  }
}
`,ge=8;function Je(e,a,s,u,o){const n=e.createComputePipeline({label:"lf",layout:"auto",compute:{module:e.createShaderModule({label:"lf-compute",code:Ze}),entryPoint:"cs_main"}}),d=n.getBindGroupLayout(0),g=[e.createBindGroup({label:"lf A→B",layout:d,entries:[{binding:0,resource:{buffer:s}},{binding:1,resource:{buffer:a[0]}},{binding:2,resource:{buffer:a[1]}}]}),e.createBindGroup({label:"lf B→A",layout:d,entries:[{binding:0,resource:{buffer:s}},{binding:1,resource:{buffer:a[1]}},{binding:2,resource:{buffer:a[0]}}]})];let c=0;return{tick(f,t){const l=f.beginComputePass({label:"lf"});l.setPipeline(n);for(let x=0;x<t;x++)l.setBindGroup(0,g[c]),l.dispatchWorkgroups(u/ge,o/ge),c=1-c;l.end()},get readBuf(){return c},setCur(f){c=f},destroy(){}}}const en=`// Two-layer SWE right-hand side: centered differences, no spatial averaging.
// Used by RK4. Lax-Friedrichs uses compute.wgsl.
//
// State layout: [h1 | u1 | v1 | h2 | u2 | v2] — each block is nx·ny floats.
//   Layer 1 = upper (free surface),  Layer 2 = lower (abyssal).
//
// Linearized two-layer SWE (Boussinesq, g=1):
//
//   dh1/dt = -H·(∂u1/∂x + ∂v1/∂y)  − (h1−h_eq)/τ  − ν₄∇⁴h1
//   du1/dt = -g·∂(h1+h2)/∂x + f_eff·v1             − ν₄∇⁴u1
//   dv1/dt = -g·∂(h1+h2)/∂y − f_eff·u1             − ν₄∇⁴v1
//
//   dh2/dt = -H2·(∂u2/∂x + ∂v2/∂y)                − ν₄∇⁴h2
//   du2/dt = -g·∂(h1+h2)/∂x − g'·∂h2/∂x + f_eff·v2 − ν₄∇⁴u2
//   dv2/dt = -g·∂(h1+h2)/∂y − g'·∂h2/∂y − f_eff·u2 − ν₄∇⁴v2
//
// where g' = g·(ρ₂−ρ₁)/ρ₂ is the reduced gravity (internal wave speed
// c₂ = √(g'·H·H2/(H+H2))).  Thermal relaxation applied to upper layer only.

struct Params {
  width:   u32,  // [0]
  height:  u32,  // [1]
  dx:      f32,  // [2]
  dt:      f32,  // [3]
  step:    u32,  // [4]
  f:       f32,  // [5]  base Coriolis
  H:       f32,  // [6]  upper layer mean depth H1
  nu4:     f32,  // [7]  hyperdiffusion coefficient
  A_eq:    f32,  // [8]  thermal forcing amplitude
  tau:     f32,  // [9]  Newtonian relaxation timescale
  beta:    f32,  // [10] beta-plane parameter
  _pad0:   u32,  // [11] (gain in render shader)
  _pad1:   u32,  // [12] (mode in render shader)
  H2:      f32,  // [13] lower layer mean depth
  g_prime: f32,  // [14] reduced gravity
  layerMode: u32,  // [15] 0=single-layer, 1=two-layer
}

@group(0) @binding(0) var<uniform>             params: Params;
@group(0) @binding(1) var<storage, read>       src:    array<f32>;
@group(0) @binding(2) var<storage, read_write> dst:    array<f32>;

const g:  f32 = 1.0;
const PI: f32 = 3.14159265358979;

// 13-point biharmonic stencil ∇⁴φ with periodic wrap, dx=1.
fn biharm(off: u32, ci: u32, cj: u32, nx: u32, ny: u32) -> f32 {
  let il  = (ci + nx - 1u) % nx;  let ir  = (ci + 1u)      % nx;
  let il2 = (ci + nx - 2u) % nx;  let ir2 = (ci + 2u)      % nx;
  let jd  = (cj + ny - 1u) % ny;  let ju  = (cj + 1u)      % ny;
  let jd2 = (cj + ny - 2u) % ny;  let ju2 = (cj + 2u)      % ny;

  let c00  = src[off + cj  * nx + ci ];
  let p10  = src[off + cj  * nx + ir ];  let m10  = src[off + cj  * nx + il ];
  let p01  = src[off + ju  * nx + ci ];  let m01  = src[off + jd  * nx + ci ];
  let p20  = src[off + cj  * nx + ir2];  let m20  = src[off + cj  * nx + il2];
  let p02  = src[off + ju2 * nx + ci ];  let m02  = src[off + jd2 * nx + ci ];
  let p11  = src[off + ju  * nx + ir ];  let p1m1 = src[off + jd  * nx + ir ];
  let m11  = src[off + ju  * nx + il ];  let m1m1 = src[off + jd  * nx + il ];

  return 20.0*c00
       - 8.0*(p10 + m10 + p01 + m01)
       + 2.0*(p11 + p1m1 + m11 + m1m1)
       +     (p20 + m20 + p02 + m02);
}

@compute @workgroup_size(8, 8)
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
  let i  = gid.x;
  let j  = gid.y;
  let nx = params.width;
  let ny = params.height;
  if (i >= nx || j >= ny) { return; }

  let il = (i + nx - 1u) % nx;  let ir = (i + 1u)      % nx;
  let jd = (j + ny - 1u) % ny;  let ju = (j + 1u)      % ny;

  let h1o = 0u;
  let u1o = nx * ny;
  let v1o = 2u * nx * ny;
  let h2o = 3u * nx * ny;
  let u2o = 4u * nx * ny;
  let v2o = 5u * nx * ny;

  let inv2dx = 1.0 / (2.0 * params.dx);

  let f_eff = params.f + params.beta * (f32(j) - f32(ny) * 0.5);
  let h1_eq = -params.A_eq * cos(2.0 * PI * f32(j) / f32(ny));

  // Center values
  let h1_c = src[h1o + j*nx + i];
  let u1_c = src[u1o + j*nx + i];
  let v1_c = src[v1o + j*nx + i];
  let u2_c = src[u2o + j*nx + i];
  let v2_c = src[v2o + j*nx + i];

  // Layer 1 divergence
  let u1_r = src[u1o + j*nx + ir];  let u1_l = src[u1o + j*nx + il];
  let v1_u = src[v1o + ju*nx + i];  let v1_d = src[v1o + jd*nx + i];

  // Layer 2 divergence
  let u2_r = src[u2o + j*nx + ir];  let u2_l = src[u2o + j*nx + il];
  let v2_u = src[v2o + ju*nx + i];  let v2_d = src[v2o + jd*nx + i];

  // Height gradients
  let h1_r = src[h1o + j*nx + ir];  let h1_l = src[h1o + j*nx + il];
  let h1_u = src[h1o + ju*nx + i];  let h1_d = src[h1o + jd*nx + i];
  let h2_r = src[h2o + j*nx + ir];  let h2_l = src[h2o + j*nx + il];
  let h2_u = src[h2o + ju*nx + i];  let h2_d = src[h2o + jd*nx + i];

  // Free-surface gradient ∂(h1+h2)/∂x,y  and  interface gradient ∂h2/∂x,y
  let dsurf_dx = (h1_r + h2_r - h1_l - h2_l) * inv2dx;
  let dsurf_dy = (h1_u + h2_u - h1_d - h2_d) * inv2dx;
  let dh2_dx   = (h2_r - h2_l) * inv2dx;
  let dh2_dy   = (h2_u - h2_d) * inv2dx;

  let dh1_relax = select(0.0, -(h1_c - h1_eq) / params.tau, params.tau > 0.0);

  // ── Layer 1 ────────────────────────────────────────────────────────────
  dst[h1o + j*nx + i] = -params.H * ((u1_r - u1_l) + (v1_u - v1_d)) * inv2dx
                        + dh1_relax
                        - params.nu4 * biharm(h1o, i, j, nx, ny);
  dst[u1o + j*nx + i] = -g * dsurf_dx + f_eff * v1_c
                        - params.nu4 * biharm(u1o, i, j, nx, ny);
  dst[v1o + j*nx + i] = -g * dsurf_dy - f_eff * u1_c
                        - params.nu4 * biharm(v1o, i, j, nx, ny);

  // ── Layer 2 ────────────────────────────────────────────────────────────
  if params.layerMode == 0u {
    dst[h2o + j*nx + i] = 0.0;
    dst[u2o + j*nx + i] = 0.0;
    dst[v2o + j*nx + i] = 0.0;
  } else {
    dst[h2o + j*nx + i] = -params.H2 * ((u2_r - u2_l) + (v2_u - v2_d)) * inv2dx
                          - params.nu4 * biharm(h2o, i, j, nx, ny);
    dst[u2o + j*nx + i] = -g * dsurf_dx - params.g_prime * dh2_dx + f_eff * v2_c
                          - params.nu4 * biharm(u2o, i, j, nx, ny);
    dst[v2o + j*nx + i] = -g * dsurf_dy - params.g_prime * dh2_dy - f_eff * u2_c
                          - params.nu4 * biharm(v2o, i, j, nx, ny);
  }
}
`,nn=`// RK4 stage blend and accumulate.
//
// Classical RK4:
//   k1 = F(uⁿ)
//   k2 = F(uⁿ + dt/2·k1)
//   k3 = F(uⁿ + dt/2·k2)
//   k4 = F(uⁿ + dt·k3)
//   u^{n+1} = uⁿ + dt/6·(k1 + 2k2 + 2k3 + k4)
//
// Four entry points — one called after each F evaluation:
//
//   cs_stage1:  buf_out = uⁿ + dt/2·k    buf_acc = k          (init)
//   cs_stage2:  buf_out = uⁿ + dt/2·k    buf_acc += 2·k
//   cs_stage3:  buf_out = uⁿ + dt·k      buf_acc += 2·k
//   cs_finalize: buf_out = uⁿ + dt/6·(buf_acc + k)
//
// Operates element-wise over the flat [h1|u1|v1|h2|u2|v2] array — no
// field-layout knowledge needed. Dispatched 1D over all 6·NX·NY elements.

struct Params {
  width:  u32,
  height: u32,
  dx:     f32,
  dt:     f32,
  step:   u32,
  f:      f32,
  H:      f32,
}

@group(0) @binding(0) var<uniform>             params:  Params;
@group(0) @binding(1) var<storage, read>       buf_n:   array<f32>;
@group(0) @binding(2) var<storage, read>       buf_k:   array<f32>;
@group(0) @binding(3) var<storage, read_write> buf_out: array<f32>;
@group(0) @binding(4) var<storage, read_write> buf_acc: array<f32>;

@compute @workgroup_size(256)
fn cs_stage1(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  if (idx >= params.width * params.height * 6u) { return; }
  buf_out[idx] = buf_n[idx] + (params.dt * 0.5) * buf_k[idx];
  buf_acc[idx] = buf_k[idx];
}

@compute @workgroup_size(256)
fn cs_stage2(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  if (idx >= params.width * params.height * 6u) { return; }
  buf_out[idx] = buf_n[idx] + (params.dt * 0.5) * buf_k[idx];
  buf_acc[idx] += 2.0 * buf_k[idx];
}

@compute @workgroup_size(256)
fn cs_stage3(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  if (idx >= params.width * params.height * 6u) { return; }
  buf_out[idx] = buf_n[idx] + params.dt * buf_k[idx];
  buf_acc[idx] += 2.0 * buf_k[idx];
}

@compute @workgroup_size(256)
fn cs_finalize(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  if (idx >= params.width * params.height * 6u) { return; }
  buf_out[idx] = buf_n[idx] + (params.dt / 6.0) * (buf_acc[idx] + buf_k[idx]);
}
`,me=8,tn=256;function rn(e,a,s,u,o){const n=u*o*6*4,d=u*o*6,g=Math.ceil(d/tn),c=e.createBuffer({label:"rk4-k",size:n,usage:GPUBufferUsage.STORAGE}),f=e.createBuffer({label:"rk4-stage",size:n,usage:GPUBufferUsage.STORAGE}),t=e.createBuffer({label:"rk4-acc",size:n,usage:GPUBufferUsage.STORAGE}),l=GPUShaderStage.COMPUTE,x=e.createBindGroupLayout({entries:[{binding:0,visibility:l,buffer:{type:"uniform"}},{binding:1,visibility:l,buffer:{type:"read-only-storage"}},{binding:2,visibility:l,buffer:{type:"storage"}}]}),j=e.createBindGroupLayout({entries:[{binding:0,visibility:l,buffer:{type:"uniform"}},{binding:1,visibility:l,buffer:{type:"read-only-storage"}},{binding:2,visibility:l,buffer:{type:"read-only-storage"}},{binding:3,visibility:l,buffer:{type:"storage"}},{binding:4,visibility:l,buffer:{type:"storage"}}]}),A=e.createPipelineLayout({bindGroupLayouts:[x]}),q=e.createPipelineLayout({bindGroupLayouts:[j]}),M=e.createShaderModule({label:"rhs",code:en}),P=e.createShaderModule({label:"blend",code:nn}),E=e.createComputePipeline({label:"rhs",layout:A,compute:{module:M,entryPoint:"cs_main"}}),U=e.createComputePipeline({label:"rk4-s1",layout:q,compute:{module:P,entryPoint:"cs_stage1"}}),T=e.createComputePipeline({label:"rk4-s2",layout:q,compute:{module:P,entryPoint:"cs_stage2"}}),O=e.createComputePipeline({label:"rk4-s3",layout:q,compute:{module:P,entryPoint:"cs_stage3"}}),V=e.createComputePipeline({label:"rk4-fin",layout:q,compute:{module:P,entryPoint:"cs_finalize"}}),W=[e.createBindGroup({label:"rhs-0",layout:x,entries:[{binding:0,resource:{buffer:s}},{binding:1,resource:{buffer:a[0]}},{binding:2,resource:{buffer:c}}]}),e.createBindGroup({label:"rhs-1",layout:x,entries:[{binding:0,resource:{buffer:s}},{binding:1,resource:{buffer:a[1]}},{binding:2,resource:{buffer:c}}]})],F=e.createBindGroup({label:"rhs-stage",layout:x,entries:[{binding:0,resource:{buffer:s}},{binding:1,resource:{buffer:f}},{binding:2,resource:{buffer:c}}]}),C=[e.createBindGroup({label:"blend-0",layout:j,entries:[{binding:0,resource:{buffer:s}},{binding:1,resource:{buffer:a[0]}},{binding:2,resource:{buffer:c}},{binding:3,resource:{buffer:f}},{binding:4,resource:{buffer:t}}]}),e.createBindGroup({label:"blend-1",layout:j,entries:[{binding:0,resource:{buffer:s}},{binding:1,resource:{buffer:a[1]}},{binding:2,resource:{buffer:c}},{binding:3,resource:{buffer:f}},{binding:4,resource:{buffer:t}}]})],G=[e.createBindGroup({label:"fin-0",layout:j,entries:[{binding:0,resource:{buffer:s}},{binding:1,resource:{buffer:a[0]}},{binding:2,resource:{buffer:c}},{binding:3,resource:{buffer:a[1]}},{binding:4,resource:{buffer:t}}]}),e.createBindGroup({label:"fin-1",layout:j,entries:[{binding:0,resource:{buffer:s}},{binding:1,resource:{buffer:a[1]}},{binding:2,resource:{buffer:c}},{binding:3,resource:{buffer:a[0]}},{binding:4,resource:{buffer:t}}]})];let B=0;const m=[u/me,o/me];function _(p){const y=B;let i=p.beginComputePass({label:"k1"});i.setPipeline(E),i.setBindGroup(0,W[y]),i.dispatchWorkgroups(m[0],m[1]),i.end(),i=p.beginComputePass({label:"s1"}),i.setPipeline(U),i.setBindGroup(0,C[y]),i.dispatchWorkgroups(g),i.end(),i=p.beginComputePass({label:"k2"}),i.setPipeline(E),i.setBindGroup(0,F),i.dispatchWorkgroups(m[0],m[1]),i.end(),i=p.beginComputePass({label:"s2"}),i.setPipeline(T),i.setBindGroup(0,C[y]),i.dispatchWorkgroups(g),i.end(),i=p.beginComputePass({label:"k3"}),i.setPipeline(E),i.setBindGroup(0,F),i.dispatchWorkgroups(m[0],m[1]),i.end(),i=p.beginComputePass({label:"s3"}),i.setPipeline(O),i.setBindGroup(0,C[y]),i.dispatchWorkgroups(g),i.end(),i=p.beginComputePass({label:"k4"}),i.setPipeline(E),i.setBindGroup(0,F),i.dispatchWorkgroups(m[0],m[1]),i.end(),i=p.beginComputePass({label:"rk4-fin"}),i.setPipeline(V),i.setBindGroup(0,G[y]),i.dispatchWorkgroups(g),i.end(),B=1-B}return{tick(p,y){for(let i=0;i<y;i++)_(p)},get readBuf(){return B},setCur(p){B=p},destroy(){c.destroy(),f.destroy(),t.destroy()}}}const S=256,I=256,_e=8,ve=.4,ye=80,an=16,he={"lax-friedrichs":Je,rk4:rn};function on(e,a,s="lax-friedrichs"){const u=GPUShaderStage.COMPUTE,o=S*I,n=o*6*4,d=[0,1].map(r=>e.createBuffer({label:`state-${r}`,size:n,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST})),g=e.createBuffer({label:"params",size:ye,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),c=o*4,f=[0,1].map(r=>e.createBuffer({label:`tracer-${r}`,size:c,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST})),t=e.createBuffer({label:"tracer-params",size:an,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});let l=0,x=0,j=.25,A=.25,q=.02,M=.1,P=.01,E=.3,U=300,T=0,O=5,V=0,W=1,F=.005,C=2,G=1;const B=new ArrayBuffer(ye),m=new Uint32Array(B),_=new Float32Array(B);m[0]=S,m[1]=I,_[2]=1,_[3]=ve;const p=new Float32Array(4);function y(){p[0]=F,p[1]=C,p[2]=G,p[3]=0,e.queue.writeBuffer(t,0,p)}y();const i=e.createShaderModule({label:"render",code:Qe}),de=e.createRenderPipeline({label:"render",layout:"auto",vertex:{module:i,entryPoint:"vs"},fragment:{module:i,entryPoint:"fs",targets:[{format:a}]},primitive:{topology:"triangle-list"}}),Oe=de.getBindGroupLayout(0),Ve=[0,1].map(r=>[0,1].map(v=>e.createBindGroup({label:`render-s${r}-t${v}`,layout:Oe,entries:[{binding:0,resource:{buffer:g}},{binding:1,resource:{buffer:d[r]}},{binding:2,resource:{buffer:f[v]}}]}))),ce=e.createBindGroupLayout({entries:[{binding:0,visibility:u,buffer:{type:"uniform"}},{binding:1,visibility:u,buffer:{type:"read-only-storage"}},{binding:2,visibility:u,buffer:{type:"read-only-storage"}},{binding:3,visibility:u,buffer:{type:"storage"}},{binding:4,visibility:u,buffer:{type:"uniform"}}]}),We=e.createComputePipeline({label:"tracer",layout:e.createPipelineLayout({bindGroupLayouts:[ce]}),compute:{module:e.createShaderModule({label:"tracer",code:Xe}),entryPoint:"cs_main"}}),Ne=[0,1].map(r=>[0,1].map(v=>e.createBindGroup({label:`tracer-tc${r}-sc${v}`,layout:ce,entries:[{binding:0,resource:{buffer:g}},{binding:1,resource:{buffer:d[v]}},{binding:2,resource:{buffer:f[r]}},{binding:3,resource:{buffer:f[1-r]}},{binding:4,resource:{buffer:t}}]})));let N=0;function Re(r,v){const h=r.beginComputePass({label:"tracer"});h.setPipeline(We),h.setBindGroup(0,Ne[N][v]),h.dispatchWorkgroups(S/_e,I/_e),h.end(),N=1-N}function fe(){e.queue.writeBuffer(d[0],0,sn(S,I,M)),e.queue.writeBuffer(d[1],0,new Float32Array(o*6))}function pe(){e.queue.writeBuffer(f[0],0,new Float32Array(o)),e.queue.writeBuffer(f[1],0,new Float32Array(o)),N=0}fe(),pe();let w=he[s](e,d,g,S,I);function De(r,v=1){m[4]=l,_[5]=x,_[6]=j,_[7]=P,_[8]=E,_[9]=U,_[10]=T,_[11]=O,m[12]=V,_[13]=A,_[14]=q,m[15]=W,_[16]=G,e.queue.writeBuffer(g,0,B);for(let h=0;h<v;h++)w.tick(r,1),Re(r,w.readBuf);l+=v}function $e(r,v){const h=r.beginRenderPass({colorAttachments:[{view:v,clearValue:{r:.03,g:.18,b:.42,a:1},loadOp:"clear",storeOp:"store"}]});h.setPipeline(de),h.setBindGroup(0,Ve[w.readBuf][N]),h.draw(6),h.end()}function Ye(){l=0,fe(),w.setCur(0),pe()}function Ke(r){const v=w.readBuf;w.destroy(),w=he[r](e,d,g,S,I),w.setCur(v)}return{tick:De,render:$e,reset:Ye,setScheme:Ke,setF(r){x=r},setH(r){j=r},setH2(r){A=r},setGPrime(r){q=r},setSigma(r){M=r},setNu4(r){P=r},setAEq(r){E=r},setTau(r){U=r},setBeta(r){T=r},setGain(r){O=r},setMode(r){V=r},setLayerMode(r){W=r},setEEvap(r){F=r,y()},setKPrec(r){C=r,y()},setQSat(r){G=r,y()},get step(){return l},get simTime(){return l*ve},get f(){return x},get H(){return j},get H2(){return A},get g_prime(){return q},get sigma(){return M},get nu4(){return P},get A_eq(){return E},get tau(){return U},get beta(){return T},get gain(){return O},get mode(){return V},get layerMode(){return W},get e_evap(){return F},get k_prec(){return C},get q_sat(){return G}}}function sn(e,a,s){const u=new Float32Array(e*a*6),o=e/2,n=a/2,d=s*e,g=s*a;for(let c=0;c<a;c++)for(let f=0;f<e;f++){const t=(f-o)/d,l=(c-n)/g;u[c*e+f]=Math.exp(-(t*t+l*l))}return u}const K=document.getElementById("status"),xe=document.getElementById("pause-btn"),un=document.getElementById("reset-btn"),be=document.getElementById("scheme-select"),se=document.getElementById("speed-slider"),ln=document.getElementById("speed-val"),X=document.getElementById("sigma-slider"),je=document.getElementById("sigma-val"),k=document.getElementById("H-slider"),dn=document.getElementById("H-val"),L=document.getElementById("H2-slider"),cn=document.getElementById("H2-val"),fn=document.getElementById("cbt-val"),pn=document.getElementById("c1-val"),Be=document.getElementById("c1-display"),H=document.getElementById("gp-slider"),gn=document.getElementById("gp-val"),mn=document.getElementById("c2-val"),R=document.getElementById("f-slider"),_n=document.getElementById("f-val"),vn=document.getElementById("lr-val"),Z=document.getElementById("nu4-slider"),qe=document.getElementById("nu4-val"),J=document.getElementById("Aeq-slider"),Pe=document.getElementById("Aeq-val"),ee=document.getElementById("tau-slider"),Ee=document.getElementById("tau-val"),ne=document.getElementById("beta-slider"),we=document.getElementById("beta-val"),b=document.getElementById("display-select"),te=document.getElementById("gain-slider"),Fe=document.getElementById("gain-val"),re=document.getElementById("evap-slider"),Ce=document.getElementById("evap-val"),ae=document.getElementById("qsat-slider"),ke=document.getElementById("qsat-val"),ie=document.getElementById("prec-slider"),Le=document.getElementById("prec-val"),ue=document.getElementById("layer1-btn"),le=document.getElementById("layer2-btn"),Ge=document.getElementById("h2-sep"),Se=document.getElementById("h2-label"),Ie=document.getElementById("gp-sep"),He=document.getElementById("gp-label"),ze=document.getElementById("h1-name"),Ae=document.getElementById("h1-hint"),Me=document.getElementById("h1-label-el");let Y=!1,oe=parseInt(se.value),z=!0;const yn=[{value:"0",text:"h anomaly (h − h_eq)"},{value:"1",text:"vorticity"},{value:"2",text:"raw h"},{value:"6",text:"tracer q (moisture)"}],hn=[{value:"0",text:"h₁ anomaly (h₁ − h_eq)"},{value:"1",text:"vorticity layer 1"},{value:"2",text:"raw h₁"},{value:"3",text:"h₂ interface"},{value:"4",text:"barotropic vorticity"},{value:"5",text:"baroclinic vorticity (ζ₁−ζ₂)"},{value:"6",text:"tracer q (moisture)"}];function Ue(e){const a=b.value;for(;b.options.length;)b.remove(0);for(const{value:s,text:u}of e)b.add(new Option(u,s));e.some(s=>s.value===a)&&(b.value=a)}function Q(e,a){dn.textContent=e.toFixed(2),z?(cn.textContent=a.toFixed(2),fn.textContent=Math.sqrt(e+a).toFixed(2)):pn.textContent=Math.sqrt(e).toFixed(2)}function D(e,a,s){if(!z)return;gn.textContent=e.toFixed(2);const u=a>0&&s>0?Math.sqrt(e*a*s/(a+s)):0;mn.textContent=u.toFixed(2)}function $(e,a,s){const u=Math.sqrt(z?a+s:a);_n.textContent=e.toFixed(3),vn.textContent=e>0?(u/e).toFixed(1):"∞"}function Te(e,a){z=e;const s=d=>{d.style.display="none"},u=d=>{d.style.display=""};if(e)u(Ge),u(Se),u(Ie),u(He),s(Be),ze.textContent="H₁",Ae.textContent="(upper depth)",Me.title="Mean depth of upper layer. Barotropic wave speed c=√(g·(H₁+H₂)).",Ue(hn),le.classList.add("layer-active"),ue.classList.remove("layer-active");else{s(Ge),s(Se),s(Ie),s(He),u(Be),ze.textContent="H",Ae.textContent="(depth)",Me.title="Mean layer depth. Wave speed c=√H.";const d=parseInt(b.value);Ue(yn),d>=3&&(b.value="0",a.setMode(0)),ue.classList.add("layer-active"),le.classList.remove("layer-active")}a.setLayerMode(e?1:0),a.reset();const o=parseFloat(k.value),n=parseFloat(L.value);Q(o,n),$(parseFloat(R.value),o,n),D(parseFloat(H.value),o,n)}async function xn(){var f;if(!navigator.gpu){K.textContent="WebGPU not supported — use Chrome 113+ or Safari 18+";return}const e=await navigator.gpu.requestAdapter();if(!e){K.textContent="No GPU adapter found";return}const a=await e.requestDevice(),u=document.getElementById("canvas").getContext("webgpu"),o=navigator.gpu.getPreferredCanvasFormat();u.configure({device:a,format:o,alphaMode:"opaque"});const n=on(a,o),d=e.info??await((f=e.requestAdapterInfo)==null?void 0:f.call(e))??{},g=d.description||d.vendor||"gpu";je.textContent=n.sigma.toFixed(2),X.value=n.sigma,k.value=n.H,L.value=n.H2,H.value=n.g_prime,Q(n.H,n.H2),D(n.g_prime,n.H,n.H2),$(n.f,n.H,n.H2),Z.value=n.nu4,qe.textContent=n.nu4.toFixed(3),J.value=n.A_eq,Pe.textContent=n.A_eq.toFixed(2),ee.value=n.tau,Ee.textContent=n.tau.toFixed(0),ne.value=n.beta,we.textContent=n.beta.toFixed(3),te.value=n.gain,Fe.textContent=n.gain.toFixed(0),b.value=n.mode.toString(),re.value=n.e_evap,Ce.textContent=n.e_evap.toFixed(3),ae.value=n.q_sat,ke.textContent=n.q_sat.toFixed(1),ie.value=n.k_prec,Le.textContent=n.k_prec.toFixed(1),xe.addEventListener("click",()=>{Y=!Y,xe.textContent=Y?"play":"pause"}),se.addEventListener("input",()=>{oe=parseInt(se.value),ln.textContent=oe}),X.addEventListener("input",()=>{const t=parseFloat(X.value);je.textContent=t.toFixed(2),n.setSigma(t),n.reset()}),k.addEventListener("input",()=>{const t=parseFloat(k.value),l=parseFloat(L.value);n.setH(t),Q(t,l),$(parseFloat(R.value),t,l),D(parseFloat(H.value),t,l)}),L.addEventListener("input",()=>{const t=parseFloat(L.value),l=parseFloat(k.value);n.setH2(t),Q(l,t),$(parseFloat(R.value),l,t),D(parseFloat(H.value),l,t)}),H.addEventListener("input",()=>{const t=parseFloat(H.value);n.setGPrime(t),D(t,parseFloat(k.value),parseFloat(L.value))}),R.addEventListener("input",()=>{const t=parseFloat(R.value);n.setF(t),$(t,parseFloat(k.value),parseFloat(L.value))}),Z.addEventListener("input",()=>{const t=parseFloat(Z.value);n.setNu4(t),qe.textContent=t.toFixed(3)}),J.addEventListener("input",()=>{const t=parseFloat(J.value);n.setAEq(t),Pe.textContent=t.toFixed(2)}),ee.addEventListener("input",()=>{const t=parseFloat(ee.value);n.setTau(t),Ee.textContent=t.toFixed(0)}),ne.addEventListener("input",()=>{const t=parseFloat(ne.value);n.setBeta(t),we.textContent=t.toFixed(3)}),te.addEventListener("input",()=>{const t=parseFloat(te.value);n.setGain(t),Fe.textContent=t.toFixed(0)}),b.addEventListener("change",()=>{n.setMode(parseInt(b.value))}),un.addEventListener("click",()=>n.reset()),be.addEventListener("change",()=>{n.setScheme(be.value)}),re.addEventListener("input",()=>{const t=parseFloat(re.value);n.setEEvap(t),Ce.textContent=t.toFixed(3)}),ae.addEventListener("input",()=>{const t=parseFloat(ae.value);n.setQSat(t),ke.textContent=t.toFixed(1)}),ie.addEventListener("input",()=>{const t=parseFloat(ie.value);n.setKPrec(t),Le.textContent=t.toFixed(1)}),ue.addEventListener("click",()=>{z&&Te(!1,n)}),le.addEventListener("click",()=>{z||Te(!0,n)});function c(){if(!Y){const t=a.createCommandEncoder({label:"frame"});n.tick(t,oe),n.render(t,u.getCurrentTexture().createView()),a.queue.submit([t.finish()])}K.textContent=`${g} | 256×256 | t = ${n.simTime.toFixed(1)}`,requestAnimationFrame(c)}requestAnimationFrame(c)}xn().catch(e=>{console.error(e),K.textContent=`Error: ${e.message}`});
