(function(){const i=document.createElement("link").relList;if(i&&i.supports&&i.supports("modulepreload"))return;for(const u of document.querySelectorAll('link[rel="modulepreload"]'))l(u);new MutationObserver(u=>{for(const n of u)if(n.type==="childList")for(const d of n.addedNodes)d.tagName==="LINK"&&d.rel==="modulepreload"&&l(d)}).observe(document,{childList:!0,subtree:!0});function s(u){const n={};return u.integrity&&(n.integrity=u.integrity),u.referrerPolicy&&(n.referrerPolicy=u.referrerPolicy),u.crossOrigin==="use-credentials"?n.credentials="include":u.crossOrigin==="anonymous"?n.credentials="omit":n.credentials="same-origin",n}function l(u){if(u.ep)return;u.ep=!0;const n=s(u);fetch(u.href,n)}})();const ge=`// Fullscreen quad renderer. Six display modes:
//   0 — h1 anomaly:            (h1 − h_eq)·gain
//   1 — vorticity layer 1:     ζ1 = (∂v1/∂x − ∂u1/∂y)·gain
//   2 — raw h1:                h1·gain
//   3 — h2 interface:          h2·gain
//   4 — barotropic vorticity:  (H·ζ1 + H2·ζ2)/(H+H2)·gain
//   5 — baroclinic vorticity:  (ζ1 − ζ2)·gain

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
  _pad:    u32,  // [15]
}

@group(0) @binding(0) var<uniform>        params: Params;
@group(0) @binding(1) var<storage, read>  field:  array<f32>;

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
  } else {
    // Mode 0: h1 anomaly relative to thermal equilibrium
    let h    = field[h1o + j*nx + i];
    let h_eq = -params.A_eq * cos(2.0 * PI * f32(j) / f32(ny));
    val = (h - h_eq) * params.gain;
  }

  return vec4f(colormap(clamp(val, -1.0, 1.0)), 1.0);
}
`,me=`// Two-layer Lax-Friedrichs SWE with Coriolis, beta-plane, and Newtonian relaxation.
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
  _pad2:   u32,  // [15]
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
`,ne=8;function _e(t,i,s,l,u){const n=t.createComputePipeline({label:"lf",layout:"auto",compute:{module:t.createShaderModule({label:"lf-compute",code:me}),entryPoint:"cs_main"}}),d=n.getBindGroupLayout(0),g=[t.createBindGroup({label:"lf A→B",layout:d,entries:[{binding:0,resource:{buffer:s}},{binding:1,resource:{buffer:i[0]}},{binding:2,resource:{buffer:i[1]}}]}),t.createBindGroup({label:"lf B→A",layout:d,entries:[{binding:0,resource:{buffer:s}},{binding:1,resource:{buffer:i[1]}},{binding:2,resource:{buffer:i[0]}}]})];let o=0;return{tick(c,r){const a=c.beginComputePass({label:"lf"});a.setPipeline(n);for(let _=0;_<r;_++)a.setBindGroup(0,g[o]),a.dispatchWorkgroups(l/ne,u/ne),o=1-o;a.end()},get readBuf(){return o},setCur(c){o=c},destroy(){}}}const he=`// Two-layer SWE right-hand side: centered differences, no spatial averaging.
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
  _pad2:   u32,  // [15]
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
  dst[h2o + j*nx + i] = -params.H2 * ((u2_r - u2_l) + (v2_u - v2_d)) * inv2dx
                        - params.nu4 * biharm(h2o, i, j, nx, ny);
  dst[u2o + j*nx + i] = -g * dsurf_dx - params.g_prime * dh2_dx + f_eff * v2_c
                        - params.nu4 * biharm(u2o, i, j, nx, ny);
  dst[v2o + j*nx + i] = -g * dsurf_dy - params.g_prime * dh2_dy - f_eff * u2_c
                        - params.nu4 * biharm(v2o, i, j, nx, ny);
}
`,xe=`// RK4 stage blend and accumulate.
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
`,te=8,be=256;function ve(t,i,s,l,u){const n=l*u*6*4,d=l*u*6,g=Math.ceil(d/be),o=t.createBuffer({label:"rk4-k",size:n,usage:GPUBufferUsage.STORAGE}),c=t.createBuffer({label:"rk4-stage",size:n,usage:GPUBufferUsage.STORAGE}),r=t.createBuffer({label:"rk4-acc",size:n,usage:GPUBufferUsage.STORAGE}),a=GPUShaderStage.COMPUTE,_=t.createBindGroupLayout({entries:[{binding:0,visibility:a,buffer:{type:"uniform"}},{binding:1,visibility:a,buffer:{type:"read-only-storage"}},{binding:2,visibility:a,buffer:{type:"storage"}}]}),x=t.createBindGroupLayout({entries:[{binding:0,visibility:a,buffer:{type:"uniform"}},{binding:1,visibility:a,buffer:{type:"read-only-storage"}},{binding:2,visibility:a,buffer:{type:"read-only-storage"}},{binding:3,visibility:a,buffer:{type:"storage"}},{binding:4,visibility:a,buffer:{type:"storage"}}]}),k=t.createPipelineLayout({bindGroupLayouts:[_]}),v=t.createPipelineLayout({bindGroupLayouts:[x]}),E=t.createShaderModule({label:"rhs",code:he}),y=t.createShaderModule({label:"blend",code:xe}),j=t.createComputePipeline({label:"rhs",layout:k,compute:{module:E,entryPoint:"cs_main"}}),w=t.createComputePipeline({label:"rk4-s1",layout:v,compute:{module:y,entryPoint:"cs_stage1"}}),G=t.createComputePipeline({label:"rk4-s2",layout:v,compute:{module:y,entryPoint:"cs_stage2"}}),H=t.createComputePipeline({label:"rk4-s3",layout:v,compute:{module:y,entryPoint:"cs_stage3"}}),L=t.createComputePipeline({label:"rk4-fin",layout:v,compute:{module:y,entryPoint:"cs_finalize"}}),S=[t.createBindGroup({label:"rhs-0",layout:_,entries:[{binding:0,resource:{buffer:s}},{binding:1,resource:{buffer:i[0]}},{binding:2,resource:{buffer:o}}]}),t.createBindGroup({label:"rhs-1",layout:_,entries:[{binding:0,resource:{buffer:s}},{binding:1,resource:{buffer:i[1]}},{binding:2,resource:{buffer:o}}]})],B=t.createBindGroup({label:"rhs-stage",layout:_,entries:[{binding:0,resource:{buffer:s}},{binding:1,resource:{buffer:c}},{binding:2,resource:{buffer:o}}]}),f=[t.createBindGroup({label:"blend-0",layout:x,entries:[{binding:0,resource:{buffer:s}},{binding:1,resource:{buffer:i[0]}},{binding:2,resource:{buffer:o}},{binding:3,resource:{buffer:c}},{binding:4,resource:{buffer:r}}]}),t.createBindGroup({label:"blend-1",layout:x,entries:[{binding:0,resource:{buffer:s}},{binding:1,resource:{buffer:i[1]}},{binding:2,resource:{buffer:o}},{binding:3,resource:{buffer:c}},{binding:4,resource:{buffer:r}}]})],A=[t.createBindGroup({label:"fin-0",layout:x,entries:[{binding:0,resource:{buffer:s}},{binding:1,resource:{buffer:i[0]}},{binding:2,resource:{buffer:o}},{binding:3,resource:{buffer:i[1]}},{binding:4,resource:{buffer:r}}]}),t.createBindGroup({label:"fin-1",layout:x,entries:[{binding:0,resource:{buffer:s}},{binding:1,resource:{buffer:i[1]}},{binding:2,resource:{buffer:o}},{binding:3,resource:{buffer:i[0]}},{binding:4,resource:{buffer:r}}]})];let m=0;const h=[l/te,u/te];function T(p){const b=m;let e=p.beginComputePass({label:"k1"});e.setPipeline(j),e.setBindGroup(0,S[b]),e.dispatchWorkgroups(h[0],h[1]),e.end(),e=p.beginComputePass({label:"s1"}),e.setPipeline(w),e.setBindGroup(0,f[b]),e.dispatchWorkgroups(g),e.end(),e=p.beginComputePass({label:"k2"}),e.setPipeline(j),e.setBindGroup(0,B),e.dispatchWorkgroups(h[0],h[1]),e.end(),e=p.beginComputePass({label:"s2"}),e.setPipeline(G),e.setBindGroup(0,f[b]),e.dispatchWorkgroups(g),e.end(),e=p.beginComputePass({label:"k3"}),e.setPipeline(j),e.setBindGroup(0,B),e.dispatchWorkgroups(h[0],h[1]),e.end(),e=p.beginComputePass({label:"s3"}),e.setPipeline(H),e.setBindGroup(0,f[b]),e.dispatchWorkgroups(g),e.end(),e=p.beginComputePass({label:"k4"}),e.setPipeline(j),e.setBindGroup(0,B),e.dispatchWorkgroups(h[0],h[1]),e.end(),e=p.beginComputePass({label:"rk4-fin"}),e.setPipeline(L),e.setBindGroup(0,A[b]),e.dispatchWorkgroups(g),e.end(),m=1-m}return{tick(p,b){for(let e=0;e<b;e++)T(p)},get readBuf(){return m},setCur(p){m=p},destroy(){o.destroy(),c.destroy(),r.destroy()}}}const I=256,q=256,re=.4,ie=64,ae={"lax-friedrichs":_e,rk4:ve};function ye(t,i,s="lax-friedrichs"){const l=I*q,u=l*6*4,n=[0,1].map(e=>t.createBuffer({label:`state-${e}`,size:u,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST})),d=t.createBuffer({label:"params",size:ie,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),g=t.createShaderModule({label:"render",code:ge}),o=t.createRenderPipeline({label:"render",layout:"auto",vertex:{module:g,entryPoint:"vs"},fragment:{module:g,entryPoint:"fs",targets:[{format:i}]},primitive:{topology:"triangle-list"}}),c=o.getBindGroupLayout(0),r=n.map((e,P)=>t.createBindGroup({label:`render-${P}`,layout:c,entries:[{binding:0,resource:{buffer:d}},{binding:1,resource:{buffer:e}}]}));let a=0,_=0,x=.25,k=.25,v=.02,E=.1,y=.01,j=.3,w=300,G=0,H=5,L=0;const S=new ArrayBuffer(ie),B=new Uint32Array(S),f=new Float32Array(S);B[0]=I,B[1]=q,f[2]=1,f[3]=re;function A(){t.queue.writeBuffer(n[0],0,je(I,q,E)),t.queue.writeBuffer(n[1],0,new Float32Array(l*6))}A();let m=ae[s](t,n,d,I,q);function h(e,P=1){B[4]=a,f[5]=_,f[6]=x,f[7]=y,f[8]=j,f[9]=w,f[10]=G,f[11]=H,B[12]=L,f[13]=k,f[14]=v,t.queue.writeBuffer(d,0,S),m.tick(e,P),a+=P}function T(e,P){const O=e.beginRenderPass({colorAttachments:[{view:P,clearValue:{r:.03,g:.18,b:.42,a:1},loadOp:"clear",storeOp:"store"}]});O.setPipeline(o),O.setBindGroup(0,r[m.readBuf]),O.draw(6),O.end()}function p(){a=0,A(),m.setCur(0)}function b(e){const P=m.readBuf;m.destroy(),m=ae[e](t,n,d,I,q),m.setCur(P)}return{tick:h,render:T,reset:p,setScheme:b,setF(e){_=e},setH(e){x=e},setH2(e){k=e},setGPrime(e){v=e},setSigma(e){E=e},setNu4(e){y=e},setAEq(e){j=e},setTau(e){w=e},setBeta(e){G=e},setGain(e){H=e},setMode(e){L=e},get step(){return a},get simTime(){return a*re},get f(){return _},get H(){return x},get H2(){return k},get g_prime(){return v},get sigma(){return E},get nu4(){return y},get A_eq(){return j},get tau(){return w},get beta(){return G},get gain(){return H},get mode(){return L}}}function je(t,i,s){const l=new Float32Array(t*i*6),u=t/2,n=i/2,d=s*t,g=s*i;for(let o=0;o<i;o++)for(let c=0;c<t;c++){const r=(c-u)/d,a=(o-n)/g;l[o*t+c]=Math.exp(-(r*r+a*a))}return l}const N=document.getElementById("status"),ue=document.getElementById("pause-btn"),Be=document.getElementById("reset-btn"),oe=document.getElementById("scheme-select"),ee=document.getElementById("speed-slider"),Pe=document.getElementById("speed-val"),R=document.getElementById("sigma-slider"),se=document.getElementById("sigma-val"),C=document.getElementById("H-slider"),Ce=document.getElementById("H-val"),F=document.getElementById("H2-slider"),Fe=document.getElementById("H2-val"),ke=document.getElementById("cbt-val"),z=document.getElementById("gp-slider"),Ee=document.getElementById("gp-val"),we=document.getElementById("c2-val"),U=document.getElementById("f-slider"),Ge=document.getElementById("f-val"),He=document.getElementById("lr-val"),D=document.getElementById("nu4-slider"),le=document.getElementById("nu4-val"),K=document.getElementById("Aeq-slider"),de=document.getElementById("Aeq-val"),Y=document.getElementById("tau-slider"),ce=document.getElementById("tau-val"),$=document.getElementById("beta-slider"),fe=document.getElementById("beta-val"),X=document.getElementById("display-select"),J=document.getElementById("gain-slider"),pe=document.getElementById("gain-val");let M=!1,Q=parseInt(ee.value);function Z(t,i){Ce.textContent=t.toFixed(2),Fe.textContent=i.toFixed(2),ke.textContent=Math.sqrt(t+i).toFixed(2)}function V(t,i,s){Ee.textContent=t.toFixed(2);const l=i>0&&s>0?Math.sqrt(t*i*s/(i+s)):0;we.textContent=l.toFixed(2)}function W(t,i,s){const l=Math.sqrt(i+s);Ge.textContent=t.toFixed(3),He.textContent=t>0?(l/t).toFixed(1):"∞"}async function Le(){var c;if(!navigator.gpu){N.textContent="WebGPU not supported — use Chrome 113+ or Safari 18+";return}const t=await navigator.gpu.requestAdapter();if(!t){N.textContent="No GPU adapter found";return}const i=await t.requestDevice(),l=document.getElementById("canvas").getContext("webgpu"),u=navigator.gpu.getPreferredCanvasFormat();l.configure({device:i,format:u,alphaMode:"opaque"});const n=ye(i,u),d=t.info??await((c=t.requestAdapterInfo)==null?void 0:c.call(t))??{},g=d.description||d.vendor||"gpu";se.textContent=n.sigma.toFixed(2),R.value=n.sigma,C.value=n.H,F.value=n.H2,z.value=n.g_prime,Z(n.H,n.H2),V(n.g_prime,n.H,n.H2),W(n.f,n.H,n.H2),D.value=n.nu4,le.textContent=n.nu4.toFixed(3),K.value=n.A_eq,de.textContent=n.A_eq.toFixed(2),Y.value=n.tau,ce.textContent=n.tau.toFixed(0),$.value=n.beta,fe.textContent=n.beta.toFixed(3),J.value=n.gain,pe.textContent=n.gain.toFixed(0),X.value=n.mode.toString(),ue.addEventListener("click",()=>{M=!M,ue.textContent=M?"play":"pause"}),ee.addEventListener("input",()=>{Q=parseInt(ee.value),Pe.textContent=Q}),R.addEventListener("input",()=>{const r=parseFloat(R.value);se.textContent=r.toFixed(2),n.setSigma(r),n.reset()}),C.addEventListener("input",()=>{const r=parseFloat(C.value),a=parseFloat(F.value);n.setH(r),Z(r,a),W(parseFloat(U.value),r,a),V(parseFloat(z.value),r,a)}),F.addEventListener("input",()=>{const r=parseFloat(F.value),a=parseFloat(C.value);n.setH2(r),Z(a,r),W(parseFloat(U.value),a,r),V(parseFloat(z.value),a,r)}),z.addEventListener("input",()=>{const r=parseFloat(z.value);n.setGPrime(r),V(r,parseFloat(C.value),parseFloat(F.value))}),U.addEventListener("input",()=>{const r=parseFloat(U.value);n.setF(r),W(r,parseFloat(C.value),parseFloat(F.value))}),D.addEventListener("input",()=>{const r=parseFloat(D.value);n.setNu4(r),le.textContent=r.toFixed(3)}),K.addEventListener("input",()=>{const r=parseFloat(K.value);n.setAEq(r),de.textContent=r.toFixed(2)}),Y.addEventListener("input",()=>{const r=parseFloat(Y.value);n.setTau(r),ce.textContent=r.toFixed(0)}),$.addEventListener("input",()=>{const r=parseFloat($.value);n.setBeta(r),fe.textContent=r.toFixed(3)}),J.addEventListener("input",()=>{const r=parseFloat(J.value);n.setGain(r),pe.textContent=r.toFixed(0)}),X.addEventListener("change",()=>{n.setMode(parseInt(X.value))}),Be.addEventListener("click",()=>n.reset()),oe.addEventListener("change",()=>{n.setScheme(oe.value)});function o(){if(!M){const r=i.createCommandEncoder({label:"frame"});n.tick(r,Q),n.render(r,l.getCurrentTexture().createView()),i.queue.submit([r.finish()])}N.textContent=`${g} | 256×256 | t = ${n.simTime.toFixed(1)}`,requestAnimationFrame(o)}requestAnimationFrame(o)}Le().catch(t=>{console.error(t),N.textContent=`Error: ${t.message}`});
