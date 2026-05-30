// Two-layer SWE right-hand side: centered differences, no spatial averaging.
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
