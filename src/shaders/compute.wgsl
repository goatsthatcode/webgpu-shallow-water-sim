// Two-layer Lax-Friedrichs SWE with Coriolis, beta-plane, and Newtonian relaxation.
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
