// Lax-Friedrichs SWE with Coriolis, beta-plane, and Newtonian relaxation.
//
// State: h (height perturbation), u (x-velocity), v (y-velocity)
// packed sequentially: [h block | u block | v block]
//
// Linearized SWE:
//   ∂h/∂t = -H(∂u/∂x + ∂v/∂y)  − (h−h_eq)/τ
//   ∂u/∂t = -g ∂h/∂x + f_eff·v
//   ∂v/∂t = -g ∂h/∂y − f_eff·u
//
// Lax-Friedrichs: replace h_i with spatial average before subtracting flux.
// Numerical diffusion coefficient dx²/(4dt)·∇² is baked in — the averaging puts
// |G(k=π)| = 1 exactly at Nyquist. Explicit ∇⁴ hyperdiffusion would push
// |G| > 1 there and blow up immediately, so nu4 is intentionally NOT applied here.
// Use RK4 (which has zero numerical diffusion) if you want ν₄ to act.
//
// Stability: c·dt·(1/dx + 1/dy) ≤ 1 → H ≤ (dx/(2·dt))² = 1.56 with dx=1, dt=0.4.

struct Params {
  width:  u32,  // [0]
  height: u32,  // [1]
  dx:     f32,  // [2]
  dt:     f32,  // [3]
  step:   u32,  // [4]
  f:      f32,  // [5]  base Coriolis
  H:      f32,  // [6]  mean depth
  nu4:    f32,  // [7]  (unused in LF — see comment above)
  A_eq:   f32,  // [8]  thermal forcing amplitude
  tau:    f32,  // [9]  Newtonian relaxation timescale (time units)
  beta:   f32,  // [10] beta-plane parameter
  _pad:   u32,  // [11]
}

@group(0) @binding(0) var<uniform>             params: Params;
@group(0) @binding(1) var<storage, read>       src:    array<f32>;
@group(0) @binding(2) var<storage, read_write> dst:    array<f32>;

const g:  f32 = 1.0;
const PI: f32 = 3.14159265358979;

fn h_off(nx: u32, ny: u32) -> u32 { return 0u; }
fn u_off(nx: u32, ny: u32) -> u32 { return nx * ny; }
fn v_off(nx: u32, ny: u32) -> u32 { return 2u * nx * ny; }

@compute @workgroup_size(8, 8)
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
  let i  = gid.x;
  let j  = gid.y;
  let nx = params.width;
  let ny = params.height;
  if (i >= nx || j >= ny) { return; }

  let il = (i + nx - 1u) % nx;
  let ir = (i + 1u)      % nx;
  let jd = (j + ny - 1u) % ny;
  let ju = (j + 1u)      % ny;

  let ho = h_off(nx, ny);
  let uo = u_off(nx, ny);
  let vo = v_off(nx, ny);

  let u_c  = src[uo + j  * nx + i ];
  let v_c  = src[vo + j  * nx + i ];

  let h_c  = src[ho + j  * nx + i ];
  let h_r  = src[ho + j  * nx + ir];
  let h_l  = src[ho + j  * nx + il];
  let h_u  = src[ho + ju * nx + i ];
  let h_d  = src[ho + jd * nx + i ];

  let u_r  = src[uo + j  * nx + ir];
  let u_l  = src[uo + j  * nx + il];
  let u_u  = src[uo + ju * nx + i ];
  let u_d  = src[uo + jd * nx + i ];

  let v_r  = src[vo + j  * nx + ir];
  let v_l  = src[vo + j  * nx + il];
  let v_u  = src[vo + ju * nx + i ];
  let v_d  = src[vo + jd * nx + i ];

  let h_avg = 0.25 * (h_r + h_l + h_u + h_d);
  let u_avg = 0.25 * (u_r + u_l + u_u + u_d);
  let v_avg = 0.25 * (v_r + v_l + v_u + v_d);

  let c = params.dt / (2.0 * params.dx);

  let f_eff    = params.f + params.beta * (f32(j) - f32(ny) * 0.5);
  let h_eq     = -params.A_eq * cos(2.0 * PI * f32(j) / f32(ny));
  let dh_relax = select(0.0, -(h_c - h_eq) / params.tau, params.tau > 0.0);

  dst[ho + j * nx + i] = h_avg
                          - params.H * c * (u_r - u_l)
                          - params.H * c * (v_u - v_d)
                          + params.dt * dh_relax;
  dst[uo + j * nx + i] = u_avg
                          - g * c * (h_r - h_l)
                          + params.dt * f_eff * v_c;
  dst[vo + j * nx + i] = v_avg
                          - g * c * (h_u - h_d)
                          - params.dt * f_eff * u_c;
}
