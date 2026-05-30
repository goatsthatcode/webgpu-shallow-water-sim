// Pure SWE right-hand side: centered differences, no spatial averaging.
// Used by RK4. Lax-Friedrichs uses compute.wgsl, which fuses RHS + LF averaging.
//
//   dh/dt = -H·(∂u/∂x + ∂v/∂y)  − ν₄·∇⁴h  − (h−h_eq)/τ
//   du/dt = -g·∂h/∂x + f_eff·v  − ν₄·∇⁴u
//   dv/dt = -g·∂h/∂y − f_eff·u  − ν₄·∇⁴v
//
//   f_eff(j) = f + β·(j − ny/2)          (beta-plane)
//   h_eq(j)  = −A_eq·cos(2π·j/ny)        (warm equator at j=ny/2, cold poles j=0,ny)

struct Params {
  width:  u32,  // [0]
  height: u32,  // [1]
  dx:     f32,  // [2]
  dt:     f32,  // [3]
  step:   u32,  // [4]
  f:      f32,  // [5]  base Coriolis
  H:      f32,  // [6]  mean depth
  nu4:    f32,  // [7]  hyperdiffusion coefficient
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

// 13-point biharmonic stencil ∇⁴φ with periodic wrap, dx=1.
// Coefficients: 20·c − 8·(±1 cardinal) + 2·(diagonals) + (±2 cardinal)
fn biharm(off: u32, ci: u32, cj: u32, nx: u32, ny: u32) -> f32 {
  let il  = (ci + nx - 1u) % nx;
  let ir  = (ci + 1u)      % nx;
  let il2 = (ci + nx - 2u) % nx;
  let ir2 = (ci + 2u)      % nx;
  let jd  = (cj + ny - 1u) % ny;
  let ju  = (cj + 1u)      % ny;
  let jd2 = (cj + ny - 2u) % ny;
  let ju2 = (cj + 2u)      % ny;

  let c00  = src[off + cj  * nx + ci ];
  let p10  = src[off + cj  * nx + ir ];
  let m10  = src[off + cj  * nx + il ];
  let p01  = src[off + ju  * nx + ci ];
  let m01  = src[off + jd  * nx + ci ];
  let p20  = src[off + cj  * nx + ir2];
  let m20  = src[off + cj  * nx + il2];
  let p02  = src[off + ju2 * nx + ci ];
  let m02  = src[off + jd2 * nx + ci ];
  let p11  = src[off + ju  * nx + ir ];
  let p1m1 = src[off + jd  * nx + ir ];
  let m11  = src[off + ju  * nx + il ];
  let m1m1 = src[off + jd  * nx + il ];

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

  let il = (i + nx - 1u) % nx;
  let ir = (i + 1u)      % nx;
  let jd = (j + ny - 1u) % ny;
  let ju = (j + 1u)      % ny;

  let ho = h_off(nx, ny);
  let uo = u_off(nx, ny);
  let vo = v_off(nx, ny);

  let h_c = src[ho + j  * nx + i ];
  let h_r = src[ho + j  * nx + ir];
  let h_l = src[ho + j  * nx + il];
  let h_u = src[ho + ju * nx + i ];
  let h_d = src[ho + jd * nx + i ];

  let u_c = src[uo + j  * nx + i ];
  let u_r = src[uo + j  * nx + ir];
  let u_l = src[uo + j  * nx + il];

  let v_c = src[vo + j  * nx + i ];
  let v_u = src[vo + ju * nx + i ];
  let v_d = src[vo + jd * nx + i ];

  let inv2dx = 1.0 / (2.0 * params.dx);

  let f_eff    = params.f + params.beta * (f32(j) - f32(ny) * 0.5);
  let h_eq     = -params.A_eq * cos(2.0 * PI * f32(j) / f32(ny));
  let dh_relax = select(0.0, -(h_c - h_eq) / params.tau, params.tau > 0.0);

  dst[ho + j * nx + i] = -params.H * ((u_r - u_l) + (v_u - v_d)) * inv2dx
                          - params.nu4 * biharm(ho, i, j, nx, ny)
                          + dh_relax;
  dst[uo + j * nx + i] = -g * (h_r - h_l) * inv2dx + f_eff * v_c
                          - params.nu4 * biharm(uo, i, j, nx, ny);
  dst[vo + j * nx + i] = -g * (h_u - h_d) * inv2dx - f_eff * u_c
                          - params.nu4 * biharm(vo, i, j, nx, ny);
}
