// Passive moisture tracer advected by layer-1 velocity using first-order upwind.
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
