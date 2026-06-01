// Fullscreen quad renderer. Seven display modes:
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
