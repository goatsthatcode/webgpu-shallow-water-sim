// Fullscreen quad renderer. Supports three display modes:
//   mode 0 — h anomaly: h − h_eq(j)   (removes the thermal background, shows waves)
//   mode 1 — relative vorticity: ζ = ∂v/∂x − ∂u/∂y  (clearest for Rossby waves)
//   mode 2 — raw h                     (unmodified height field)

struct Params {
  width:  u32,  // [0]
  height: u32,  // [1]
  dx:     f32,  // [2]
  dt:     f32,  // [3]
  step:   u32,  // [4]
  f:      f32,  // [5]
  H:      f32,  // [6]
  nu4:    f32,  // [7]
  A_eq:   f32,  // [8]
  tau:    f32,  // [9]
  beta:   f32,  // [10]
  gain:   f32,  // [11]  colormap amplification
  mode:   u32,  // [12]  0=h_anom, 1=vorticity
}

@group(0) @binding(0) var<uniform>             params: Params;
@group(0) @binding(1) var<storage, read>       field:  array<f32>;

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

// Diverging aqua-planet colormap, input in [-1, 1].
fn colormap(t: f32) -> vec3f {
  let trough = vec3f(0.08, 0.02, 0.35); // deep indigo
  let base   = vec3f(0.03, 0.18, 0.42); // ocean blue
  let crest  = vec3f(0.72, 0.96, 0.98); // bright seafoam
  if (t < 0.0) { return mix(base, trough, clamp(-t, 0.0, 1.0)); }
  return mix(base, crest, clamp(t, 0.0, 1.0));
}

@fragment
fn fs(in: VertOut) -> @location(0) vec4f {
  let nx = params.width;
  let ny = params.height;
  let i  = u32(clamp(in.uv.x, 0.0, 0.9999) * f32(nx));
  let j  = u32(clamp(in.uv.y, 0.0, 0.9999) * f32(ny));

  let il = (i + nx - 1u) % nx;
  let ir = (i + 1u)      % nx;
  let jd = (j + ny - 1u) % ny;
  let ju = (j + 1u)      % ny;

  var val: f32;
  if params.mode == 1u {
    // Relative vorticity ζ = (∂v/∂x − ∂u/∂y) · gain
    let uo   = nx * ny;
    let vo   = 2u * nx * ny;
    let dv_dx = field[vo + j  * nx + ir] - field[vo + j  * nx + il];
    let du_dy = field[uo + ju * nx + i ] - field[uo + jd * nx + i ];
    val = (dv_dx - du_dy) * 0.5 * params.gain;
  } else if params.mode == 2u {
    val = field[j * nx + i] * params.gain;
  } else {
    // h anomaly relative to thermal equilibrium
    let h    = field[j * nx + i];
    let h_eq = -params.A_eq * cos(2.0 * PI * f32(j) / f32(ny));
    val = (h - h_eq) * params.gain;
  }

  return vec4f(colormap(clamp(val, -1.0, 1.0)), 1.0);
}
