// Fullscreen quad + diverging colormap that reads the simulation buffer directly.

struct Params {
  width:  u32,
  height: u32,
  dx:     f32,
  dt:     f32,
  frame:  u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> field: array<f32>;

struct VertOut {
  @builtin(position) pos: vec4f,
  @location(0)       uv:  vec2f,
}

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VertOut {
  // two triangles covering clip space (-1..+1)
  var corners = array<vec2f, 6>(
    vec2f(-1.0, -1.0),
    vec2f( 1.0, -1.0),
    vec2f(-1.0,  1.0),
    vec2f(-1.0,  1.0),
    vec2f( 1.0, -1.0),
    vec2f( 1.0,  1.0),
  );
  let p = corners[vi];

  var out: VertOut;
  out.pos = vec4f(p, 0.0, 1.0);
  // Map clip-space to UV with y flipped so (0,0) is the top-left of the field.
  out.uv  = vec2f((p.x + 1.0) * 0.5, (1.0 - p.y) * 0.5);
  return out;
}

// Diverging blue-white-red colormap, input expected in [-1, 1].
fn colormap(x: f32) -> vec3f {
  let t = clamp(x, -1.0, 1.0);
  let cold = vec3f(0.10, 0.30, 0.85);
  let mid  = vec3f(0.97, 0.97, 0.97);
  let hot  = vec3f(0.85, 0.20, 0.10);
  if (t < 0.0) {
    return mix(cold, mid, t + 1.0);
  }
  return mix(mid, hot, t);
}

@fragment
fn fs(in: VertOut) -> @location(0) vec4f {
  let i = u32(clamp(in.uv.x, 0.0, 0.9999) * f32(params.width));
  let j = u32(clamp(in.uv.y, 0.0, 0.9999) * f32(params.height));
  let v = field[j * params.width + i];
  return vec4f(colormap(v), 1.0);
}
