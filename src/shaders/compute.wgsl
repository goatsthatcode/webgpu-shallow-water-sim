// Stage 1 compute shader: identity copy from src → dst.
// Future stages replace the body with the SWE update; the bind layout stays the same.

struct Params {
  width:  u32,
  height: u32,
  dx:     f32,
  dt:     f32,
  frame:  u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read>       src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(8, 8)
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  let j = gid.y;
  if (i >= params.width || j >= params.height) { return; }

  let idx = j * params.width + i;
  dst[idx] = src[idx];
}
