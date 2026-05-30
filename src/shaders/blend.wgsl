// RK4 stage blend and accumulate.
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
// Operates element-wise over the flat [h | u | v] array — no field-layout
// knowledge needed. Dispatched 1D over all 3·NX·NY elements.

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
  if (idx >= params.width * params.height * 3u) { return; }
  buf_out[idx] = buf_n[idx] + (params.dt * 0.5) * buf_k[idx];
  buf_acc[idx] = buf_k[idx];
}

@compute @workgroup_size(256)
fn cs_stage2(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  if (idx >= params.width * params.height * 3u) { return; }
  buf_out[idx] = buf_n[idx] + (params.dt * 0.5) * buf_k[idx];
  buf_acc[idx] += 2.0 * buf_k[idx];
}

@compute @workgroup_size(256)
fn cs_stage3(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  if (idx >= params.width * params.height * 3u) { return; }
  buf_out[idx] = buf_n[idx] + params.dt * buf_k[idx];
  buf_acc[idx] += 2.0 * buf_k[idx];
}

@compute @workgroup_size(256)
fn cs_finalize(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  if (idx >= params.width * params.height * 3u) { return; }
  buf_out[idx] = buf_n[idx] + (params.dt / 6.0) * (buf_acc[idx] + buf_k[idx]);
}
