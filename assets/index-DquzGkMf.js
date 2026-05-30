(function(){const a=document.createElement("link").relList;if(a&&a.supports&&a.supports("modulepreload"))return;for(const o of document.querySelectorAll('link[rel="modulepreload"]'))f(o);new MutationObserver(o=>{for(const n of o)if(n.type==="childList")for(const d of n.addedNodes)d.tagName==="LINK"&&d.rel==="modulepreload"&&f(d)}).observe(document,{childList:!0,subtree:!0});function l(o){const n={};return o.integrity&&(n.integrity=o.integrity),o.referrerPolicy&&(n.referrerPolicy=o.referrerPolicy),o.crossOrigin==="use-credentials"?n.credentials="include":o.crossOrigin==="anonymous"?n.credentials="omit":n.credentials="same-origin",n}function f(o){if(o.ep)return;o.ep=!0;const n=l(o);fetch(o.href,n)}})();const le=`// Fullscreen quad renderer. Supports three display modes:
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
`,de=`// Lax-Friedrichs SWE with Coriolis, beta-plane, and Newtonian relaxation.
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
`,$=8;function ce(e,a,l,f,o){const n=e.createComputePipeline({label:"lf",layout:"auto",compute:{module:e.createShaderModule({label:"lf-compute",code:de}),entryPoint:"cs_main"}}),d=n.getBindGroupLayout(0),g=[e.createBindGroup({label:"lf A→B",layout:d,entries:[{binding:0,resource:{buffer:l}},{binding:1,resource:{buffer:a[0]}},{binding:2,resource:{buffer:a[1]}}]}),e.createBindGroup({label:"lf B→A",layout:d,entries:[{binding:0,resource:{buffer:l}},{binding:1,resource:{buffer:a[1]}},{binding:2,resource:{buffer:a[0]}}]})];let u=0;return{tick(c,r){const s=c.beginComputePass({label:"lf"});s.setPipeline(n);for(let m=0;m<r;m++)s.setBindGroup(0,g[u]),s.dispatchWorkgroups(f/$,o/$),u=1-u;s.end()},get readBuf(){return u},setCur(c){u=c},destroy(){}}}const fe=`// Pure SWE right-hand side: centered differences, no spatial averaging.
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
`,pe=`// RK4 stage blend and accumulate.
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
`,X=8,ge=256;function me(e,a,l,f,o){const n=f*o*3*4,d=f*o*3,g=Math.ceil(d/ge),u=e.createBuffer({label:"rk4-k",size:n,usage:GPUBufferUsage.STORAGE}),c=e.createBuffer({label:"rk4-stage",size:n,usage:GPUBufferUsage.STORAGE}),r=e.createBuffer({label:"rk4-acc",size:n,usage:GPUBufferUsage.STORAGE}),s=GPUShaderStage.COMPUTE,m=e.createBindGroupLayout({entries:[{binding:0,visibility:s,buffer:{type:"uniform"}},{binding:1,visibility:s,buffer:{type:"read-only-storage"}},{binding:2,visibility:s,buffer:{type:"storage"}}]}),_=e.createBindGroupLayout({entries:[{binding:0,visibility:s,buffer:{type:"uniform"}},{binding:1,visibility:s,buffer:{type:"read-only-storage"}},{binding:2,visibility:s,buffer:{type:"read-only-storage"}},{binding:3,visibility:s,buffer:{type:"storage"}},{binding:4,visibility:s,buffer:{type:"storage"}}]}),C=e.createPipelineLayout({bindGroupLayouts:[m]}),y=e.createPipelineLayout({bindGroupLayouts:[_]}),w=e.createShaderModule({label:"rhs",code:fe}),v=e.createShaderModule({label:"blend",code:pe}),j=e.createComputePipeline({label:"rhs",layout:C,compute:{module:w,entryPoint:"cs_main"}}),E=e.createComputePipeline({label:"rk4-s1",layout:y,compute:{module:v,entryPoint:"cs_stage1"}}),G=e.createComputePipeline({label:"rk4-s2",layout:y,compute:{module:v,entryPoint:"cs_stage2"}}),F=e.createComputePipeline({label:"rk4-s3",layout:y,compute:{module:v,entryPoint:"cs_stage3"}}),P=e.createComputePipeline({label:"rk4-fin",layout:y,compute:{module:v,entryPoint:"cs_finalize"}}),b=[e.createBindGroup({label:"rhs-0",layout:m,entries:[{binding:0,resource:{buffer:l}},{binding:1,resource:{buffer:a[0]}},{binding:2,resource:{buffer:u}}]}),e.createBindGroup({label:"rhs-1",layout:m,entries:[{binding:0,resource:{buffer:l}},{binding:1,resource:{buffer:a[1]}},{binding:2,resource:{buffer:u}}]})],k=e.createBindGroup({label:"rhs-stage",layout:m,entries:[{binding:0,resource:{buffer:l}},{binding:1,resource:{buffer:c}},{binding:2,resource:{buffer:u}}]}),x=[e.createBindGroup({label:"blend-0",layout:_,entries:[{binding:0,resource:{buffer:l}},{binding:1,resource:{buffer:a[0]}},{binding:2,resource:{buffer:u}},{binding:3,resource:{buffer:c}},{binding:4,resource:{buffer:r}}]}),e.createBindGroup({label:"blend-1",layout:_,entries:[{binding:0,resource:{buffer:l}},{binding:1,resource:{buffer:a[1]}},{binding:2,resource:{buffer:u}},{binding:3,resource:{buffer:c}},{binding:4,resource:{buffer:r}}]})],H=[e.createBindGroup({label:"fin-0",layout:_,entries:[{binding:0,resource:{buffer:l}},{binding:1,resource:{buffer:a[0]}},{binding:2,resource:{buffer:u}},{binding:3,resource:{buffer:a[1]}},{binding:4,resource:{buffer:r}}]}),e.createBindGroup({label:"fin-1",layout:_,entries:[{binding:0,resource:{buffer:l}},{binding:1,resource:{buffer:a[1]}},{binding:2,resource:{buffer:u}},{binding:3,resource:{buffer:a[0]}},{binding:4,resource:{buffer:r}}]})];let B=0;const h=[f/X,o/X];function U(i){const p=B;let t=i.beginComputePass({label:"k1"});t.setPipeline(j),t.setBindGroup(0,b[p]),t.dispatchWorkgroups(h[0],h[1]),t.end(),t=i.beginComputePass({label:"s1"}),t.setPipeline(E),t.setBindGroup(0,x[p]),t.dispatchWorkgroups(g),t.end(),t=i.beginComputePass({label:"k2"}),t.setPipeline(j),t.setBindGroup(0,k),t.dispatchWorkgroups(h[0],h[1]),t.end(),t=i.beginComputePass({label:"s2"}),t.setPipeline(G),t.setBindGroup(0,x[p]),t.dispatchWorkgroups(g),t.end(),t=i.beginComputePass({label:"k3"}),t.setPipeline(j),t.setBindGroup(0,k),t.dispatchWorkgroups(h[0],h[1]),t.end(),t=i.beginComputePass({label:"s3"}),t.setPipeline(F),t.setBindGroup(0,x[p]),t.dispatchWorkgroups(g),t.end(),t=i.beginComputePass({label:"k4"}),t.setPipeline(j),t.setBindGroup(0,k),t.dispatchWorkgroups(h[0],h[1]),t.end(),t=i.beginComputePass({label:"rk4-fin"}),t.setPipeline(P),t.setBindGroup(0,H[p]),t.dispatchWorkgroups(g),t.end(),B=1-B}return{tick(i,p){for(let t=0;t<p;t++)U(i)},get readBuf(){return B},setCur(i){B=i},destroy(){u.destroy(),c.destroy(),r.destroy()}}}const S=256,q=256,J=.4,Q=64,Z={"lax-friedrichs":ce,rk4:me};function be(e,a,l="lax-friedrichs"){const f=S*q,o=f*3*4,n=[0,1].map(i=>e.createBuffer({label:`state-${i}`,size:o,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST})),d=e.createBuffer({label:"params",size:Q,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),g=e.createShaderModule({label:"render",code:le}),u=e.createRenderPipeline({label:"render",layout:"auto",vertex:{module:g,entryPoint:"vs"},fragment:{module:g,entryPoint:"fs",targets:[{format:a}]},primitive:{topology:"triangle-list"}}),c=u.getBindGroupLayout(0),r=n.map((i,p)=>e.createBindGroup({label:`render-${p}`,layout:c,entries:[{binding:0,resource:{buffer:d}},{binding:1,resource:{buffer:i}}]}));let s=0,m=0,_=.25,C=.1,y=.01,w=.3,v=300,j=0,E=5,G=0;const F=new ArrayBuffer(Q),P=new Uint32Array(F),b=new Float32Array(F);P[0]=S,P[1]=q,b[2]=1,b[3]=J;function k(){e.queue.writeBuffer(n[0],0,xe(S,q,C)),e.queue.writeBuffer(n[1],0,new Float32Array(f*3))}k();let x=Z[l](e,n,d,S,q);function H(i,p=1){P[4]=s,b[5]=m,b[6]=_,b[7]=y,b[8]=w,b[9]=v,b[10]=j,b[11]=E,P[12]=G,e.queue.writeBuffer(d,0,F),x.tick(i,p),s+=p}function B(i,p){const t=i.beginRenderPass({colorAttachments:[{view:p,clearValue:{r:.03,g:.18,b:.42,a:1},loadOp:"clear",storeOp:"store"}]});t.setPipeline(u),t.setBindGroup(0,r[x.readBuf]),t.draw(6),t.end()}function h(){s=0,k(),x.setCur(0)}function U(i){const p=x.readBuf;x.destroy(),x=Z[i](e,n,d,S,q),x.setCur(p)}return{tick:H,render:B,reset:h,setScheme:U,setF(i){m=i},setH(i){_=i},setSigma(i){C=i},setNu4(i){y=i},setAEq(i){w=i},setTau(i){v=i},setBeta(i){j=i},setGain(i){E=i},setMode(i){G=i},get step(){return s},get simTime(){return s*J},get f(){return m},get H(){return _},get sigma(){return C},get nu4(){return y},get A_eq(){return w},get tau(){return v},get beta(){return j},get gain(){return E},get mode(){return G}}}function xe(e,a,l){const f=new Float32Array(e*a*3),o=e/2,n=a/2,d=l*e,g=l*a;for(let u=0;u<a;u++)for(let c=0;c<e;c++){const r=(c-o)/d,s=(u-n)/g;f[u*e+c]=Math.exp(-(r*r+s*s))}return f}const A=document.getElementById("status"),ee=document.getElementById("pause-btn"),he=document.getElementById("reset-btn"),ne=document.getElementById("scheme-select"),Y=document.getElementById("speed-slider"),_e=document.getElementById("speed-val"),O=document.getElementById("sigma-slider"),te=document.getElementById("sigma-val"),L=document.getElementById("H-slider"),ye=document.getElementById("H-val"),ve=document.getElementById("c-val"),W=document.getElementById("f-slider"),je=document.getElementById("f-val"),Be=document.getElementById("lr-val"),M=document.getElementById("nu4-slider"),re=document.getElementById("nu4-val"),R=document.getElementById("Aeq-slider"),ie=document.getElementById("Aeq-val"),N=document.getElementById("tau-slider"),ae=document.getElementById("tau-val"),V=document.getElementById("beta-slider"),oe=document.getElementById("beta-val"),z=document.getElementById("display-select"),T=document.getElementById("gain-slider"),ue=document.getElementById("gain-val");let I=!1,D=parseInt(Y.value);function se(e){ye.textContent=e.toFixed(2),ve.textContent=Math.sqrt(e).toFixed(2)}function K(e,a){const l=Math.sqrt(a);je.textContent=e.toFixed(3),Be.textContent=e>0?(l/e).toFixed(1):"∞"}async function Pe(){var c;if(!navigator.gpu){A.textContent="WebGPU not supported — use Chrome 113+ or Safari 18+";return}const e=await navigator.gpu.requestAdapter();if(!e){A.textContent="No GPU adapter found";return}const a=await e.requestDevice(),f=document.getElementById("canvas").getContext("webgpu"),o=navigator.gpu.getPreferredCanvasFormat();f.configure({device:a,format:o,alphaMode:"opaque"});const n=be(a,o),d=e.info??await((c=e.requestAdapterInfo)==null?void 0:c.call(e))??{},g=d.description||d.vendor||"gpu";te.textContent=n.sigma.toFixed(2),O.value=n.sigma,L.value=n.H,se(n.H),K(n.f,n.H),M.value=n.nu4,re.textContent=n.nu4.toFixed(3),R.value=n.A_eq,ie.textContent=n.A_eq.toFixed(2),N.value=n.tau,ae.textContent=n.tau.toFixed(0),V.value=n.beta,oe.textContent=n.beta.toFixed(3),T.value=n.gain,ue.textContent=n.gain.toFixed(0),z.value=n.mode.toString(),ee.addEventListener("click",()=>{I=!I,ee.textContent=I?"play":"pause"}),Y.addEventListener("input",()=>{D=parseInt(Y.value),_e.textContent=D}),O.addEventListener("input",()=>{const r=parseFloat(O.value);te.textContent=r.toFixed(2),n.setSigma(r),n.reset()}),L.addEventListener("input",()=>{const r=parseFloat(L.value);n.setH(r),se(r),K(parseFloat(W.value),r)}),W.addEventListener("input",()=>{const r=parseFloat(W.value);n.setF(r),K(r,parseFloat(L.value))}),M.addEventListener("input",()=>{const r=parseFloat(M.value);n.setNu4(r),re.textContent=r.toFixed(3)}),R.addEventListener("input",()=>{const r=parseFloat(R.value);n.setAEq(r),ie.textContent=r.toFixed(2)}),N.addEventListener("input",()=>{const r=parseFloat(N.value);n.setTau(r),ae.textContent=r.toFixed(0)}),V.addEventListener("input",()=>{const r=parseFloat(V.value);n.setBeta(r),oe.textContent=r.toFixed(3)}),T.addEventListener("input",()=>{const r=parseFloat(T.value);n.setGain(r),ue.textContent=r.toFixed(0)}),z.addEventListener("change",()=>{n.setMode(parseInt(z.value))}),he.addEventListener("click",()=>n.reset()),ne.addEventListener("change",()=>{n.setScheme(ne.value)});function u(){if(!I){const r=a.createCommandEncoder({label:"frame"});n.tick(r,D),n.render(r,f.getCurrentTexture().createView()),a.queue.submit([r.finish()])}A.textContent=`${g} | 256×256 | t = ${n.simTime.toFixed(1)}`,requestAnimationFrame(u)}requestAnimationFrame(u)}Pe().catch(e=>{console.error(e),A.textContent=`Error: ${e.message}`});
