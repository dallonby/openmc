//! \file geometry.h
//! Device port of OpenMC's CSG tracking engine (cell.cpp, surface.cpp,
//! lattice.cpp, geometry.cpp @ develop 86ceaad3c). Algorithm structure and
//! tie-break semantics mirror the CPU implementation; tolerances are the
//! fp32 equivalents defined in types.h. Differences from CPU (documented in
//! docs/metal_port.md): fp32 arithmetic, bounded virtual-crossing loop,
//! no torus surfaces, no hex lattices, no neighbor lists (level-local
//! re-search on crossing instead).

#pragma once

#ifdef GPU_HOST_DEBUG
struct GpuGeomState;
void gpu_host_debug_escape(THREAD const GpuGeomState* gs);
#endif

// ---------------------------------------------------------------- vectors --
struct GpuVec3 {
  float x, y, z;
};

DEVICE_FN GpuVec3 gpu_v3(float x, float y, float z)
{
  GpuVec3 v;
  v.x = x;
  v.y = y;
  v.z = z;
  return v;
}
DEVICE_FN float gpu_dot(GpuVec3 a, GpuVec3 b)
{
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
DEVICE_FN GpuVec3 gpu_add(GpuVec3 a, GpuVec3 b)
{
  return gpu_v3(a.x + b.x, a.y + b.y, a.z + b.z);
}
DEVICE_FN GpuVec3 gpu_sub(GpuVec3 a, GpuVec3 b)
{
  return gpu_v3(a.x - b.x, a.y - b.y, a.z - b.z);
}
DEVICE_FN GpuVec3 gpu_scale(GpuVec3 a, float s)
{
  return gpu_v3(a.x * s, a.y * s, a.z * s);
}
DEVICE_FN float gpu_norm(GpuVec3 a)
{
  return sqrtf(gpu_dot(a, a));
}
// Row-major 3x3 rotation (matches Position::rotate)
DEVICE_FN GpuVec3 gpu_rotate(GpuVec3 v, GLOBAL const float* R)
{
  return gpu_v3(v.x * R[0] + v.y * R[1] + v.z * R[2],
    v.x * R[3] + v.y * R[4] + v.z * R[5], v.x * R[6] + v.y * R[7] + v.z * R[8]);
}
// u - 2(n.u)/(n.n) n  (matches Position::reflect)
DEVICE_FN GpuVec3 gpu_reflect_dir(GpuVec3 u, GpuVec3 n)
{
  float s = 2.0f * gpu_dot(u, n) / gpu_dot(n, n);
  return gpu_sub(u, gpu_scale(n, s));
}

// -------------------------------------------------------------- data view --
struct GpuGeomData {
  GLOBAL const GpuSurface* surfaces;
  GLOBAL const GpuCell* cells;
  GLOBAL const GpuUniverse* universes;
  GLOBAL const GpuLattice* lattices;
  GLOBAL const GpuMaterial* materials;
  GLOBAL const int32_gpu* i32;
  GLOBAL const float* f32;
};

// ---------------------------------------------------------- surface math --

DEVICE_FN float gpu_surf_evaluate(GpuGeomData g, int32_gpu i_surf, GpuVec3 r)
{
  GpuSurface s = g.surfaces[i_surf];
  GLOBAL const float* c = g.f32 + s.coeff_off;
  switch (s.type) {
  case GPU_SURF_X_PLANE:
    return r.x - c[0];
  case GPU_SURF_Y_PLANE:
    return r.y - c[0];
  case GPU_SURF_Z_PLANE:
    return r.z - c[0];
  case GPU_SURF_PLANE:
    return c[0] * r.x + c[1] * r.y + c[2] * r.z - c[3];
  case GPU_SURF_X_CYLINDER: {
    float y = r.y - c[0], z = r.z - c[1];
    return y * y + z * z - c[2] * c[2];
  }
  case GPU_SURF_Y_CYLINDER: {
    float x = r.x - c[0], z = r.z - c[1];
    return x * x + z * z - c[2] * c[2];
  }
  case GPU_SURF_Z_CYLINDER: {
    float x = r.x - c[0], y = r.y - c[1];
    return x * x + y * y - c[2] * c[2];
  }
  case GPU_SURF_SPHERE: {
    float x = r.x - c[0], y = r.y - c[1], z = r.z - c[2];
    return x * x + y * y + z * z - c[3] * c[3];
  }
  case GPU_SURF_X_CONE: {
    float x = r.x - c[0], y = r.y - c[1], z = r.z - c[2];
    return y * y + z * z - c[3] * x * x;
  }
  case GPU_SURF_Y_CONE: {
    float x = r.x - c[0], y = r.y - c[1], z = r.z - c[2];
    return x * x + z * z - c[3] * y * y;
  }
  case GPU_SURF_Z_CONE: {
    float x = r.x - c[0], y = r.y - c[1], z = r.z - c[2];
    return x * x + y * y - c[3] * z * z;
  }
  case GPU_SURF_QUADRIC: {
    float x = r.x, y = r.y, z = r.z;
    return x * (c[0] * x + c[3] * y + c[6]) + y * (c[1] * y + c[4] * z + c[7]) +
           z * (c[2] * z + c[5] * x + c[8]) + c[9];
  }
  }
  return GPU_INFTY;
}

DEVICE_FN GpuVec3 gpu_surf_normal(GpuGeomData g, int32_gpu i_surf, GpuVec3 r)
{
  GpuSurface s = g.surfaces[i_surf];
  GLOBAL const float* c = g.f32 + s.coeff_off;
  switch (s.type) {
  case GPU_SURF_X_PLANE:
    return gpu_v3(1.0f, 0.0f, 0.0f);
  case GPU_SURF_Y_PLANE:
    return gpu_v3(0.0f, 1.0f, 0.0f);
  case GPU_SURF_Z_PLANE:
    return gpu_v3(0.0f, 0.0f, 1.0f);
  case GPU_SURF_PLANE:
    return gpu_v3(c[0], c[1], c[2]);
  case GPU_SURF_X_CYLINDER:
    return gpu_v3(0.0f, 2.0f * (r.y - c[0]), 2.0f * (r.z - c[1]));
  case GPU_SURF_Y_CYLINDER:
    return gpu_v3(2.0f * (r.x - c[0]), 0.0f, 2.0f * (r.z - c[1]));
  case GPU_SURF_Z_CYLINDER:
    return gpu_v3(2.0f * (r.x - c[0]), 2.0f * (r.y - c[1]), 0.0f);
  case GPU_SURF_SPHERE:
    return gpu_v3(
      2.0f * (r.x - c[0]), 2.0f * (r.y - c[1]), 2.0f * (r.z - c[2]));
  case GPU_SURF_X_CONE:
    return gpu_v3(
      -2.0f * c[3] * (r.x - c[0]), 2.0f * (r.y - c[1]), 2.0f * (r.z - c[2]));
  case GPU_SURF_Y_CONE:
    return gpu_v3(
      2.0f * (r.x - c[0]), -2.0f * c[3] * (r.y - c[1]), 2.0f * (r.z - c[2]));
  case GPU_SURF_Z_CONE:
    return gpu_v3(
      2.0f * (r.x - c[0]), 2.0f * (r.y - c[1]), -2.0f * c[3] * (r.z - c[2]));
  case GPU_SURF_QUADRIC:
    return gpu_v3(2.0f * c[0] * r.x + c[3] * r.y + c[5] * r.z + c[6],
      2.0f * c[1] * r.y + c[3] * r.x + c[4] * r.z + c[7],
      2.0f * c[2] * r.z + c[4] * r.y + c[5] * r.x + c[8]);
  }
  return gpu_v3(0.0f, 0.0f, 1.0f);
}

//! Distance along u to the surface; GPU_INFTY if no positive hit.
//! on_side: 0 when not on this surface, else +1/-1 = the sense side the
//! particle is currently on (the sign of its on-surface token).
//! side_hint: the sense side this CELL's region expects for the surface
//! (the region token's sign) — used only to disambiguate f == 0 exactly,
//! where -0/ui = +0 would otherwise fabricate a zero-distance crossing in
//! BOTH directions (fp32 corner points quantize onto planes; the resulting
//! spurious re-reflections ping-pong until the event cap).
DEVICE_FN float gpu_surf_distance(GpuGeomData g, int32_gpu i_surf, GpuVec3 r,
  GpuVec3 u, int32_gpu on_side, int32_gpu side_hint)
{
  GpuSurface s = g.surfaces[i_surf];
  GLOBAL const float* c = g.f32 + s.coeff_off;
  bool coincident = (on_side != 0);

  switch (s.type) {
  case GPU_SURF_X_PLANE:
  case GPU_SURF_Y_PLANE:
  case GPU_SURF_Z_PLANE: {
    float f, ui;
    if (s.type == GPU_SURF_X_PLANE) {
      f = r.x - c[0];
      ui = u.x;
    } else if (s.type == GPU_SURF_Y_PLANE) {
      f = r.y - c[0];
      ui = u.y;
    } else {
      f = r.z - c[0];
      ui = u.z;
    }
    // No coincidence band for planes: the on-surface token plus the sign
    // of d handle every case. The token's suppression is DIRECTIONAL: an
    // on-plane particle moving away never re-crosses (INFTY), but one
    // moving back toward the plane must produce a d = 0 crossing so
    // boundary conditions fire — otherwise an on-wall particle that
    // scatters outward streams straight through the wall. The token flip
    // at the crossing prevents re-crossing loops.
    if (ui == 0.0f)
      return GPU_INFTY;
    int32_gpu side = on_side;
    if (side == 0 && f == 0.0f)
      side = side_hint;
    if (side != 0)
      return ((side > 0) ? (ui < 0.0f) : (ui > 0.0f)) ? 0.0f : GPU_INFTY;
    float d = -f / ui;
    return (d < 0.0f) ? GPU_INFTY : d;
  }
  case GPU_SURF_PLANE: {
    float f = c[0] * r.x + c[1] * r.y + c[2] * r.z - c[3];
    float proj = c[0] * u.x + c[1] * u.y + c[2] * u.z;
    if (proj == 0.0f)
      return GPU_INFTY;
    int32_gpu side = on_side;
    if (side == 0 && f == 0.0f)
      side = side_hint;
    if (side != 0)
      return ((side > 0) ? (proj < 0.0f) : (proj > 0.0f)) ? 0.0f : GPU_INFTY;
    float d = -f / proj;
    return (d < 0.0f) ? GPU_INFTY : d;
  }
  case GPU_SURF_X_CYLINDER:
  case GPU_SURF_Y_CYLINDER:
  case GPU_SURF_Z_CYLINDER: {
    float r1, r2, u1, u2, radius = c[2];
    if (s.type == GPU_SURF_X_CYLINDER) {
      r1 = r.y - c[0];
      r2 = r.z - c[1];
      u1 = u.y;
      u2 = u.z;
    } else if (s.type == GPU_SURF_Y_CYLINDER) {
      r1 = r.x - c[0];
      r2 = r.z - c[1];
      u1 = u.x;
      u2 = u.z;
    } else {
      r1 = r.x - c[0];
      r2 = r.y - c[1];
      u1 = u.x;
      u2 = u.y;
    }
    float a = u1 * u1 + u2 * u2;
    if (a == 0.0f)
      return GPU_INFTY;
    float k = r1 * u1 + r2 * u2;
    float cc = r1 * r1 + r2 * r2 - radius * radius;
    float quad = k * k - a * cc;
    if (quad < 0.0f)
      return GPU_INFTY;
    if (coincident || fabsf(cc) < GPU_FP_COINCIDENT) {
      // on surface: if moving inward, hit the far side
      return (k >= 0.0f) ? GPU_INFTY : (-k + sqrtf(quad)) / a;
    } else if (cc < 0.0f) {
      return (-k + sqrtf(quad)) / a; // inside
    } else {
      float d = (-k - sqrtf(quad)) / a; // outside
      return (d < 0.0f) ? GPU_INFTY : d;
    }
  }
  case GPU_SURF_SPHERE: {
    float x = r.x - c[0], y = r.y - c[1], z = r.z - c[2];
    float k = x * u.x + y * u.y + z * u.z;
    float cc = x * x + y * y + z * z - c[3] * c[3];
    float quad = k * k - cc;
    if (quad < 0.0f)
      return GPU_INFTY;
    if (coincident || fabsf(cc) < GPU_FP_COINCIDENT) {
      return (k >= 0.0f) ? GPU_INFTY : (-k + sqrtf(quad));
    } else if (cc < 0.0f) {
      return -k + sqrtf(quad);
    } else {
      float d = -k - sqrtf(quad);
      return (d < 0.0f) ? GPU_INFTY : d;
    }
  }
  case GPU_SURF_X_CONE:
  case GPU_SURF_Y_CONE:
  case GPU_SURF_Z_CONE: {
    float x = r.x - c[0], y = r.y - c[1], z = r.z - c[2];
    float rsq = c[3];
    float a, k, cc;
    if (s.type == GPU_SURF_X_CONE) {
      a = u.y * u.y + u.z * u.z - rsq * u.x * u.x;
      k = y * u.y + z * u.z - rsq * x * u.x;
      cc = y * y + z * z - rsq * x * x;
    } else if (s.type == GPU_SURF_Y_CONE) {
      a = u.x * u.x + u.z * u.z - rsq * u.y * u.y;
      k = x * u.x + z * u.z - rsq * y * u.y;
      cc = x * x + z * z - rsq * y * y;
    } else {
      a = u.x * u.x + u.y * u.y - rsq * u.z * u.z;
      k = x * u.x + y * u.y - rsq * z * u.z;
      cc = x * x + y * y - rsq * z * z;
    }
    float d;
    if (a == 0.0f) {
      if (k == 0.0f)
        return GPU_INFTY;
      d = -0.5f * cc / k;
    } else {
      float quad = k * k - a * cc;
      if (quad < 0.0f)
        return GPU_INFTY;
      float sq = sqrtf(quad);
      if (coincident || fabsf(cc) < GPU_FP_COINCIDENT) {
        d = (k >= 0.0f) ? (-k - sq) / a : (-k + sq) / a;
      } else {
        float b = (-k - sq) / a;
        float e = (-k + sq) / a;
        // nearest positive root
        if (b < 0.0f)
          b = GPU_INFTY;
        if (e < 0.0f)
          e = GPU_INFTY;
        d = fminf(b, e);
      }
    }
    return (d <= 0.0f || d == GPU_INFTY) ? GPU_INFTY : d;
  }
  case GPU_SURF_QUADRIC: {
    float x = r.x, y = r.y, z = r.z;
    float A = c[0], B = c[1], C = c[2], D = c[3], E = c[4], F = c[5], G = c[6],
          H = c[7], J = c[8];
    float a = A * u.x * u.x + B * u.y * u.y + C * u.z * u.z + D * u.x * u.y +
              E * u.y * u.z + F * u.x * u.z;
    float k = (A * u.x * x + B * u.y * y + C * u.z * z +
               0.5f * (D * (u.x * y + u.y * x) + E * (u.y * z + u.z * y) +
                        F * (u.x * z + u.z * x) + G * u.x + H * u.y + J * u.z));
    float cc = gpu_surf_evaluate(g, i_surf, r);
    float d;
    if (a == 0.0f) {
      if (k == 0.0f)
        return GPU_INFTY;
      d = -0.5f * cc / k;
      if (coincident || fabsf(cc) < GPU_FP_COINCIDENT)
        return GPU_INFTY;
    } else {
      float quad = k * k - a * cc;
      if (quad < 0.0f)
        return GPU_INFTY;
      float sq = sqrtf(quad);
      if (coincident || fabsf(cc) < GPU_FP_COINCIDENT) {
        d = (k >= 0.0f) ? (-k - sq) / a : (-k + sq) / a;
      } else {
        float b = (-k - sq) / a;
        float e = (-k + sq) / a;
        if (b < 0.0f)
          b = GPU_INFTY;
        if (e < 0.0f)
          e = GPU_INFTY;
        d = fminf(b, e);
      }
    }
    return (d <= 0.0f || d == GPU_INFTY) ? GPU_INFTY : d;
  }
  }
  return GPU_INFTY;
}

//! Halfspace sense with directional tie-break (Surface::sense)
DEVICE_FN bool gpu_surf_sense(
  GpuGeomData g, int32_gpu i_surf, GpuVec3 r, GpuVec3 u)
{
  float f = gpu_surf_evaluate(g, i_surf, r);
  if (fabsf(f) < GPU_FP_COINCIDENT) {
    return gpu_dot(u, gpu_surf_normal(g, i_surf, r)) > 0.0f;
  }
  return f > 0.0f;
}

// ------------------------------------------------------- region evaluation --

//! Region::contains for simple (intersection-only) cells
DEVICE_FN bool gpu_contains_simple(GpuGeomData g, GLOBAL const int32_gpu* tok,
  uint32_gpu n, GpuVec3 r, GpuVec3 u, int32_gpu on_surface)
{
  for (uint32_gpu i = 0; i < n; ++i) {
    int32_gpu token = tok[i];
    if (token == on_surface) {
      // known to satisfy this halfspace
    } else if (-token == on_surface) {
      return false;
    } else {
      bool sense = gpu_surf_sense(g, (token > 0 ? token : -token) - 1, r, u);
      if (sense != (token > 0))
        return false;
    }
  }
  return true;
}

//! Region::contains_complex: stackless single pass over infix expression
//! with short-circuit skip over balanced parenthesis blocks.
DEVICE_FN bool gpu_contains_complex(GpuGeomData g, GLOBAL const int32_gpu* tok,
  uint32_gpu n, GpuVec3 r, GpuVec3 u, int32_gpu on_surface)
{
  bool in_cell = true;
  int32_gpu total_depth = 0;
  uint32_gpu i = 0;
  while (i < n) {
    int32_gpu token = tok[i];
    if (token == GPU_OP_LEFT_PAREN) {
      ++total_depth;
    } else if (token == GPU_OP_RIGHT_PAREN) {
      --total_depth;
    } else if (token == GPU_OP_UNION || token == GPU_OP_INTERSECTION) {
      bool skip = (token == GPU_OP_UNION) ? in_cell : !in_cell;
      if (skip) {
        if (total_depth == 0)
          return in_cell;
        // skip forward over the balanced remainder of this paren block
        int32_gpu depth = 1;
        do {
          ++i;
          if (i >= n)
            break;
          if (tok[i] == GPU_OP_RIGHT_PAREN)
            --depth;
          else if (tok[i] == GPU_OP_LEFT_PAREN)
            ++depth;
        } while (depth > 0);
        --total_depth;
      }
    } else {
      // surface token
      if (token == on_surface) {
        in_cell = true;
      } else if (-token == on_surface) {
        in_cell = false;
      } else {
        bool sense = gpu_surf_sense(g, (token > 0 ? token : -token) - 1, r, u);
        in_cell = (sense == (token > 0));
      }
    }
    ++i;
  }
  return in_cell;
}

DEVICE_FN bool gpu_cell_contains(
  GpuGeomData g, int32_gpu i_cell, GpuVec3 r, GpuVec3 u, int32_gpu on_surface)
{
  GpuCell c = g.cells[i_cell];
  if (c.n_tokens == 0)
    return true;
  GLOBAL const int32_gpu* tok = g.i32 + c.token_off;
  return c.simple ? gpu_contains_simple(g, tok, c.n_tokens, r, u, on_surface)
                  : gpu_contains_complex(g, tok, c.n_tokens, r, u, on_surface);
}

// --------------------------------------------------------- cell distances --

struct GpuCellDist {
  float d;
  int32_gpu surf; // signed token of surface to be crossed (entered halfspace)
};

//! Region::distance_to_nearest_surface — first-wins relative tie-break,
//! iteration order over tokens is semantically significant.
DEVICE_FN GpuCellDist gpu_cell_distance_nearest(GpuGeomData g,
  GLOBAL const int32_gpu* tok, uint32_gpu n, GpuVec3 r, GpuVec3 u,
  int32_gpu on_surface, bool ignore_coincident)
{
  GpuCellDist out;
  out.d = GPU_INFTY;
  out.surf = GPU_SURFACE_NONE;
  int32_gpu on_abs = on_surface > 0 ? on_surface : -on_surface;
  for (uint32_gpu i = 0; i < n; ++i) {
    int32_gpu token = tok[i];
    if (token >= GPU_OP_UNION)
      continue;
    int32_gpu abs_tok = token > 0 ? token : -token;
    int32_gpu on_side =
      (abs_tok == on_abs) ? ((on_surface > 0) ? 1 : -1) : 0;
    float d = gpu_surf_distance(
      g, abs_tok - 1, r, u, on_side, (token > 0) ? 1 : -1);
    if (ignore_coincident && d < GPU_FP_COINCIDENT)
      continue;
    if (d < out.d) {
      if (out.d - d >= GPU_FP_PRECISION * out.d) {
        out.d = d;
        out.surf = -token;
      }
    }
  }
  return out;
}

//! CSGCell::distance dispatch, with the complex-region virtual-crossing
//! loop bounded by the geometric maximum (2 crossings per quadric token).
DEVICE_FN GpuCellDist gpu_cell_distance(
  GpuGeomData g, int32_gpu i_cell, GpuVec3 r, GpuVec3 u, int32_gpu on_surface)
{
  GpuCell c = g.cells[i_cell];
  GpuCellDist none;
  none.d = GPU_INFTY;
  none.surf = GPU_SURFACE_NONE;
  if (c.n_tokens == 0)
    return none;
  GLOBAL const int32_gpu* tok = g.i32 + c.token_off;
  if (c.simple)
    return gpu_cell_distance_nearest(
      g, tok, c.n_tokens, r, u, on_surface, false);

  // complex region: advance past virtual crossings until region membership
  // actually changes. Coincident hits are skipped only while actually on a
  // surface (CPU: ignore_coincident_surfaces = on_surface != 0) — dropping
  // them unconditionally would skip a genuinely-near first boundary.
  // A quadric crosses a straight ray at most twice, so 2*n_tokens bounds
  // the real crossings: exhausting the loop provably means no boundary
  // (the CPU loop is unbounded but terminates for the same reason).
  bool in_region = gpu_contains_complex(g, tok, c.n_tokens, r, u, on_surface);
  float d_total = 0.0f;
  GpuVec3 rr = r;
  int32_gpu on = on_surface;
  int32_gpu max_iter = 2 * (int32_gpu)c.n_tokens + 4;
  for (int32_gpu iter = 0; iter < max_iter; ++iter) {
    GpuCellDist cand =
      gpu_cell_distance_nearest(g, tok, c.n_tokens, rr, u, on, on != 0);
    if (cand.d == GPU_INFTY)
      return none;
    d_total += cand.d;
    rr = gpu_add(rr, gpu_scale(u, cand.d));
    int32_gpu abs_s = cand.surf > 0 ? cand.surf : -cand.surf;
    // re-sign from the actual crossing direction
    GpuVec3 nrm = gpu_surf_normal(g, abs_s - 1, rr);
    int32_gpu signed_surf = (gpu_dot(u, nrm) > 0.0f) ? abs_s : -abs_s;
    if (gpu_contains_complex(g, tok, c.n_tokens, rr, u, signed_surf) !=
        in_region) {
      GpuCellDist out;
      out.d = d_total;
      out.surf = signed_surf;
      return out;
    }
    on = signed_surf;
  }
  return none; // give up: treated as no boundary in this cell
}

// ---------------------------------------------------------------- lattice --

DEVICE_FN bool gpu_lat_valid(GpuLattice lat, THREAD const int32_gpu* i)
{
  return i[0] >= 0 && i[0] < lat.nx && i[1] >= 0 && i[1] < lat.ny &&
         i[2] >= 0 && i[2] < lat.nz;
}

DEVICE_FN int32_gpu gpu_lat_flat(GpuLattice lat, THREAD const int32_gpu* i)
{
  return lat.nx * lat.ny * i[2] + lat.nx * i[1] + i[0];
}

DEVICE_FN bool gpu_isclose(float a, float b, float rel, float abs_tol)
{
  float m = fmaxf(fabsf(a), fabsf(b));
  return fabsf(a - b) <= fmaxf(rel * m, abs_tol);
}

//! RectLattice::get_indices with direction-aware coincidence handling
DEVICE_FN void gpu_lat_get_indices(
  GpuLattice lat, GpuVec3 r, GpuVec3 u, THREAD int32_gpu* out)
{
  float ix = (r.x - lat.llx) / lat.px;
  float iy = (r.y - lat.lly) / lat.py;
  float iz = lat.is_3d ? (r.z - lat.llz) / lat.pz : 0.5f;
  float cx = (float)lroundf(ix);
  float cy = (float)lroundf(iy);
  float cz = (float)lroundf(iz);
  if (fabsf(ix - cx) < GPU_LAT_COINCIDENT)
    out[0] = (u.x > 0.0f) ? (int32_gpu)cx : (int32_gpu)cx - 1;
  else
    out[0] = (int32_gpu)floorf(ix);
  if (fabsf(iy - cy) < GPU_LAT_COINCIDENT)
    out[1] = (u.y > 0.0f) ? (int32_gpu)cy : (int32_gpu)cy - 1;
  else
    out[1] = (int32_gpu)floorf(iy);
  if (!lat.is_3d)
    out[2] = 0;
  else if (fabsf(iz - cz) < GPU_LAT_COINCIDENT)
    out[2] = (u.z > 0.0f) ? (int32_gpu)cz : (int32_gpu)cz - 1;
  else
    out[2] = (int32_gpu)floorf(iz);
}

//! RectLattice::get_local_position (tile-centered coordinates)
DEVICE_FN GpuVec3 gpu_lat_local(
  GpuLattice lat, GpuVec3 r, THREAD const int32_gpu* i)
{
  r.x -= lat.llx + ((float)i[0] + 0.5f) * lat.px;
  r.y -= lat.lly + ((float)i[1] + 0.5f) * lat.py;
  if (lat.is_3d)
    r.z -= lat.llz + ((float)i[2] + 0.5f) * lat.pz;
  return r;
}

struct GpuLatDist {
  float d;
  int32_gpu trans[3];
};

//! RectLattice::distance — r is tile-local (centered)
DEVICE_FN GpuLatDist gpu_lat_distance(GpuLattice lat, GpuVec3 r, GpuVec3 u)
{
  float x0 = copysignf(0.5f * lat.px, u.x);
  float y0 = copysignf(0.5f * lat.py, u.y);
  float dx = (u.x == 0.0f) ? GPU_INFTY : (x0 - r.x) / u.x;
  float dy = (u.y == 0.0f) ? GPU_INFTY : (y0 - r.y) / u.y;
  float dz = GPU_INFTY;
  if (lat.is_3d) {
    float z0 = copysignf(0.5f * lat.pz, u.z);
    dz = (u.z == 0.0f) ? GPU_INFTY : (z0 - r.z) / u.z;
  }
  float d = fminf(dx, fminf(dy, dz));
  GpuLatDist out;
  out.d = d;
  out.trans[0] = 0;
  out.trans[1] = 0;
  out.trans[2] = 0;
  if (gpu_isclose(d, dx, GPU_LAT_TIE_REL, GPU_LAT_TIE_ABS))
    out.trans[0] = (u.x > 0.0f) ? 1 : -1;
  if (gpu_isclose(d, dy, GPU_LAT_TIE_REL, GPU_LAT_TIE_ABS))
    out.trans[1] = (u.y > 0.0f) ? 1 : -1;
  if (lat.is_3d && gpu_isclose(d, dz, GPU_LAT_TIE_REL, GPU_LAT_TIE_ABS))
    out.trans[2] = (u.z > 0.0f) ? 1 : -1;
  return out;
}

// ---------------------------------------------------------- particle geom --

struct GpuCoord {
  GpuVec3 r;
  GpuVec3 u;
  int32_gpu cell;
  int32_gpu universe;
  int32_gpu lattice;
  int32_gpu li[3];
  uint32_gpu rotated;
};

struct GpuBoundary {
  float d;
  int32_gpu surface;     // signed token; sign = halfspace being entered
  int32_gpu coord_level; // 1-based
  int32_gpu lat_trans[3];
};

struct GpuGeomState {
  GpuCoord coord[GPU_MAX_COORD];
  int32_gpu n_coord;
  int32_gpu surface; // signed on-surface token
  int32_gpu material;
  float sqrtkT;
  float density_mult;
  GpuBoundary boundary;
  uint32_gpu lost;
};

DEVICE_FN void gpu_coord_reset(THREAD GpuCoord* c)
{
  c->cell = GPU_C_NONE;
  c->universe = GPU_C_NONE;
  c->lattice = GPU_C_NONE;
  c->li[0] = 0;
  c->li[1] = 0;
  c->li[2] = 0;
  c->rotated = 0;
}

//! Universe::find_cell — first-match linear scan over the universe cell list
DEVICE_FN int32_gpu gpu_universe_find_cell(
  GpuGeomData g, int32_gpu i_univ, GpuVec3 r, GpuVec3 u, int32_gpu on_surface)
{
  GpuUniverse uni = g.universes[i_univ];
  for (uint32_gpu k = 0; k < uni.n_cells; ++k) {
    int32_gpu ic = g.i32[uni.cells_off + k];
    if (gpu_cell_contains(g, ic, r, u, on_surface))
      return ic;
  }
  return GPU_C_NONE;
}

//! find_cell_inner: descend from level p->n_coord-1 until a material cell.
//! Assumes coord[n_coord-1].universe and r/u are set. Returns false if lost.
DEVICE_FN bool gpu_find_cell_inner(
  GpuGeomData g, THREAD GpuGeomState* p, int32_gpu n_coord_levels)
{
  for (;; ++p->n_coord) {
    if (p->n_coord > n_coord_levels)
      return false;
    int32_gpu lev = p->n_coord - 1;
    THREAD GpuCoord* c = &p->coord[lev];
    if (c->cell == GPU_C_NONE) {
      int32_gpu ic =
        gpu_universe_find_cell(g, c->universe, c->r, c->u, p->surface);
      if (ic == GPU_C_NONE)
        return false;
      c->cell = ic;
    }
    GpuCell cell = g.cells[c->cell];
    if (cell.fill_type == GPU_FILL_MATERIAL) {
      p->material = cell.material;
      p->sqrtkT = cell.sqrtkT;
      p->density_mult = cell.density_mult;
      return true;
    } else if (cell.fill_type == GPU_FILL_UNIVERSE) {
      if (lev + 1 >= GPU_MAX_COORD)
        return false;
      THREAD GpuCoord* nc = &p->coord[lev + 1];
      gpu_coord_reset(nc);
      nc->r = c->r;
      nc->u = c->u;
      if (cell.trans_off >= 0) {
        nc->r.x -= g.f32[cell.trans_off];
        nc->r.y -= g.f32[cell.trans_off + 1];
        nc->r.z -= g.f32[cell.trans_off + 2];
      }
      if (cell.rot_off >= 0) {
        nc->r = gpu_rotate(nc->r, g.f32 + cell.rot_off);
        nc->u = gpu_rotate(nc->u, g.f32 + cell.rot_off);
        nc->rotated = 1;
      }
      nc->universe = cell.fill;
    } else { // lattice fill
      if (lev + 1 >= GPU_MAX_COORD)
        return false;
      THREAD GpuCoord* nc = &p->coord[lev + 1];
      gpu_coord_reset(nc);
      GpuVec3 rr = c->r;
      GpuVec3 uu = c->u;
      uint32_gpu rot = 0;
      if (cell.trans_off >= 0) {
        rr.x -= g.f32[cell.trans_off];
        rr.y -= g.f32[cell.trans_off + 1];
        rr.z -= g.f32[cell.trans_off + 2];
      }
      if (cell.rot_off >= 0) {
        rr = gpu_rotate(rr, g.f32 + cell.rot_off);
        uu = gpu_rotate(uu, g.f32 + cell.rot_off);
        rot = 1;
      }
      GpuLattice lat = g.lattices[cell.fill];
      int32_gpu li[3];
      gpu_lat_get_indices(lat, rr, uu, li);
      nc->r = gpu_lat_local(lat, rr, li);
      nc->u = uu;
      nc->rotated = rot;
      nc->lattice = cell.fill;
      nc->li[0] = li[0];
      nc->li[1] = li[1];
      nc->li[2] = li[2];
      int32_gpu iu;
      if (gpu_lat_valid(lat, li))
        iu = g.i32[lat.univ_off + gpu_lat_flat(lat, li)];
      else if (lat.outer != GPU_C_NONE)
        iu = lat.outer;
      else
        return false;
      nc->universe = iu;
    }
    p->coord[p->n_coord].cell = GPU_C_NONE;
  }
}

//! exhaustive_find_cell: restart from root universe
DEVICE_FN bool gpu_exhaustive_find_cell(
  GpuGeomData g, THREAD GpuGeomState* p, int32_gpu root, int32_gpu levels)
{
  GpuVec3 r = p->coord[0].r;
  GpuVec3 u = p->coord[0].u;
  for (int i = 0; i < GPU_MAX_COORD; ++i)
    gpu_coord_reset(&p->coord[i]);
  p->coord[0].r = r;
  p->coord[0].u = u;
  p->coord[0].universe = root;
  p->n_coord = 1;
  return gpu_find_cell_inner(g, p, levels);
}

//! Level-local re-search after a surface crossing: keep levels < n_coord,
//! clear deeper ones, search the universe at the current lowest level.
DEVICE_FN bool gpu_local_find_cell(
  GpuGeomData g, THREAD GpuGeomState* p, int32_gpu levels)
{
  for (int i = p->n_coord; i < GPU_MAX_COORD; ++i)
    gpu_coord_reset(&p->coord[i]);
  p->coord[p->n_coord - 1].cell = GPU_C_NONE;
  return gpu_find_cell_inner(g, p, levels);
}

//! distance_to_boundary: min over all coordinate levels of cell-surface and
//! lattice-tile distances, preferring higher levels within relative
//! precision (mirrors geometry.cpp:419).
DEVICE_FN GpuBoundary gpu_distance_to_boundary(
  GpuGeomData g, THREAD GpuGeomState* p)
{
  GpuBoundary info;
  info.d = GPU_INFTY;
  info.surface = GPU_SURFACE_NONE;
  info.coord_level = 0;
  info.lat_trans[0] = 0;
  info.lat_trans[1] = 0;
  info.lat_trans[2] = 0;
  float d = GPU_INFTY;

  for (int32_gpu i = 0; i < p->n_coord; ++i) {
    THREAD GpuCoord* c = &p->coord[i];
    GpuCellDist sd = gpu_cell_distance(g, c->cell, c->r, c->u, p->surface);

    float d_lat = GPU_INFTY;
    GpuLatDist ld;
    ld.trans[0] = 0;
    ld.trans[1] = 0;
    ld.trans[2] = 0;
    if (c->lattice != GPU_C_NONE) {
      GpuLattice lat = g.lattices[c->lattice];
      ld = gpu_lat_distance(lat, c->r, c->u);
      d_lat = ld.d;
      // fp32 flight overshoot can land the tile-local point epsilon past a
      // face while the indices still name the old tile: the crossing has
      // already happened physically, so take it now at d = 0 (the trans
      // for the overshot face is already set) and let gpu_cross_lattice
      // re-index — previously this state was marked lost
      if (d_lat < 0.0f)
        d_lat = 0.0f;
    }

    if (sd.d < d_lat - GPU_FP_COINCIDENT) {
      if (d == GPU_INFTY || ((d - sd.d) / d >= GPU_FP_REL_PRECISION &&
                              d - sd.d >= GPU_FP_ABS_TIEBREAK)) {
        d = sd.d;
        info.d = sd.d;
        info.surface = sd.surf;
        info.coord_level = i + 1;
        info.lat_trans[0] = 0;
        info.lat_trans[1] = 0;
        info.lat_trans[2] = 0;
      }
    } else if (d_lat < GPU_INFTY) {
      if (d == GPU_INFTY || ((d - d_lat) / d >= GPU_FP_REL_PRECISION &&
                              d - d_lat >= GPU_FP_ABS_TIEBREAK)) {
        d = d_lat;
        info.d = d_lat;
        info.surface = GPU_SURFACE_NONE;
        info.coord_level = i + 1;
        info.lat_trans[0] = ld.trans[0];
        info.lat_trans[1] = ld.trans[1];
        info.lat_trans[2] = ld.trans[2];
      }
    }
  }
  return info;
}

//! move all coordinate levels forward by `length`
DEVICE_FN void gpu_move_distance(THREAD GpuGeomState* p, float length)
{
  for (int32_gpu j = 0; j < p->n_coord; ++j) {
    p->coord[j].r = gpu_add(p->coord[j].r, gpu_scale(p->coord[j].u, length));
  }
}

//! cross_lattice: apply the tile translation at the boundary coord level and
//! recompute local position from the parent frame (mirrors geometry.cpp:359)
DEVICE_FN bool gpu_cross_lattice(
  GpuGeomData g, THREAD GpuGeomState* p, int32_gpu root, int32_gpu levels)
{
  THREAD GpuCoord* c = &p->coord[p->n_coord - 1];
  GpuLattice lat = g.lattices[c->lattice];
  c->li[0] += p->boundary.lat_trans[0];
  c->li[1] += p->boundary.lat_trans[1];
  c->li[2] += p->boundary.lat_trans[2];

  // rebuild local position from parent level
  THREAD GpuCoord* parent = &p->coord[p->n_coord - 2];
  GpuVec3 rr = parent->r;
  GpuVec3 uu = parent->u;
  GpuCell pcell = g.cells[parent->cell];
  if (pcell.trans_off >= 0) {
    rr.x -= g.f32[pcell.trans_off];
    rr.y -= g.f32[pcell.trans_off + 1];
    rr.z -= g.f32[pcell.trans_off + 2];
  }
  if (pcell.rot_off >= 0) {
    rr = gpu_rotate(rr, g.f32 + pcell.rot_off);
    uu = gpu_rotate(uu, g.f32 + pcell.rot_off);
  }
  c->r = gpu_lat_local(lat, rr, c->li);
  c->u = uu;

  bool ok;
  if (!gpu_lat_valid(lat, c->li)) {
    // The particle left the lattice. When the lattice has an outer
    // universe, continue in it at the current level: the extrapolated
    // tile frame set above is exactly the frame the normal lattice-fill
    // descent would use for out-of-range indices, and it avoids
    // re-deciding an on-boundary fp32 point from the root (which flips a
    // sign coin on the crossed face and loses the particle — measured
    // ~1e-4 of MG-lattice histories). Without an outer universe, fall
    // back to the CPU's base-coordinate re-search (geometry.cpp
    // cross_lattice), with a nudge retry for the on-surface point.
    if (lat.outer != GPU_C_NONE) {
      c->universe = lat.outer;
      c->cell = GPU_C_NONE;
      if (gpu_find_cell_inner(g, p, levels))
        ok = true;
      else
        ok = gpu_exhaustive_find_cell(g, p, root, levels);
    } else {
      ok = gpu_exhaustive_find_cell(g, p, root, levels);
    }
  } else {
    c->universe = g.i32[lat.univ_off + gpu_lat_flat(lat, c->li)];
    c->cell = GPU_C_NONE;
    if (gpu_find_cell_inner(g, p, levels))
      ok = true;
    else
      // corner crossing rescue: full re-search from root
      ok = gpu_exhaustive_find_cell(g, p, root, levels);
  }
#ifdef GPU_HOST_DEBUG
  if (ok && !gpu_cell_contains(
              g, p->coord[0].cell, p->coord[0].r, p->coord[0].u, p->surface))
    gpu_host_debug_escape(p);
#endif
  return ok;
}
