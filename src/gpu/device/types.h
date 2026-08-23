//! \file types.h
//! POD data model shared between host flatteners and device transport
//! kernels. Every struct here must be layout-identical under host C++,
//! Metal, and CUDA: fixed-width scalars only, no pointers (32-bit offsets
//! into typed arenas instead), explicit ordering to avoid padding surprises.

#pragma once

// NOTE: dialect.h must be included before this header.

// ---------------------------------------------------------------------------
// Constants mirrored from include/openmc/{constants.h,cell.h}
// Values are load-bearing (token classification tests) — never renumber.
// ---------------------------------------------------------------------------
#define GPU_OP_LEFT_PAREN 2147483647
#define GPU_OP_RIGHT_PAREN 2147483646
#define GPU_OP_COMPLEMENT 2147483645
#define GPU_OP_INTERSECTION 2147483644
#define GPU_OP_UNION 2147483643

#define GPU_C_NONE (-1)
#define GPU_SURFACE_NONE 0
#define GPU_MATERIAL_VOID (-1)
#define GPU_INFTY 3.0e38f

// fp32 geometry tolerances. The CPU code's FP_COINCIDENT (1e-12),
// FP_PRECISION (1e-14) and TINY_BIT (1e-8) are below fp32 resolution; these
// are the same knobs re-derived for 24-bit mantissas. Robustness relies (as
// on CPU) primarily on the on-surface token mechanism and directional sense
// tie-breaks, not on the epsilons themselves.
#define GPU_FP_COINCIDENT 1.0e-6f
#define GPU_FP_PRECISION 5.0e-7f
#define GPU_FP_REL_PRECISION 1.0e-4f
#define GPU_TINY_BIT 1.0e-5f

// Absolute significance floor for boundary selection: a lower coordinate
// level (or lattice edge) may only steal the boundary from a higher level
// when it is closer by at least this much. Protects against fp32 noise when
// a lattice edge coincides with a real surface (e.g. box wall = lattice
// outer edge) and both distances are near zero.
#define GPU_FP_ABS_TIEBREAK 1.0e-6f

// Lattice tile-index coincidence band (index space, direction-aware pick)
#define GPU_LAT_COINCIDENT 2.0e-6f

// Bounded replacement for the CPU's unbounded distance_complex loop

// Lattice corner-tie detection: two axis distances count as a simultaneous
// (corner) crossing only when they agree to fp32 rounding, i.e. a few ulps.
// The CPU uses (1e-12 rel, 1e-14 abs) — exact ties at fp64 resolution.
#define GPU_LAT_TIE_REL 5.0e-6f
#define GPU_LAT_TIE_ABS 5.0e-7f

// Maximum geometry nesting depth supported by the fixed-size coordinate
// stack in kernels. Flattening fails (falls back to CPU) if exceeded.
#ifndef GPU_MAX_COORD
#define GPU_MAX_COORD 8 // runtime MSL compile specializes to n_coord_levels+1
#endif

// ---------------------------------------------------------------------------
// Surfaces
// ---------------------------------------------------------------------------
#define GPU_SURF_X_PLANE 0
#define GPU_SURF_Y_PLANE 1
#define GPU_SURF_Z_PLANE 2
#define GPU_SURF_PLANE 3
#define GPU_SURF_X_CYLINDER 4
#define GPU_SURF_Y_CYLINDER 5
#define GPU_SURF_Z_CYLINDER 6
#define GPU_SURF_SPHERE 7
#define GPU_SURF_X_CONE 8
#define GPU_SURF_Y_CONE 9
#define GPU_SURF_Z_CONE 10
#define GPU_SURF_QUADRIC 11
// torus types are not supported in v1 — flattener rejects them

#define GPU_BC_TRANSMISSION 0
#define GPU_BC_VACUUM 1
#define GPU_BC_REFLECTIVE 2
#define GPU_BC_WHITE 3

struct GpuSurface {
  uint32_gpu type;      // GPU_SURF_*
  uint32_gpu bc;        // GPU_BC_*
  float albedo;         // > 0 → weight-splitting albedo applied at BC
  uint32_gpu coeff_off; // offset into f32 arena (up to 10 coefficients)
};

// ---------------------------------------------------------------------------
// Cells / universes / lattices
// ---------------------------------------------------------------------------
#define GPU_FILL_MATERIAL 0
#define GPU_FILL_UNIVERSE 1
#define GPU_FILL_LATTICE 2

struct GpuCell {
  int32_gpu fill_type;  // GPU_FILL_*
  int32_gpu fill;       // universe or lattice index (C_NONE for material)
  int32_gpu material;   // material index or GPU_MATERIAL_VOID (single-entry)
  int32_gpu universe;   // universe this cell belongs to
  float sqrtkT;         // sqrt(k_B T) in sqrt(eV) (single-entry)
  float density_mult;   // density multiplier
  uint32_gpu token_off; // region tokens: offset into i32 arena
  uint32_gpu n_tokens;  // token count (0 → cell fills all space)
  uint32_gpu simple;    // 1 = intersection-only token list (no operators)
  int32_gpu trans_off;  // f32 arena offset of 3-vector translation, or -1
  int32_gpu rot_off;    // f32 arena offset of row-major 3x3 rotation, or -1
};

struct GpuUniverse {
  uint32_gpu cells_off; // offset into i32 arena (list of cell indices)
  uint32_gpu n_cells;
};

struct GpuLattice { // rectangular lattices only in v1
  int32_gpu nx, ny, nz;
  float llx, lly, llz; // lower_left
  float px, py, pz;    // pitch
  uint32_gpu univ_off; // i32 arena: nx*ny*nz universe indices (x fastest)
  int32_gpu outer;     // universe index or C_NONE
  uint32_gpu is_3d;
};

// ---------------------------------------------------------------------------
// Materials
// ---------------------------------------------------------------------------
struct GpuMaterial {
  uint32_gpu nuclide_off; // i32 arena: nuclide indices (CE mode)
  uint32_gpu n_nuclides;
  uint32_gpu density_off; // f32 arena: atom densities (atom/b-cm)
  uint32_gpu fissionable; // 1 if material contains fissionable nuclides
  int32_gpu mg_off;       // MG mode: offset into mg f32 arena, or -1
  int32_gpu sab_off;      // CE mode: i32 arena offset of per-nuclide S(a,b)
                          // table index (GPU_C_NONE if none), or -1
  int32_gpu sab_frac_off; // f32 arena: per-nuclide S(a,b) fraction
};

// (multigroup per-material layout: see GpuMgMat in device/mg.h)

// ---------------------------------------------------------------------------
// Banks
// ---------------------------------------------------------------------------
struct GpuSourceSite {
  float r[3];
  float u[3];
  float E; // energy (CE) or group-as-float (MG, matches CPU convention)
  float time;
  float wgt;
  int32_gpu delayed_group;
  int32_gpu parent_id;  // 0-based current_work of parent particle
  int32_gpu progeny_id; // per-parent fission-site sequence number
  // variance-reduction state carried across a spill/re-dispatch boundary so
  // a split particle resumes exactly where its parent left off
  float wgt_born;    // source weight of the originating history
  float wgt_ww_born; // weight window the history was born in (-1 = unset)
  float ww_factor;   // Particle::ww_factor()
  int32_gpu n_split; // cumulative splits of this history
  // Unique particle id assigned at spill time (32-bit halves). Each spilled
  // secondary seeds its streams from its OWN id: seeding from the
  // re-dispatch thread id would replay a primary's stream, and copying the
  // parent's state would make every sibling an identical particle. The id
  // space is disjoint from primary ids and monotonic across generations.
  uint32_gpu uid_lo, uid_hi;
};

// ---------------------------------------------------------------------------
// Tallies
// ---------------------------------------------------------------------------
#define GPU_FILTER_CELL 0
#define GPU_FILTER_MATERIAL 1
#define GPU_FILTER_ENERGY 2
#define GPU_FILTER_UNIVERSE 3
#define GPU_FILTER_MESH 4

#define GPU_SCORE_FLUX 0
#define GPU_SCORE_TOTAL 1
#define GPU_SCORE_ABSORPTION 2
#define GPU_SCORE_FISSION 3
#define GPU_SCORE_NU_FISSION 4
#define GPU_SCORE_ELASTIC 5
#define GPU_SCORE_SCATTER 6

#define GPU_ESTIMATOR_ANALOG 0
#define GPU_ESTIMATOR_TRACKLENGTH 1
#define GPU_ESTIMATOR_COLLISION 2

struct GpuFilterDesc {
  uint32_gpu type; // GPU_FILTER_*
  uint32_gpu n_bins;
  uint32_gpu map_off; // cell/material/universe: i32 arena map entity->bin
                      // energy: f32 arena offset of n_bins+1 edges
  uint32_gpu stride;  // results-layout stride of this filter
  int32_gpu mesh;     // mesh filter: index into meshes array, else -1
};

#define GPU_MESH_REGULAR 0
#define GPU_MESH_CYLINDRICAL 1

struct GpuMesh { // untranslated/unrotated (flatten rejects those)
  int32_gpu kind;       // GPU_MESH_REGULAR | GPU_MESH_CYLINDRICAL
  int32_gpu n_dim;      // 1..3 (bin layout: StructuredMesh::get_bin_from_indices)
  int32_gpu nx, ny, nz; // shape (# cells per axis; 1 for unused dimensions)
  // --- regular (uniform) ---
  float llx, lly, llz;  // lower_left
  float urx, ury, urz;  // upper_right (fp32 cast; edge rules compare to it)
  float wx, wy, wz;     // element width
  // --- cylindrical (explicit r/phi/z grids) ---
  float ox, oy, oz;     // origin
  uint32_gpu rgrid_off;   // f32 arena: nx+1 radial edges
  uint32_gpu phigrid_off; // f32 arena: ny+1 azimuthal edges (radians)
  uint32_gpu zgrid_off;   // f32 arena: nz+1 axial edges
  int32_gpu full_phi;     // 1 if phi grid spans [0,2pi]
};

// Filters per tally are bounded so the device can enumerate every
// combination of per-level filter matches (cell/universe filters can match
// at several coordinate levels; CPU FilterBinIter scores the product).
#ifndef GPU_MAX_TALLY_FILTERS
#define GPU_MAX_TALLY_FILTERS 4 // runtime MSL compile specializes to the model max
#endif

struct GpuTallyDesc {
  uint32_gpu accum_off; // f32 accumulator arena offset
  uint32_gpu n_filter_bins;
  uint32_gpu n_scores;
  uint32_gpu estimator;  // GPU_ESTIMATOR_*
  uint32_gpu filter_off; // index of first GpuFilterDesc in filter array
  uint32_gpu n_filters;
  uint32_gpu score_off; // i32 arena offset of score codes
};

// ---------------------------------------------------------------------------
// Per-dispatch control block
// ---------------------------------------------------------------------------
#define GPU_RUN_EIGENVALUE 0
#define GPU_RUN_FIXED_SOURCE 1

#define GPU_MODE_MG 0
#define GPU_MODE_CE 1

// Indices into the f32 "reduction slots" buffer: one 8-wide slot group per
// 256-particle block, host-reduced in fp64.
#define GPU_RED_K_TRACKLENGTH 0
#define GPU_RED_K_COLLISION 1
#define GPU_RED_K_ABSORPTION 2
#define GPU_RED_LEAKAGE 3
// per-particle event count, summed on the host: lets kernel cost be quoted
// per event rather than per particle, so timing probes that change how much
// transport happens can still be compared fairly
#define GPU_RED_EVENTS 4
// cross-section evaluations. The result is cached across events at the same
// (material, energy, density), so this bounds how much any cross-section-path
// optimisation can win. Measured at 1.13 per event on the W slab: the cache
// almost never hits, because energy changes at every collision and material
// at every crossing.
#define GPU_RED_XSEVAL 5
// SIMD-group efficiency for the history loop: sum(events) / (width *
// max(events)) across the group. This is the fraction of lane-slots doing
// real work while the group's longest history runs; everything else is
// masked off. Bounds what staging the kernel by operation type could win.
#define GPU_RED_SIMDEFF 6
#define GPU_RED_WIDTH 7

struct GpuControl {
  uint32_gpu n_particles;   // particles this dispatch
  uint32_gpu source_offset; // index of first source site in bank
  uint32_gpu run_mode;      // GPU_RUN_*
  uint32_gpu energy_mode;   // GPU_MODE_*
  float keff;               // running keff for fission-site normalization
  uint32_gpu n_groups;      // MG group count
  uint32_gpu max_events;    // event cap per particle
  uint32_gpu n_tallies;     // active tally count
  uint64_gpu seed_base;     // (total_gen + overall_generation - 1)*n_particles
  uint64_gpu master_seed;
  uint64_gpu prn_stride;
  uint32_gpu fission_bank_cap;
  uint32_gpu secondary_bank_cap;
  int32_gpu root_universe;
  int32_gpu n_coord_levels;
  uint32_gpu n_cells;
  uint32_gpu n_surfaces;
  float energy_min; // CE: transport energy floor (cutoff)
  float energy_max;
  uint32_gpu weight_window_checkpoint_surface; // reserved
  uint32_gpu survival_biasing;                 // reserved (v1: analog)
  float weight_cutoff;
  float weight_survive;
  uint32_gpu mg_bin_avg_off; // f32 arena: MG group mean energies [G]
  // continuous-energy mode
  uint32_gpu n_nuclides;
  uint32_gpu ce_n_log_bins;
  float ce_log_spacing;
  uint32_gpu urr_on;
  uint32_gpu debug_iso_mu; // ablation: force isotropic CM elastic
  int32_gpu trace_id;      // 1-based particle to trace, or -1
  float energy_cutoff;     // CE: kill neutrons below this after a collision
  float free_gas_threshold;     // settings::free_gas_threshold (in kT units)
  uint32_gpu mg_default_iv_off; // f32 arena: default inverse velocity [G]
  // Tally accumulation uses tally_replicas independent fp32 banks (thread
  // tid scores into bank tid % tally_replicas; the host sums banks in
  // fp64). A single fp32 bank saturates on large batches: once a bin's
  // per-batch sum nears 2^24, sub-ulp track contributions round away and
  // the tally biases low (measured: -1.3% on a 4e6-particle-batch
  // deep-penetration front bin, -0.07% at 1e6).
  uint32_gpu tally_accum_stride; // floats per bank (= tally_accum_size)
  uint32_gpu tally_replicas;    // number of banks (power of two)
  // ---- variance reduction (fixed-source, non-multiplying models only) ----
  uint32_gpu ww_on;               // weight windows active
  int32_gpu ww_mesh;              // index into meshes[] for the WW mesh
  uint32_gpu ww_n_energy;         // # energy groups (>=1)
  uint32_gpu ww_ebounds_off;      // f32: ww_n_energy+1 bounds (0 if single)
  uint32_gpu ww_lower_off;        // f32: [n_energy][n_mesh] lower bounds
  uint32_gpu ww_upper_off;        // f32: [n_energy][n_mesh] upper bounds
  uint32_gpu ww_n_mesh_bins;      // mesh bins per energy group
  float ww_survival_ratio;        // survival_weight = lower * ratio
  float ww_max_lb_ratio;
  float ww_weight_cutoff;
  int32_gpu ww_max_split;
  int32_gpu ww_max_history_splits;
  uint32_gpu ww_checkpoint_collision;
  uint32_gpu ww_checkpoint_surface;
  uint32_gpu spill_cap;           // capacity of the spill bank
  uint32_gpu source_is_spill;     // source sites carry their own particle id
  uint32_gpu spill_uid_lo;        // base of the disjoint spill id space
  uint32_gpu spill_uid_hi;
  // Which macroscopic quantities the active tallies actually score. The
  // elastic term costs a per-nuclide loop on EVERY flight, so it is only
  // computed when some tally asks for it (flux-only shielding tallies do
  // not).
  uint32_gpu need_elastic;
  uint32_gpu need_scatter;
  // i32 arena: per-surface adjacency index (n_surfaces offsets, each to a
  // [count, cell...] list of the cells whose region references the surface)
  uint32_gpu surf_adj_off;
};

// device trace record (debug)
struct GpuTraceRec {
  float code; // 0 fly, 1 collide, 2 elastic, 3 inelastic-mt
  float a, b, c;
};

// Runtime counters (device-side atomics), fixed slots in a u32 buffer
#define GPU_CTR_FISSION_BANK 0
#define GPU_CTR_SECONDARY_BANK 1
#define GPU_CTR_LOST 2
#define GPU_CTR_MAX_EVENT_HIT 3
// loss-site classification (diagnostics)
#define GPU_CTR_LOST_INIT 4
#define GPU_CTR_LOST_ADVANCE 5
#define GPU_CTR_LOST_LATTICE 6
#define GPU_CTR_LOST_REFLECT 7
// debug event-trace cursor (OPENMC_TRACE_ID); never aliases a loss counter
#define GPU_CTR_TRACE 8
#define GPU_CTR_LOST_RECONCILE 9
// variance reduction: secondaries spilled to the global bank for the host's
// re-dispatch loop, and secondaries dropped because that bank was full
#define GPU_CTR_SPILL 10
#define GPU_CTR_SPILL_DROP 11
// monotonic within a generation (never reset by the drain loop): supplies
// each spilled secondary a unique particle id
#define GPU_CTR_SPILL_SERIAL 12
#define GPU_CTR_COUNT 13
