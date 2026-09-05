/* Prompt 14/16 resumption -- two small C entry points needed ONLY by the Fortran
 * field registry (cuda/interface/fv3_gpu_field_registry.F90).
 *
 * Both exist to keep an existing project mechanism the single source of truth
 * instead of transcribing it into Fortran, per the reuse rule in CLAUDE.md:
 *
 *   1. fv3_gpu_field_spec_query -- the stable field specification table
 *      (runtime/fv3_gpu_contract.cpp specs[]) decides the stagger, access mode
 *      and element width every descriptor must carry; make_device_view rejects
 *      any descriptor that disagrees. Transcribing 100 rows of (stagger,
 *      access) into Fortran would create a second table that can silently drift
 *      from the first. The registry queries this one instead, so a stagger
 *      mismatch is structurally impossible rather than merely unlikely.
 *
 *   2. fv3_gpu_registry_damping_column -- fv3_damping_translation.hpp is the
 *      project's single authoritative host-side translation of the per-level
 *      damping block at dyn_core.F90:677-765. It returns std::vector, so it
 *      cannot cross a C ABI directly; this shim copies into caller-provided
 *      arrays. The Fortran side does NOT re-derive the sponge arms.
 *
 * Neither function is part of the public dyn_core ABI (fv3_gpu_api.h): both are
 * registry-construction helpers, and the Fortran registry declares their
 * interfaces locally for exactly that reason.
 *
 * Numerical semantics changed: NO -- no formula is implemented here.
 */

#include "fv3_gpu_internal.hpp"
#include "fv3_damping_translation.hpp"

#include <cstdint>
#include <vector>

extern "C" {

/* Report what the stable field specification requires of field `field_id`.
 *
 * Returns 0 on success, 1 if the id is not in the table.
 *
 * element_bytes is the width make_device_view will insist on, or 0 when the
 * contract leaves it to the descriptor (the three integral damping-order
 * arrays; their consumers read them through an fv_real view, so the caller
 * should then use the fv_real width).
 */
int fv3_gpu_field_spec_query(int32_t field_id, int32_t *rank, int32_t *stagger,
                             int32_t *access, int32_t *element_bytes) {
  using namespace fv3::gpu;
  const FieldId id = field_id_from_abi(static_cast<uint32_t>(field_id));
  if (id == FieldId::invalid) return 1;
  const FieldSpec *s = field_spec(id);
  if (!s) return 1;
  if (rank) *rank = static_cast<int32_t>(s->rank);
  if (stagger) *stagger = static_cast<int32_t>(s->stagger);
  if (access) *access = static_cast<int32_t>(s->access);
  if (element_bytes) {
    /* Mirrors the width rule make_device_view applies (fv3_gpu_contract.cpp):
     * accumulator_real is pinned at 8 bytes regardless of build precision,
     * fv_real follows the build, and anything else is unconstrained. */
    if (s->logical_type == LogicalType::accumulator_real)
      *element_bytes = 8;
    else if (s->logical_type == LogicalType::fv_real)
      *element_bytes = static_cast<int32_t>(sizeof(fv_real));
    else
      *element_bytes = 0;
  }
  return 0;
}

/* Run the authoritative damping translation once and copy the ten per-level
 * columns the storage contract carries into caller-provided arrays.
 *
 * Every array must be at least npz elements long. The three order columns are
 * emitted as doubles holding exactly-representable integers, which is what the
 * WP-B binder expects of damping_ndif / damping_nord_k / damping_nord_w.
 *
 * Returns 0 on success, 1 on a null/degenerate argument, 2 if the translation
 * produced a column shorter than npz.
 */
int fv3_gpu_registry_damping_column(
    int32_t npz, int32_t nord, int32_t n_sponge, int32_t n_zfilter,
    int32_t do_vort_damp, int32_t hydrostatic, double dddmp, double d2_bg,
    double d4_bg_top, double d4_bg_bot, double d2_bg_k1, double d2_bg_k2,
    double vtdm4, double d_con, double d_ext, double ke_bg, double ptop,
    const double *pfull, double *out_d2_bg, double *out_d4_bg,
    double *out_damp_v, double *out_damp_w, double *out_damp_t,
    double *out_d_con, double *out_kgb, double *out_ndif, double *out_nord_k,
    double *out_nord_w) {
  using namespace fv3::gpu::damping;
  if (npz <= 0 || !pfull || !out_d2_bg || !out_d4_bg || !out_damp_v ||
      !out_damp_w || !out_damp_t || !out_d_con || !out_kgb || !out_ndif ||
      !out_nord_k || !out_nord_w)
    return 1;

  DampingFlags f{};
  f.npz = static_cast<int>(npz);
  f.nord = static_cast<int>(nord);
  f.n_sponge = static_cast<int>(n_sponge);
  f.n_zfilter = static_cast<int>(n_zfilter);
  f.do_vort_damp = do_vort_damp != 0;
  f.hydrostatic = hydrostatic != 0;
  f.dddmp = dddmp;
  f.d2_bg = d2_bg;
  f.d4_bg_top = d4_bg_top;
  f.d4_bg_bot = d4_bg_bot;
  f.d2_bg_k1 = d2_bg_k1;
  f.d2_bg_k2 = d2_bg_k2;
  f.vtdm4 = vtdm4;
  f.d_con = d_con;
  f.d_ext = d_ext;
  f.ke_bg = ke_bg;
  f.ptop = ptop;

  const DampingColumn c = translate_damping(f, pfull);
  const std::size_t n = static_cast<std::size_t>(npz);
  if (c.d2_bg.size() < n || c.d4_bg.size() < n || c.damp_v.size() < n ||
      c.damp_w.size() < n || c.damp_t.size() < n || c.d_con.size() < n ||
      c.kgb.size() < n || c.ndif.size() < n || c.nord_k.size() < n ||
      c.nord_w.size() < n)
    return 2;

  for (std::size_t k = 0; k < n; ++k) {
    out_d2_bg[k] = c.d2_bg[k];
    out_d4_bg[k] = c.d4_bg[k];
    out_damp_v[k] = c.damp_v[k];
    out_damp_w[k] = c.damp_w[k];
    out_damp_t[k] = c.damp_t[k];
    out_d_con[k] = c.d_con[k];
    out_kgb[k] = c.kgb[k];
    out_ndif[k] = static_cast<double>(c.ndif[k]);
    out_nord_k[k] = static_cast<double>(c.nord_k[k]);
    out_nord_w[k] = static_cast<double>(c.nord_w[k]);
  }
  return 0;
}

} // extern "C"
