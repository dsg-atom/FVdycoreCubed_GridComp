# Milestone 6A.5 — Authoritative c8 Grid Metric Export and CUDA Ingestion

Status: **CLOSED**. Authoritative non-uniform c8 grid metrics are now
exported from the real Fortran `gridstruct`, converted, and ingested by the
CUDA validation harness behind a validation-only, opt-in flag. The default
(fabricated-metric) benchmark path is unchanged and unaffected.

## 1. Authoritative metric inventory

Source: `Atm%gridstruct`, exported by `fv3_sp_dump_grid_metrics`
(`fv3_savepoint_mod.F90`, GEOSfvdycore_fortran_perf fixture), savepoint name
`spgrid`, one export per Fortran MPI rank (pe 0..5, one per cubed-sphere
tile), it=0 (static, dumped once at init).

37 metric entries per pe: 31 single-component 2D fields, 2 four/nine-
component 2D fields (`sin_sg`, `cos_sg`, Fortran exports all 9 FV3 sg-slots,
CUDA consumes slots 1-4), 2 global scalars (`da_min`, `da_min_c`,
Fortran-side `global_mx`/`global_mx_c` reductions — identical across all 6
tiles, confirmed empirically: `da_min=9.338975e+11`, `da_min_c=8.869285e+11`).

Full field/dims/bounds/staggering table (one row per field): see
`spgrid_it0_pe0.meta` in either export run directory (§2) — it already *is*
the authoritative bounds table; no value is duplicated here to avoid drift.
Representative bounds classes (c8, halo=3, `isd..ied+1`-style variants):

| class | example fields | bounds pattern |
|---|---|---|
| cell-center | `area, rarea, dxa, dya, cosa_s, rsin2, f0` | `(-2,11)x(-2,11)` |
| u-staggered | `dx, rdx, dyc, rdyc, cosa_v, sina_v, rsin_v, divg_v, del6_v` | `(-2,11)x(-2,12)` |
| v-staggered | `dy, rdy, dxc, rdxc, cosa_u, sina_u, rsin_u, divg_u, del6_u` | `(-2,12)x(-2,11)` |
| corner | `sina, cosa, fC, area_c, rarea_c` | `(-2,12)x(-2,12)` |
| no-halo | `rsina` | `(1,9)x(1,9)` |
| 4-component (sg) | `sin_sg, cos_sg` | `(-2,11)x(-2,11)x(1,9)` |

CUDA-consumer mapping: `HostState` (`run_small_perf_bench.cpp:193-279`)
fabricates all of the above on ONE uniform box `[0,ni_h)x[0,nj_h)`,
`ni_h = NI + 2*HALO + 1` (= 15 for c8), `idx = i + j*ni_h`. Explicitly
**excluded** from this milestone's scope (dynamical-state/transient, not
static grid, despite superficially being 2D): `divg2, ws3, phis, zs, ws`.
Confirmed by grep that `grid_metrics_6a5::load()` references none of them.

## 2. Export

Fixture: `GEOSfvdycore_fortran_perf` (fixture HEAD
`8ba7bdd4f7f647cb6e6d70060c131e20938f1f0c`), c8/npz=10/n_split=3/R8/6-rank
production first-run config (`test_case=14`, `hord_mt/vt/tm=6, hord_dp=-6,
nord=2, d4_bg=0.12, vtdm4=0.01, a_imp=1.0, grid_type=0`) — same config as
Prompt 6A.

Instrumentation gap found and fixed this milestone: `area_c`/`rarea_c`
(dual/corner-cell area and its reciprocal — the authoritative counterpart to
CUDA's fabricated `rarea_c`) were missing from the first export pass. Added
two `fv3_sp_dump2d` calls to `fv3_sp_dump_grid_metrics`
(`fv3_savepoint_mod.F90`); fixture rebuilt (`make -j8
StandAlone_FV3_Dycore.x`, exit 0); `install/bin/` binary refreshed via the
same `cp`-after-`make`-bypass workaround used in Prompt 6A (unrelated
pre-existing `read_ops_bcs` f2py `make install` failure, not touched).

Export jobs (corrected, final):
`experiments/c8_L10_grid_metrics_6a5_run{1,2}/`, jobs `58115069`/`58115070`,
both COMPLETED. 228 `spgrid_*` files per run (38 per pe x 6 pe).

## 3. Reproducibility

`cmp -s` over every `spgrid_*` file, run1 vs run2: **mismatch=0** (complete,
corrected field set, 228 files). Bit-identical across two independent runs.

## 4. CUDA ingestion

`cuda/dyn_core/benchmarks/run_small_perf_bench.cpp`:

- New `--grid-metrics <dir>` CLI flag (default: unset/empty — fabricated
  path unchanged).
- New `grid_metrics_6a5::load(HostState&, int tile, dir)`: reads
  `<dir>/tile<pe>/<field>.bin` (pe = tile - 1; CUDA `tile` is 1-based, the
  converter's `tile0..tile5` output is 0-based Fortran-pe-indexed) for all 31
  single-component fields plus `sin_sg`/`cos_sg`, overwriting `HostState`'s
  fabricated values in place.
- Call site: `setup()`, strictly between
  `make_host_state(tile)`/`perturb()` and `make_fields(*m.h)` — i.e. before
  the buffers are registered for H2D transfer, so the GPU receives whatever
  `HostState` holds at that point regardless of source. No kernel or device
  code touched.
- No numerical semantics changed when the flag is unset (verified: fabricated
  path re-run, `PASS`, exit 0, same benchmark harness behavior as before this
  milestone).

Converter: `tools/grid_metrics/convert_fortran_grid_metrics.py`, reuses
`tools/savepoints/savepoint_reader.load_field()` (Prompt 6A infrastructure,
no new binary reader written). Remaps each field's authoritative
Fortran-staggered bounds onto the harness's uniform box via nearest-neighbor
clamp (`fv(a) = a - HALO + 1`, out-of-bounds Fortran indices clamped to the
nearest valid one — the union-box padding region is a harness index-
registration artifact from Milestone 3A.8, not a physical extrapolation).

**Bug found and fixed during this milestone**: the converter's first version
wrote `.bin` files in `i*NJ_H + j` (numpy default C-order) flat order, but
the CUDA harness's convention is `idx = i + j*ni_h` (`j*NI_H + i`) — i.e. the
converter was writing the transpose of what CUDA expects. An earlier ad-hoc
verification pass only sampled diagonal points `(a,b)` with `a==b`, which is
invariant to this bug (`i*N+i == i*N+i` regardless of row/column-major) and
so did not catch it. Fixed by writing `remapped.T` (single fields) /
per-component `.T` (sin_sg/cos_sg) before `tofile()`. Re-verified against
**off-diagonal** points (§5) — this is exactly the class of bug the new
permanent test (§11) exists to catch if it recurs.

Output location: `tools/grid_metrics/c8_authoritative/tile{0..5}/*.bin` +
`manifest.json` — **gitignored** (`.gitignore`: `tools/grid_metrics/c8_authoritative/`),
regenerable from the Fortran export via the converter; not committed, per
Task B §11 ("no large binary commits"). The raw Fortran `spgrid_*` export
itself remains on the shared filesystem only (same convention as the Prompt
6A savepoints), not in this repo.

## 5. CPU-side verification (pre-GPU)

Off-diagonal spot-checks (`(0,5),(5,0),(7,3),(3,7),(14,2),(2,14)`) of
converted `dx.bin` (single-component) and `sin_sg.bin` (4-component) against
direct `load_field()` reads of the raw Fortran export: **all matched
exactly** post-transpose-fix. Formalized as a permanent test (§11) and run
across all 6 tiles: `ALL_OK`.

## 6. GPU-side verification

`srun --partition=gpu_a100 --constraint=rome --gres=gpu:1`, built
`fv3_gpu_perf_bench` in the existing `build_cuda_r8_perf/` tree (clean
compile, no warnings from the new code), ran:

```
./fv3_gpu_perf_bench --grid 8 --npz 10 --n_split 3 --warmup 1 --iters 2 \
  --repeats 1 --mode resident --hardware A100_6A5 \
  --grid-metrics ../tools/grid_metrics/c8_authoritative
```

Result: all 6 tiles report `[6A.5] tile N: authoritative grid metrics loaded
from .../tileM`; all fields finite; `delp` positive over the owned box on
every tile; benchmark harness prints `=== 3A.11 CUDA perf benchmark: PASS
===`; `exit 0`. **GPU EXECUTED: PASS.** (Per project terminology: this is GPU
execution with authoritative metrics loaded, NOT numerical validation —
Prompt 6B is still required for that, and remains NOT RUN.)

`u`/`delp` deltas vs. tile-start state are of different magnitude than the
fabricated-metric run (expected — non-uniform authoritative metrics change
the numerics quantitatively, as intended), but remain finite and
delp-positive — no instability introduced by the metric swap in this short
smoke run.

## 7. Fabrication audit

`grid_metrics_6a5::load()` overrides exactly: 31 single-component fields
(`dx, dy, dxa, dya, rdx, rdy, rdxa, rdya, dxc, dyc, rdxc, rdyc, area, rarea,
rarea_c, f0, f_c, cosa_u, sina_u, rsin_u, cosa_v, sina_v, rsin_v, cosa_s,
rsin2, cosa, rsina, divg_u, del6_u, divg_v, del6_v`) and 2 four-component
fields (`sin_sg, cos_sg`) — the complete static-grid-metric subset of
`HostState`'s fabrication block. Confirmed by grep: none of the intentionally
out-of-scope dynamical/transient fields (`divg2, ws3, phis, zs, ws`) are
referenced by the loader.

**Known pre-existing discrepancy, NOT fixed in this milestone** (out of
scope — would require new cross-tile/cross-rank reduction engineering, not
metric ingestion): Fortran's `da_min`/`da_min_c` are **global** (all-6-tile)
reductions (confirmed: identical value on every exported tile/pe). The CUDA
harness's own `cfg.da_min`/`cfg.da_min_c` derivation (`production_config()`)
computes a **per-tile-local** min over that tile's own `area`/`rarea_c` —
even after this milestone's metric ingestion, that per-tile-local derivation
is unchanged (the exported global scalars are recorded in each tile's
`manifest.json` for audit but are **not** injected into `cfg`). Flagged here
for whoever next touches damping-config derivation or resumes Prompt 6B
numerical comparison; not on this milestone's critical path (ingestion is
metric *data*, not damping-config derivation).

When `--grid-metrics` is unset, zero fabricated metrics are overridden —
verified by re-running the fabricated path this session (§4) and by the code
structure itself (`load()` is only reachable inside the
`if (!g_grid_metrics_dir.empty())` guard).

## 8. Non-uniformity sanity check

Authoritative metrics are genuinely non-uniform, unlike the fabricated
uniform-box metrics they replace:

- `dx`: 980305.3–1375622.8 m (17 distinct rounded values) vs. fabricated
  uniform 100000.0 m.
- `area`: up to 1.86e12 m^2, with a `±1.0e8` sentinel at certain
  cubed-sphere diagonal-corner halo cells — this is the well-known FV3
  convention of filling structurally-invalid corner-halo cells with a large
  sentinel so inadvertent stencil use is detectable; not a defect.
- `sin_sg`: spans 1e-08 to 1.0 (51 distinct rounded values).

## 9. Regression

Full `ctest` in `build_cuda_r8_perf/` (A100, same Slurm session as §6):
**22/22 PASS** (one target, `fv3_cube_seam_test`, had never been built in
this particular build tree before this session — confirmed pre-existing and
unrelated to the 6A.5 edits by building it standalone with no errors; not a
regression). Fabricated-metric benchmark path re-run without
`--grid-metrics`: unchanged behavior, `PASS`, `exit 0`.

## 10. Prompt 6B resumption

**CLOSED (addendum, later session)** — the SP0-state loader flagged below as
the next concrete piece of work has since been built and the SP0 identity
check has passed **BIT-IDENTICAL, 60/60 fields**. Full detail:
`docs/agent_handoff/milestone_6a5_section10_sp0_loader.md`. Original note
preserved below for history.

~~NOT attempted beyond the scope explicitly authorized.~~ Prompt 6B SP0
comparison requires a Fortran-SP0-dynamical-state -> CUDA H2D loader, which
Prompt 6B's own report states was never built — that is new infrastructure
beyond grid-metric ingestion and is out of this milestone's scope per Task B
§10's instruction to stop at "SP0-identity verification" without building
new SP0/SP1-3 infrastructure. Per §10 of Task B: STOP here. Authoritative
metrics are ready; the SP0-state loader is the next concrete piece of work
for whoever resumes Prompt 6B.

## 11. Outputs

- `tools/grid_metrics/convert_fortran_grid_metrics.py` — converter (fixed
  transpose bug this session).
- `tools/grid_metrics/test_grid_metrics_ingestion.py` — permanent focused
  regression test (off-diagonal spot-checks, all 6 tiles, single- and
  multi-component fields). Run:
  ```
  module load python/GEOSpyD/Min24.4.0-0_py3.11
  python3 tools/grid_metrics/test_grid_metrics_ingestion.py \
    <fortran_savepoint_dir> tools/grid_metrics/c8_authoritative
  ```
  Result this session: `ALL_OK`.
- `tools/grid_metrics/c8_authoritative/` — converted per-tile `.bin` +
  `manifest.json` (gitignored, regenerable, not committed).
- `cuda/dyn_core/benchmarks/run_small_perf_bench.cpp` — `--grid-metrics` flag
  + `grid_metrics_6a5::load()` + call site (validation-only, gated,
  fabricated path unaffected).
- `.gitignore` — added `tools/grid_metrics/c8_authoritative/`.
- `GEOSfvdycore_fortran_perf` fixture (separate tree):
  `fv3_savepoint_mod.F90` — added `area_c`/`rarea_c` dumps (gap fix).

No large binaries committed to this repo. Raw Fortran `spgrid_*` export
remains on the shared filesystem only:
`GEOSfvdycore_fortran_perf/experiments/c8_L10_grid_metrics_6a5_run{1,2}/savepoints/`.

## 12. CURRENT_STATE.md

Not updated. Nothing in `CURRENT_STATE.md`'s governing facts changed:
Prompt 6B numerical comparison is still NOT RUN; the CUDA vs. Fortran
dynamical-state comparison scope, blockers, and next steps described there
are unaffected. This milestone only replaces the CUDA harness's *input grid
metrics* with authoritative ones (opt-in) — it does not touch, run, or
close any part of the SP0-SP3 comparison itself. §5/§9 of `CURRENT_STATE.md`
remain accurate as written.

---

## FINAL REPORT

- Milestone: 6A.5 — Authoritative c8 Grid Metric Export and CUDA Ingestion
- Branch: `cuda/gpu-bringup`
- §1 metrics table: DONE (see §1 above + `spgrid_it0_pe0.meta`)
- §2 export: DONE (jobs 58115069/58115070, COMPLETED, 228 files/run)
- §3 reproducibility: DONE (bit-identical, mismatch=0)
- §4 CUDA ingestion: DONE (`--grid-metrics` flag, `grid_metrics_6a5::load()`,
  compiles clean, fabricated path unaffected)
- §5 CPU-side verification: DONE (off-diagonal spot-check + permanent test,
  `ALL_OK`, all 6 tiles)
- §6 GPU-side verification: DONE (A100 `gpu_a100`/`rome`, GPU EXECUTED: PASS,
  exit 0, all fields finite, delp positive)
- §7 fabrication audit: DONE (31+2 fields overridden; dynamical/transient
  fields correctly excluded; da_min/da_min_c local-vs-global discrepancy
  flagged, not fixed, out of scope)
- §8 non-uniformity check: DONE (dx/area/sin_sg genuinely non-uniform;
  corner sentinel explained)
- §9 regression: DONE (22/22 ctest PASS; fabricated path unchanged)
- §10 Prompt 6B resumption: STOPPED as instructed — SP0/SP1-3 dynamical
  loader is explicitly out of scope for this milestone
- §11 outputs: DONE (converter, test, gitignored data dir, one CUDA source
  file changed, one Fortran fixture file changed, this handoff doc)
- §12 documentation: DONE (this doc); `CURRENT_STATE.md` NOT updated (no
  governing fact changed)
- Numerical semantics changed in any authoritative Fortran or CUDA
  algorithm: **NO** (grid-metric data ingestion only; the one Fortran change
  is an additive export-only dump of already-existing `gridstruct` fields;
  the CUDA changes are gated behind an opt-in, default-off flag)
- Nothing committed, pushed, or merged (not requested).

AUTHORITATIVE GRID METRICS READY — RESUME PROMPT 6B
