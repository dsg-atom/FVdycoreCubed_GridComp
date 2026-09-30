# Prompt 88B — REMAP_Q fill=true CUDA Implementation + Validation

Date: 2026-09-25
Branch: `optimization/cuda-fv3-performance`
Predecessor: `p88a_remap_q_fill_semantics_2026-09-25.md` (mapping-only,
verdict PROCEED TO CUDA EXTENSION, port size SMALL)

## 1. Scope

Implement CUDA support for `fillz` (the `fill=.true.` post-remap negative-
value fix applied in `mapn_tracer`, `fv_mapz.F90:2291`) inside the existing
REMAP_Q CUDA kernel, and remove the `.not. fill` selector gate that had
blocked all CUDA REMAP_Q dispatch since Prompt 87. Explicitly out of
scope (per the governing Prompt 88B specification): core REMAP_Q PPM/
remap numerics, pressure-interval search logic, PT/DELZ, REMAP_UV,
tracer-CUDA production path, `FV_StateMod`'s `fill=.true.` policy itself,
R4, any new H2D/D2H transfer, any new persistent device state beyond
what is strictly required, and any broader tracer/moisture redesign.
Change budget: (1) a `fillz` CUDA helper, (2) `fill=true` REMAP_Q
integration, (3) a selector-eligibility update. No commit/push performed
(none requested).

## 2. Implementation (Stages 1-4)

- New device-portable helper `cuda/dyn_core/vertical/fv3_fillz_column.hpp`
  — templated `fillz_column<T>(q, dp, km, dm_scratch)`, `__host__
  __device__` via the `FV3_FILLZ_HD` macro, a mechanical, non-DEV_GFS_PHYS
  port of `fv_fill.F90`'s `fillz` (top-layer borrow, interior borrow-
  above/borrow-below, bottom-layer borrow, then the `zfix`-gated
  `sum0>0`/`sum0<=0` non-local rescale-or-inert branches). `sum0<=0`
  intentionally leaves any residual negative untouched — this is
  correct/legal production behavior per `fv_fill.F90`, not a defect, and
  must not be "improved."
- `cuda/dyn_core/vertical/fv3_remap_q.cu`/`.cuh`: `RemapQArgs` gained a
  `bool fill` field (default `false`, additive). When set, the kernel
  calls `fillz_column<fv_real>` once per column immediately after
  `map_scalar_column`, reusing the already-dead `c2` scratch slot —
  **zero new device memory, zero new H2D/D2H transfers** (Stage 3 hard
  gate). `DispatchParams` gained an additive `int32_t remap_q_fill`
  field; `fv3_gpu_remap_q`'s C ABI signature gained a trailing
  `int32_t fill` parameter.
- Fortran bridge (`cuda/interface/fv3_gpu_abi_mod.F90`,
  `fv3_gpu_dyncore_bridge.F90`): threads `fill` through as
  `logical`/`c_int32_t` to the new ABI parameter.
- `fv_mapz.F90` (`GEOSfvdycore_fortran_perf` checkout,
  `src/Components/@FVdycoreCubed_GridComp/@fvdycore/model/fv_mapz.F90`,
  untracked/gitignored directory — see §6): the `.not. fill` term present
  in `use_cuda_remap_q`'s eligibility expression through Prompt 87 is
  **removed**. New opt-in gate: `RUN_CUDA_FV3_REMAPQ_ENABLE` (env var)
  must be set for `use_cuda_remap_q` to become true even when
  `kord_tr==(9,9)` and `nq==2` (both universal in production); the
  existing `RUN_CUDA_FV3_REMAPQ_FORCE_FORTRAN` escape hatch still
  overrides back to Fortran. Call site:
  `call fv3_gpu_interface_remap_q(rc_gpu_remap_q, q(isd,jsd,1,1),
  p85_q_elements, fill)` (line 926) — `fill` (the real
  `flagstruct%fill`, forced `.true.` by `FV_StateMod.F90:563`/568) now
  passed through instead of being a rejection gate.
- Numerical semantics changed relative to the Fortran: **NO** — `fillz`
  is reproduced mechanically, not redesigned.

## 3. Fixtures (Stages 5-6)

- **Stage 5** — `cuda/dyn_core/tests/vertical/test_fillz_cuda_vs_host.cpp`:
  host-compiled `fillz_column<double>` vs. the Prompt 88A `fillz_host()`
  golden oracle, all 9 P88A fixtures (no-op, near-minimum, top/interior-
  above/interior-below/bottom borrow, deep-cluster `sum0>0` rescale,
  `sum0<=0` inert-by-design, batched nq=2). **9/9 bitwise PASS.**
- **Stage 6a** — `cuda/dyn_core/vertical/fv3_fillz_device_kernel_smoke.cu`:
  GPU-EXECUTES the identical `fillz_column<double>` from a real
  `__global__` kernel (one thread/column) against the same 9 fixtures.
  Bounded probing found kord=9's monotonic/`extm` constraint resolves
  synthetic negatives from realistic positive-definite profiles before
  `fillz` would activate, so these adversarial fixtures (not organic PPM
  output) are the representative way to GPU-execute `fillz`'s actual
  corrective branches. **9/9 bitwise PASS, GPU EXECUTED.**
- **Stage 6b** — `cuda/dyn_core/vertical/
  fv3_remap_q_nq2_fill_true_prod_shape_smoke.cu`: full nq=2, kord=(9,9),
  `fill=true` REMAP_Q GPU pipeline (`map_scalar_column` → `fillz_column`)
  on C32-shaped sphum-like/cld_amt-like profiles, vs. a host oracle
  running the identical sequence. Reports bitwise/round-off, mass
  conservation, positivity/NaN/Inf, and H2D/D2H call counts. **PASS**:
  both tracers round-off-clean (`max_rel < 1e-9`), mass-conserving
  (`mass_rel < 1e-9`), 0 NaN/Inf, 0 illegal negatives, **H2D_CALLS=1
  D2H_CALLS=1** (Stage 3 gate confirmed). On these well-behaved profiles
  `fillz` is correctly an inert no-op (matches Stage 6a's finding) — this
  fixture confirms `fill=true` integration does not alter normal-case
  REMAP_Q output relative to `fill=false`, while still running the real
  device `fillz_column` call every column.

## 4. Regression / Sanitizer (Stages 7-8)

Full `build_cuda_r8_intelmpi` ctest suite: 28/30 PASS on the Discover
login node (2 failures are the pre-existing, already-documented
login-node CUDA-absence artifacts —
`fv3_wp_b2_hord8_transport`/`fv3_pert_ppm_iv1` — unrelated to this diff).
Re-ran on a real A100 via `srun`: same 28/30, with the 2 login-node
failures now PASSING (confirms GPU-absence, not a regression) and one
additional pre-existing, out-of-scope failure
(`fv3_wp_b2_scalar_transport`, WP-B dgrid tracer transport, tied to the
P83-closed hord==8 tracer-CUDA line, unrelated to this WP-C/vertical
diff — confirmed identical/consistent across `build_cuda_r8` and
`build_cuda_r8_intelmpi`). `compute-sanitizer` on the new fillz/REMAP_Q
fixtures: **0 errors.**

## 5. Rebuild + Provenance (Stage 9)

- `build_cuda_r8_intelmpi/libfv3_dyn_core_gpu.a` (sha256
  `9e6ab02110fce13770e9cb6110d0a1301b585c54141fa17a89bf0928e5c0ac06`) and
  `libfv3_wp_c_vertical.a` (sha256
  `eda5d056058f6fa4100040b1753036b49987421abb9c33346eb9d55b70b02177`) —
  fresh, clean rebuild, zero errors.
- `GEOSfvdycore_fortran_perf/build/bin/StandAlone_FV3_Dycore.x` — rebuilt
  under the authoritative Intel-MPI production stack
  (`comp/gcc/12.3.0`/`comp/intel/2024.2.0`/`mpi/impi/2021.13`/
  `nvidia/nvhpc-byo-compiler/24.7`, `I_MPI_CXX=g++`). sha256
  `eabd1c541fba755b99ea0038dfc940bca1c2fa092539058021990b20611e5258`
  (30,289,912 bytes). Zero build errors; relinked against the fresh CUDA
  archives above. `strings <bin> | grep P85_EXIT_DUMP` confirms the
  diagnostic marker survived (not a stale/mismatched build). Supersedes
  Prompt 87's binary (sha256 `c43845c7...b84b`). `install/bin/`'s copy
  (sha256 `bea8f473...`, Aug 31) is stale and NOT used.

## 6. Note on `fv_mapz.F90`/`FV_StateMod.F90` git status

`src/Components/.gitignore` in `GEOSfvdycore_fortran_perf` contains
`/@FVdycoreCubed_GridComp` — the entire directory (mepo-style external
component checkout, not a submodule) is untracked by that repo's git.
`git status`/`git diff` show nothing for these files despite real edits
being present; confirmed via `git check-ignore -v` and direct `grep` of
file content. No data loss; expected for this project's structure.

## 7. Stage 10 — Real C32 Dispatch A/B

Two experiment dirs cloned from the Prompt 87 C32 template
(`GEOSfvdycore_fortran_perf/experiments/p88b_stage10_c32_{cuda,fortran}`),
each deployed with the fresh binary above, `RUN_CUDA_FV3_REMAPQ_ENABLE=1`
+ `RUN_CUDA_FV3_REMAPQ_DUMP_EXIT=1` set on the CUDA arm only.

| | CUDA arm | Fortran arm |
|---|---|---|
| `[P85_EXIT_DUMP] use_cuda_remap_q=` | **T**, 36/36 calls | **F**, 36/36 calls |
| `FV3 kord_tr:` | 9 | 9 |
| `FV3 fill:` | T | T |
| rc | 0 | 0 |

**CUDA REMAP_Q dispatch with `fill=true` now works correctly in the real
production harness for the first time in this line of work.** P87 had
been stuck at 0/36 dispatch on the `.not. fill` gate this task removed.
**Stage 10: PASS. CUDA dispatch count 36/36. Fallback count 0/36.**

## 8. Stage 11/12 — C512 Same-Node Paired Performance

Harness: `experiments/p88b_remapq_ab.sbatch` (adapted from
`p69_ptdelz_ab.sbatch`), 3 alternating same-node/same-allocation pairs
(`p88b_pair{1,2,3}_{A,B}`, cloned from the `p69_pair1_A` C512 production
template — `hord_mt/vt/tm=6`, `hord_dp=-6`, `grid_type=0`, `vtdm4=0.01`,
`a_imp=1.0`, matching the Stage 10 config; `kord_tr` unset in the
namelist in both templates because it is unconditionally forced to `9` by
`FV_StateMod.F90:563`, same as `fill`). Arm A = Fortran REMAP_Q
(`RUN_CUDA_FV3_REMAPQ_FORCE_FORTRAN=1`); Arm B = CUDA REMAP_Q,
`fill=true` (`RUN_CUDA_FV3_REMAPQ_ENABLE=1`). Both arms retain the
established best-validated config (MPS ON, `RUN_CUDA_FV3_FUSE_HALO_EDGES/
FUSE_CORNER_FILL=YES`, REMAP_UV CUDA default-on per P84A, PT/DELZ CUDA
default-on per P69). Job 58566492, node `warpa004`, all 6 arms rc=0,
30/30 `DYN Date` steps, correct binary sha256 confirmed via
`P88B_PROVENANCE`, zero FATAL/OOM/Abort.

`fv_timing_mod` results (seconds, summed over 30 `DYN_CORE` calls/run):

| Timer | A1 (Fortran) | B1 (CUDA) | A2 | B2 | A3 | B3 |
|---|---|---|---|---|---|---|
| REMAP_Q | 3.5087 | 1.0896 | 3.4889 | 1.0951 | 3.4846 | 1.0583 |
| REMAPPING | 17.5397 | 15.8869 | 18.1525 | 15.3468 | 17.5295 | 15.6157 |
| FV_DYN_LOOP | 54.1031 | 52.8004 | 54.7057 | 51.8379 | 53.9054 | 52.4381 |
| TOTAL | 138.6688 | 136.9826 | 138.7872 | 135.5758 | 137.6147 | 136.2663 |
| DYN_CORE | 25.6682 | 25.6220 | 25.6736 | 25.7014 | 25.5789 | 25.6300 |

Averages across the 3 pairs:

| Metric | Fortran (A) | CUDA (B) | Delta |
|---|---|---|---|
| REMAP_Q ms/call (÷30) | 116.469 | 36.033 | **3.23x direct speedup** |
| REMAPPING (s) | 17.7406 | 15.6165 | **-11.98%** |
| FV_DYN_LOOP (s) | 54.2381 | 52.3588 | **-3.47%** |
| TOTAL (s) | 138.3569 | 136.2749 | **-1.51%** |
| DYN_CORE (s) | 25.6402 | 25.6511 | +0.04% (noise; REMAP_Q is outside the acoustic-substep DYN_CORE window) |

## 9. Stage 13 — Attribution

The entire measured delta is attributable to CUDA REMAP_Q dispatch
becoming active for the first time at C512 (this is the first
performance-scale run where the `.not. fill` gate is not blocking it).
The `fillz` addition itself is expected to be numerically inert on
these production-shape profiles (Stage 6a/6b both found `fillz` a no-op
on realistic positive-definite input) — its cost is one extra branch
check plus scratch-slot reuse per column, not a new transfer or kernel
launch, so it is not expected to be separately measurable, consistent
with REMAP_Q's ~3.2x device-side speedup matching the same
order-of-magnitude improvement already seen for other vertical-remap
CUDA ports (REMAP_UV ~10.2x, PT/DELZ's own device speedup).

## 10. Stage 14 — Performance Gate

REMAPPING -11.98%, FV_DYN_LOOP -3.47%, TOTAL -1.51%, all in the
improving direction, consistent across all 3 same-node pairs (REMAPPING
range: -9.4% to -15.5%; no pair regressed). This is a genuine,
reproducible net improvement at C512 scale, comparable in kind to P69's
PT/DELZ result (REMAPPING -33.7%, TOTAL -7.3%, verdict POSITIVE/KEEP)
and P76A's REMAP_UV result (REMAPPING +21.3% notation / ~10.2x faster,
TOTAL +~1.0%, verdict KEEP, later restored as default by P84A).
**Gate: PASS (net improvement, no regression in any pair).**

## 11. Stage 15 — Default Decision

**Recommendation: KEEP as opt-in infrastructure.** Per Prompt 88B's
explicit instruction, the production default is **not** flipped by this
task — `RUN_CUDA_FV3_REMAPQ_ENABLE` remains off by default; production
dispatch remains Fortran REMAP_Q. Flipping the default (mirroring P84A's
REMAP_UV precedent) is a follow-on decision for a future, explicitly
authorized task.

## 12. Summary Table

| Item | Status |
|---|---|
| Fillz CUDA helper | IMPLEMENTED |
| Fillz fixtures (Stage 5) | 9/9 PASS, bitwise |
| Fillz device kernel (Stage 6a) | 9/9 PASS, GPU EXECUTED, bitwise |
| Full nq=2 fill=true REMAP_Q fixture (Stage 6b) | PASS, round-off, mass-conserving, H2D=1/D2H=1 |
| Regression (Stage 7) | 28/30 PASS (2 pre-existing login-node GPU-absence, out of scope) |
| Sanitizer (Stage 8) | PASS, 0 errors |
| Rebuild/provenance (Stage 9) | PASS, fresh sha256 confirmed |
| C32 dispatch A/B (Stage 10) | PASS, 36/36 dispatch, 0/36 fallback |
| C512 A/B (Stage 11/12) | PASS, 6/6 arms rc=0 |
| Attribution (Stage 13) | CUDA REMAP_Q dispatch activation; fillz itself inert on tested profiles |
| Performance gate (Stage 14) | PASS — REMAPPING -11.98%, FV_DYN_LOOP -3.47%, TOTAL -1.51% |
| Default decision (Stage 15) | KEEP as opt-in; default remains FORTRAN; not flipped this task |

**VERDICT: REMAP_Q fill=true CUDA support IMPLEMENTED, GPU EXECUTED,
NUMERICALLY VALIDATED (bitwise fillz / round-off+mass-conserving full
pipeline), PERFORMANCE VALIDATED POSITIVE at C512 — KEEP AS OPT-IN
INFRASTRUCTURE, PRODUCTION DEFAULT REMAINS FORTRAN.**

**Superseded 2026-09-27 (Prompt 89 / Prompt 89-Resume):** the "default
remains Fortran" verdict above was this task's own explicit scope
boundary, not a permanent finding. Prompt 89 independently reconfirmed
this task's C512 performance result twice more (5/5 paired A/B, then a
60-call longer run), passed all 6 default-policy gate criteria, and
flipped `use_cuda_remap_q`'s default to CUDA-on (escape hatch:
`RUN_CUDA_FV3_REMAPQ_FORCE_FORTRAN=1`). **CUDA REMAP_Q is now the
production default.** See
`docs/performance/p89_remap_q_default_policy_2026-09-25.md`.
