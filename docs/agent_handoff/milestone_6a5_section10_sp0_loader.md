# Milestone 6A.5 Section 10 — SP0 Loader and SP0 Identity Proof

Status: **CLOSED**. Built the minimum Fortran-SP0 -> CUDA H2D loader identified
as the last blocker in `docs/agent_handoff/prompt_6b_cuda_fortran_validation.md`
§7/§10 and `docs/agent_handoff/milestone_6a5_authoritative_grid_metrics.md` §10.
Verified: authoritative Fortran SP0 dynamical state + authoritative c8 grid
metrics + authoritative global `da_min`/`da_min_c` = CUDA validation state
before the first numerical kernel launch. **SP0 IDENTITY: BIT-IDENTICAL**,
60/60 fields (10 fields x 6 tiles). No SP1/SP2/SP3 work attempted — stopped
immediately after SP0 identity passed, per scope.

## 1. Field inventory and index mapping (no missing-CUDA-destination case)

The Prompt 6A SP0 savepoint (`sp0_it0_pe<N>.meta`/`.bin`, N=0..5, c8/npz=10)
exports exactly 10 dyn_core dynamical-state fields: `u, v, w, delp, pt, delz,
q_con, pe, peln, pk`. All 10 have registered CUDA `FieldId` entries
(`cuda/dyn_core/include/fv3_gpu_internal.hpp`) and a `HostState` destination
(`run_small_perf_bench.cpp`). Section 2's stop condition ("if any SP0 field
has no CUDA destination, STOP") was **not triggered**.

Index mapping (derived directly from `make_host_state()`/`make_fields()`, not
from comments), confirmed by 60/60 bit-identical results below:

```
HALO = 3, ni_h = nj_h = NI + 2*HALO + 1 (= 15 for c8)

(i,j,k) fields [u,v,w,delp,pt,delz,q_con,pk]:
  slot = (i+HALO-1) + (j+HALO-1)*ni_h + k_cuda*ni_h*nj_h
  k_cuda = k_fortran - 1

(i,k,j) fields [pe,peln] -- the FV3 pe/peln bounds-order exception:
  slot = (i+HALO-1) + k_cuda*ni_h + (j+HALO-1)*ni_h*npz1
  k_cuda = k_fortran - 1
```

`pk` uses `npz1 = NPZ+1 = 11` levels (interface field); `pe`/`peln` are sized
`ni_h*npz1*nj_h`. All other SP0 fields use `NPZ = 10` levels,
`ni_h*nj_h*NPZ`.

## 2. Converter: Fortran SP0 savepoint -> CUDA flat binaries

`tools/savepoints/convert_sp0_to_cuda.py` (new). Reuses
`tools/savepoints/savepoint_reader.load_field()` — no new binary reader.
Performs index re-embedding only (Fortran-bounds sub-box -> CUDA flat array,
zero-padded outside the sub-box, matching `HostState`'s own
zero-initialization there). Also copies the authoritative **global**
`da_min`/`da_min_c` scalars out of the existing Milestone 6A.5
`tools/grid_metrics/c8_authoritative/tile{0..5}/manifest.json` (`"scalars"`
key) — no new Fortran export.

Run (login node, pure Python, no GPU needed):

```
module load python/GEOSpyD/Min24.4.0-0_py3.11
python3 tools/savepoints/convert_sp0_to_cuda.py \
  <fortran_savepoint_dir> tools/savepoints/sp0_cuda_authoritative
```

Result this session: `da_min=9.338975e+11 da_min_c=8.869285e+11 (global, all 6
tiles identical)`; wrote 10 fields + scalars for all 6 tiles. Output is
gitignored (regenerable), per the same convention as
`tools/grid_metrics/c8_authoritative/`.

## 3. CUDA ingestion

`cuda/dyn_core/benchmarks/run_small_perf_bench.cpp`:

- New `HostState` fields: `override_da_min`, `override_da_min_c` (default
  `-1`, the "not set" sentinel).
- New `sp0_state_6a5` namespace: `load(HostState&, tile, dir)` reads
  `<dir>/tile<pe>/<field>.bin` (pe = tile-1) for all 10 SP0 fields directly
  into `HostState`'s existing buffers, plus `da_min_da_min_c.bin` (2 float64)
  into the two override fields.
- Call site: `setup()`, strictly between `grid_metrics_6a5::load()` and
  `make_fields(*m.h)` — before H2D registration, so the GPU receives whatever
  `HostState` holds regardless of source. Load order is exactly: metrics ->
  global scalars -> SP0 dynamical state -> H2D/bind -> verify device state ->
  STOP, per Task §4.
- `production_config()`: if `override_da_min >= 0`, `cfg.da_min`/`cfg.da_min_c`
  are set from the override (global, Fortran `global_mx`/`global_mx_c`
  reduction) instead of the pre-existing per-tile-local
  `min(area)`/`1/max(rarea_c)` derivation. Guarded by the `-1` sentinel so the
  default/synthetic path is provably unaffected. Both the per-tile-local
  "(derived)" value and the "(AUTHORITATIVE ... override)" value are printed
  when active, so neither replaces the other silently.
- New CLI flags: `--sp0-state DIR`, `--sp0-verify-only`, `--sp0-dump DUMPDIR`.
- New `run_sp0_verify_only()` (anonymous namespace, alongside `run_one_repeat`):
  six-tile `phase_e::setup()` (H2D) -> per-tile `fv3_gpu_sync_to_host()` +
  `fv3_gpu_event_wait()` (D2H, masked to exactly the 10 SP0 `FieldId`s via
  `phase_e::sp0_field_mask()`) -> inline finite/delp-positive checks -> dump to
  disk (`phase_e::dump_sp0_state()`) -> `phase_e::teardown()`. **Never calls
  `run_mode_a`/`run_mode_b`/`fv3_gpu_dyn_core_step`** — confirmed by direct
  code inspection: no such call appears anywhere in `run_sp0_verify_only()`,
  and dispatched from `main()` before (not through) the normal
  `run_one_repeat()` loops, which is the only place those calls occur.

No numerical semantics changed in any authoritative Fortran or CUDA algorithm.

## 4. GPU-side execution

`srun --partition=gpu_a100 --constraint=rome --gres=gpu:1`, existing
`build_cuda_r8_perf/` tree, clean incremental rebuild (no warnings):

```
./fv3_gpu_perf_bench --grid 8 --npz 10 --n_split 3 \
  --grid-metrics ../tools/grid_metrics/c8_authoritative \
  --sp0-state ../tools/savepoints/sp0_cuda_authoritative \
  --sp0-verify-only \
  --sp0-dump ../tools/savepoints/sp0_cuda_dump_run1
```

Result: all 6 tiles report `[6A.5-Sec10] tile N: authoritative SP0 state +
global da_min=9.338975e+11 da_min_c=8.869285e+11 loaded from .../tileM`; all
10 fields finite on every tile (`u,v,w,delp,pt,delz,q_con,pe,peln,pk`); `delp`
positive over the owned 8x8x10 box on every tile; `production_config()`
printed the override line
(`da_min=9.338975e+11 da_min_c=8.869285e+11 (AUTHORITATIVE Fortran global
override, Milestone 6A.5 Sec 10)`) on all 6 tiles, immediately after the
per-tile-local `(derived)` line. Harness printed `=== Milestone 6A.5 Section
10: SP0 verify-only: PASS ===` / `Numerical kernels executed: NO`, `exit 0`.
**GPU EXECUTED: PASS.**

## 5. SP0 identity check (device-side proof)

`tools/savepoints/compare_sp0_identity.py` (new). Reuses
`savepoint_reader.load_field()`/`classify_diff()`/`sanity_check_field()` — no
new comparison logic invented. Gathers the CUDA post-D2H flat dump back into
each field's Fortran-covered sub-box (the exact inverse of the converter's
scatter, same index-mapping constants imported from
`convert_sp0_to_cuda.py` — one mapping definition, not two) and compares
against a direct `load_field()` read of the original Fortran SP0 savepoint.
The harness's uniform-box zero-padding outside each field's authentic Fortran
bounds is excluded from the comparison (an index-registration artifact, not a
Fortran-savepoint value — see Milestone 6A.5 §4's converter note for the same
distinction on the grid-metrics side).

```
python3 tools/savepoints/compare_sp0_identity.py \
  <fortran_savepoint_dir> tools/savepoints/sp0_cuda_dump_run1
```

Result: **60/60 fields BIT-IDENTICAL** (10 fields x 6 tiles), 0 machine-
precision-equivalent, 0 FAIL. No representation-conversion equivalence was
needed — every field matched exactly (0 ULP), because the loader performs a
pure re-embedding of the same float64 values with no arithmetic in between.
All fields finite (sanity-checked inline by the comparator); `delp` positive
(checked as part of the same sanity pass, all pe).

## 6. da_min/da_min_c local-vs-global regression test (Task §3)

Section 3 requires a focused regression test distinguishing validation-mode
(global override) from synthetic/performance-mode (per-tile-local
derivation). Built a synthetic test fixture
(`tools/savepoints/sp0_cuda_sentinel_test/`, symlinked field `.bin`s to the
authoritative converter output, `da_min_da_min_c.bin` overwritten with
distinct sentinel values `1.111111e10`/`2.222222e10` — chosen distinct from
any real value so the test cannot pass by numeric coincidence) and ran two
GPU invocations back to back:

**Test A** (`--sp0-state <sentinel dir> --sp0-verify-only`, no
`--grid-metrics`): six tiles report **six different per-tile-local "(derived)"
values** (`9.800768e9`..`9.908873e9`, confirming the per-tile-local derivation
is genuinely non-global even under this uniform-box fabricated-metric
default), but the override line prints the exact sentinel
(`1.111111e10`/`2.222222e10`) **identically on all 6 tiles**, immediately
after each tile's differing derived line — proving the override branch fires,
takes precedence over the per-tile-local value, and forces a genuinely global
(tile-invariant) value.

**Test B** (no `--sp0-state`, no `--grid-metrics`, default synthetic
benchmark path, `--warmup 0 --iters 1 --repeats 1`): only the six differing
"(derived)" lines print (identical numbers to Test A's derived lines,
confirming the default path's own arithmetic is unaffected); **no override
line appears at all**; harness printed `=== 3A.11 CUDA perf benchmark: PASS
===`, `exit 0`. This is also the Section 7-permitted "one synthetic benchmark
smoke test" proving the default path is unaffected by this milestone's
changes.

This is a code-path/branch-precedence test (sentinel values, not real
Fortran/CUDA data), which is why it is decisive regardless of the
coincidence noted in §7 below.

## 7. Note: real authoritative da_min/da_min_c happens to numerically coincide

With the real authoritative c8 grid metrics loaded (`--grid-metrics`), the
per-tile-local `(derived)` value observed in §4 (`9.338975e+11`) is
numerically identical to the authoritative global override value on this
specific c8 grid — i.e. on real production data, this milestone's override
has **zero visible numerical effect** on this specific configuration. This is
consistent with c8 cubed-sphere tile symmetry (every tile's minimum-area cell
sits at a congruent location) and does not indicate the override is a no-op:
§6's sentinel test proves the branch and precedence are real and load-bearing
in general (e.g. a future non-uniform/adaptive grid, or a different tile
count, where per-tile minima would genuinely differ from the global one).
The pre-existing local-vs-global architectural gap for the *default*
synthetic/performance path (documented in
`milestone_6a5_authoritative_grid_metrics.md` §7) remains, by design —
Section 3 scoped the fix to validation mode only.

## 8. Cost control (Task §7)

Tests run: exactly the three permitted — (1) SP0 loader test (§5, the
identity comparator), (2) authoritative metric/scalar load test (§6, the
da_min/da_min_c distinction), (3) one synthetic benchmark smoke test (§6 Test
B). No SP1/SP2/SP3 support built. No new Fortran savepoints generated. No
full `ctest` sweep run — only `run_small_perf_bench.cpp` (a benchmark
harness, not shared runtime/kernel code) changed, so a full regression matrix
was not required; this mirrors the same judgment made in Milestone 6A.5 §9
though this session did not repeat that full sweep here.

## 9. Outputs

- `tools/savepoints/convert_sp0_to_cuda.py` — SP0 Fortran-savepoint ->
  CUDA-flat-binary converter (new).
- `tools/savepoints/compare_sp0_identity.py` — SP0 identity comparator (new).
- `cuda/dyn_core/benchmarks/run_small_perf_bench.cpp` — `sp0_state_6a5`
  namespace, `override_da_min`/`override_da_min_c`, `--sp0-state`/
  `--sp0-verify-only`/`--sp0-dump` flags, `run_sp0_verify_only()`,
  `phase_e::sp0_field_mask()`/`phase_e::dump_sp0_state()`.
- `.gitignore` — added `tools/savepoints/sp0_cuda_authoritative/`,
  `tools/savepoints/sp0_cuda_dump_*/`, `tools/savepoints/sp0_cuda_sentinel_test/`
  (all derived/generated, regenerable, not committed).
- `docs/agent_handoff/milestone_6a5_authoritative_grid_metrics.md` — §10
  updated to point here.
- `docs/agent_handoff/CURRENT_STATE.md` — §7/§8/§9/§10 updated (Prompt 6B
  no longer blocked on the SP0 loader; SP1 capture is the next concrete
  step).

No large binaries committed. Nothing pushed or merged (not requested).

---

## FINAL REPORT

- Task: Milestone 6A.5 Section 10 closing proof (SP0 loader + SP0 identity)
- Branch: `cuda/gpu-bringup`
- §1 REUSE EXISTING SAVEPOINT READER: DONE — `savepoint_reader.load_field()`/
  `classify_diff()`/`sanity_check_field()` reused verbatim, no second format
- §2 MINIMAL SP0 INGESTION PATH: DONE — all 10 SP0 fields mapped to existing
  CUDA `FieldId`/`HostState` destinations; `pe`/`peln` (i,k,j) exception
  handled explicitly and distinctly from the (i,j,k) fields; stop condition
  not triggered
- §3 GLOBAL da_min/da_min_c INJECTION: DONE — validation-mode-only override,
  guarded by a sentinel, default/synthetic path unaffected (regression test
  §6/Test B), source documented per mode (both "(derived)" and
  "(AUTHORITATIVE ... override)" lines print when active)
- §4 VALIDATION LOAD ORDER: DONE, exact order enforced by code structure
  (metrics -> scalars -> SP0 state -> H2D -> verify -> STOP); zero
  `fv3_gpu_dyn_core_step` calls in `run_sp0_verify_only()`, confirmed by
  direct code inspection
- §5 SP0 IDENTITY CHECK: DONE — **BIT-IDENTICAL, 60/60 fields** (10 fields x
  6 tiles), 0 machine-precision-equivalent, 0 FAIL; no full array dumps, only
  compact per-field verdicts
- §6 DEVICE-SIDE PROOF: DONE — D2H via `fv3_gpu_sync_to_host` (not host
  parsing alone); all fields finite; delp positive; global da_min/da_min_c
  identical across all 6 tiles; all six tiles correct; zero numerical
  kernels executed
- §7 COST CONTROL: DONE — exactly 3 tests run (SP0 loader test, scalar/metric
  load distinction test, one synthetic smoke test); no SP1-3 work, no new
  Fortran savepoints, no full ctest sweep, no large regression matrix
- §8 ACCEPTANCE: all criteria met (see §1-§7 above)
- §9 DOCUMENTATION: this doc (new); `milestone_6a5_authoritative_grid_metrics.md`
  §10 updated to point here; `CURRENT_STATE.md` updated (project status
  materially changed — Prompt 6B's last blocker is closed)
- Numerical semantics changed in any authoritative Fortran or CUDA algorithm:
  **NO**
- Nothing committed, pushed, or merged (not requested)

AUTHORITATIVE STARTING STATE VERIFIED — RESUME PROMPT 6B
