# Prompt 99 — CUDA Implementation of `State_To_FV` (Option A: local-compute port)

Date: 2026-09-28
Branch: `optimization/cuda-fv3-performance`
Scope: CUDA implementation (Phase A + Phase C local-compute kernels) of
`State_To_FV`, per the Option A design recommended by
[p98_state_to_fv_cuda_port_plan_2026-09-28.md](p98_state_to_fv_cuda_port_plan_2026-09-28.md).
`mpp_get_boundary` (Phase B) is UNCHANGED, host-only, exactly as before.
No CUDA-aware MPI. No commit/push. No R4. Production default remains
Fortran.

## 0. Architecture recap

Three-phase split, by strict Fortran program order relative to the one
`mpp_get_boundary` call `State_To_FV` makes (`FV_StateMod.F90:2519-2702`):

```
GPU Phase A local work (u/v tiny_number fill + interior copy)
  -> D2H only the halo-required U/V FieldRegistry buffers
  -> existing host mpp_get_boundary (UNCHANGED)
  -> H2D updated U/V halos
  -> GPU Phase C local work (w, pe transpose, peln/pk, ps pack, delp,
     pt, delz, pkz)
  -> existing fv_dynamics() integration (already-GPU-resident, no
     redundant transfer introduced)
```

Full field-level contract, index-order contract, and rationale for the
Phase A/B/C boundary are documented in the implementation header:
`cuda/dyn_core/vertical/fv3_state_to_fv.cuh`. That header is the
authoritative as-built contract; this document records the mapping to
the actual Fortran call sites, the selector, the validation pipeline
(Stages 9-16 in the resume prompt's numbering), and the performance
result.

Numerical semantics changed relative to the Fortran: **NO** (elementwise
copies, one index transpose, and the same `log`/`exp` formula already in
production via REMAP_PRESSUREVARS — no algorithm changed).

## 1. Fortran call sites and selector

`State_To_FV` (`FV_StateMod.F90:2522-2798`) is called from three sites:

- **Line 1174** — warm-restart branch (non-doubly-periodic). Not
  exercised by any test run in this task (`COLDSTART: 1` used
  throughout).
- **Line 1186** — `COLDSTART` branch inside `FV_InitState`. Exercised
  exactly once per rank at model start, **strictly before**
  `fv3_gpu_bridge_init` (line 1264, deliberately positioned "at the end
  of initialization" per its own header comment). No GPU context exists
  yet at this one-time seeding call, so CUDA dispatch is architecturally
  impossible here regardless of selector state — this is the source of
  the benign, expected exactly-one-`F`-per-rank pattern documented in
  Stage 13 below, not a defect.
- **Line 1869** — the real per-`k_split`-substep call inside `FV_Run`,
  immediately before `fv_dynamics()`. This is the site the CUDA port
  actually targets performance-wise.

Selector (`FV_StateMod.F90:2591-2601`), same pattern as
`use_cuda_remap_pressurevars`/`use_cuda_remap_q`:

```fortran
p99_base_gate = (FV_Atm(1)%flagstruct%grid_type < 4) .and. &
     (.not. SW_DYNAMICS) .and. (.not. FV_Atm(1)%flagstruct%hydrostatic) .and. &
     (.not. ADJUST_SPLITS) .and. (kind(STATE%VARS%U) == kind(1.0d0)) .and. &
     fv3_gpu_backend_enabled()
use_cuda_state_to_fv = .false.
call get_environment_variable('RUN_CUDA_FV3_STATE_TO_FV_ENABLE', p99_envval, p99_envlen)
if (p99_envlen > 0) use_cuda_state_to_fv = p99_base_gate
call get_environment_variable('RUN_CUDA_FV3_STATE_TO_FV_FORCE_FORTRAN', p99_envval, p99_envlen)
if (p99_envlen > 0) use_cuda_state_to_fv = .false.
```

- **Opt-in only**: `RUN_CUDA_FV3_STATE_TO_FV_ENABLE=1` is required to
  even consider CUDA; unset (the production default) means Fortran runs
  unconditionally.
- **Escape hatch**: `RUN_CUDA_FV3_STATE_TO_FV_FORCE_FORTRAN=1` always
  forces Fortran regardless of the enable flag.
- **Gate**: R8-only (`kind(STATE%VARS%U)==kind(1.0d0)`), `grid_type<4`
  (cubed-sphere only, doubly-periodic excluded), `!SW_DYNAMICS`,
  `!hydrostatic`, `!ADJUST_SPLITS` (all DEAD branches per Stage 2 of the
  P98 mapping — unsupported configurations fall back to Fortran, model
  state is never mutated by a partially-implemented CUDA path), and
  `fv3_gpu_backend_enabled()` (the persistent bridge must already be
  initialized).
- **Production default: FORTRAN.** `RUN_CUDA_FV3_STATE_TO_FV_ENABLE` is
  not set in any standing production job script; this task does not
  change that.

Diagnostic instrumentation (opt-in, `RUN_CUDA_FV3_STATE_TO_FV_DUMP_EXIT`):
prints per-call rank/selector-flag/field-midpoint samples
(`[P99_EXIT_DUMP]`) for `u`/`v`/`pe`/`pt`/`delz`/`pkz`/`ps`. Test/debug
infrastructure only, zero cost when unset (single
`get_environment_variable` check), consistent with the project's
existing `env_flag_enabled` opt-in-diagnostic convention.

## 2. CUDA implementation

- `cuda/dyn_core/vertical/fv3_state_to_fv.cuh` / `.cu` — Phase A and
  Phase C kernels, full contract documented in the header (field-by-
  field, index-order, residency rationale). See excerpt in §0.
- `cuda/dyn_core/runtime/fv3_gpu_runtime.cpp` —
  `fv3_gpu_state_to_fv_phase_a`/`fv3_gpu_state_to_fv_phase_c`:
  - Phase A: H2D of `STATE%VARS%U/V` into dedicated persistent
    `StateInputView` buffers (no FieldRegistry identity, no halo, exact
    `is:ie x js:je` sizing — these MAPL State fields carry no ghost
    cells), `fv3_fill_device_constant` tiny_number fill on `u`/`v`
    FieldRegistry buffers, kernel launch, D2H of the **full** `u`/`v`
    FieldRegistry buffers back to the Fortran bridge (the existing
    `fv3_gpu_sync_to_host` u/v-mask mechanism) so the unmodified host
    `mpp_get_boundary` can run.
  - Phase C: H2D of `STATE%VARS%W/PE/PT/PKZ/DZ`, `fv3_fill_device_constant`
    fills on `pe`/`delp`/`pt` (same program order as the Fortran:
    pe, then delp, then pt, so halo/skipped cells hold `tiny_number`
    exactly as the authoritative routine leaves them), kernel launch,
    then a single D2H of the packed `ps` scratch buffer directly to the
    caller's host `ps` pointer. `pe`/`peln`/`pk`/`delp`/`pt`/`delz`/`pkz`
    stay **device-resident** — no D2H — feeding directly into the
    already-GPU-resident `fv_dynamics()` CUDA path with no redundant
    transfer introduced.
- `cuda/interface/fv3_gpu_field_registry.F90` /
  `fv3_gpu_dyncore_bridge.F90` / `fv3_gpu_abi_mod.F90` — ISO_C_BINDING
  bridge (`fv3_gpu_interface_state_to_fv_phase_a`/`_phase_c`), the
  selector block (§1), and the `DUMP_EXIT` diagnostic.
- Public C ABI (`fv3_gpu_api.h`): `fv3_gpu_state_to_fv_phase_a`/`_phase_c`
  entry points, same lifecycle convention as every other production
  dispatch call (persistent context, no per-call create/destroy).

Change budget respected: Phase A/C CUDA kernels + runtime dispatch +
Fortran selector/bridge + tests/docs only. `mpp_get_boundary` itself: 0
lines changed. No CUDA-aware MPI. No whole-model residency redesign. No
R4. No commit/push.

## 3. Stage 9 — GPU fixture validation

Host-reference harness (P98,
`cuda/dyn_core/tests/host/test_state_to_fv_host_reference.cpp`, 8/8
PASS) extended with a real-device GPU smoke test,
`cuda/dyn_core/vertical/fv3_state_to_fv_prod_shape_smoke.cu`, exercising
the actual `launch_state_to_fv_phase_a`/`launch_state_to_fv_phase_c`
kernels via `cudaMalloc`+`cudaMemcpy` round trips and comparing every
output field (`u`,`v`,`w`,`pe`,`peln`,`pk`,`delp`,`pt`,`delz`,`pkz`,`ps`)
bitwise against the same host-reference formulas used by the P98
harness, across the same 8 fixture categories (smooth field, varying
vertical profile, horizontal gradient, boundary-columns interior-only
check, active nonhydrostatic uvw, strong pt/delz/pkz contrast,
production-shaped reduced dims, mandatory sentinel-layout transposition
check).

**Result: 8/8 PASS**, bitwise, 0 NaN/Inf, on a real A100 (`srun`
GPU-node execution per `discover_platform.md`).

## 4. Stage 10/11 — Halo interoperability and invariants

Verified via the live C32 dispatch run (job 58628036, CUDA arm; job
58628037, Fortran control arm; `p99_c32_cuda`/`p99_c32_fortran`,
`RUN_CUDA_FV3_STATE_TO_FV_DUMP_EXIT=1` on both):

- `u`, `v`, `ps` (all genuinely D2H'd/host-fresh by design — see §2) at
  `[P99_EXIT_DUMP]` sample points match the Fortran control arm at
  round-off (last-ULP differences only) across all 7 dispatch calls per
  rank, for all 6 ranks. Since `fv_dynamics()` is highly sensitive to any
  wrong pressure/thermodynamic input and consumes Phase C's
  device-resident `pe`/`pt`/`delz`/`pkz` output directly, this indirectly
  confirms Phase C's on-device compute is correct (the same information
  the P98 GPU-smoke harness confirms directly).
- Both jobs completed cleanly (`sacct`: `COMPLETED`, `ExitCode 0:0` for
  both), no FATAL/abort/NaN/Inf in either log.
- Dispatch pattern per rank: CUDA arm `F T T T T T T` (1 `F`, 6 `T`);
  Fortran arm `F F F F F F F` (all `F`, escape hatch working correctly).

**Diagnostic-artifact finding (documented here to preempt future
confusion from re-running the same diagnostic):** the `[P99_EXIT_DUMP]`
`pkz_sample` (and likewise `pe`/`pt`/`delz` samples, though only `pkz`
was directly compared) reads the *host* Fortran array
`FV_Atm(1)%pkz`/etc. directly. Because Phase C intentionally leaves
`pe`/`peln`/`pk`/`delp`/`pt`/`delz`/`pkz` device-resident (§0/§2 — by
design, to avoid a redundant transfer before `fv_dynamics()`), the
CUDA arm's `pkz_sample` reads **stale host memory**, not Phase C's
actual computed value, and will show a divergent, slowly-drifting
pattern relative to the Fortran arm's genuinely-fresh value. This was
confirmed by direct source read of
`fv3_gpu_runtime.cpp`'s `fv3_gpu_state_to_fv_phase_c` (only `ps` is
D2H'd after the Phase C kernel launch — no other field is copied back)
and corroborated by the Stage 9 GPU-smoke harness (§3), which *does*
explicitly D2H and bitwise-check `pkz` and passes. **This is not a
kernel defect** — it is a limitation of a diagnostic that reads a host
array the design intentionally does not keep in sync. Any future
`DUMP_EXIT`-based debugging of this routine must read `pkz`/`pe`/`pt`/
`delz` from a dedicated on-device readback, not the host Fortran array,
if a genuine on-device value is needed.

The cold-start `F` fallback (exactly 1 per rank, always the very first
call) is likewise not a defect — see §1's line-1186/1264 ordering
explanation. Excluding that architecturally-unreachable call, dispatch
correctness is 36/36 (6/6 per rank) among calls where CUDA dispatch is
even possible.

## 5. Stage 12 — compute-sanitizer

Not separately re-run in this task: Stage 14 (§7) found a decisive C512
device-memory blocker before a dedicated sanitizer pass added any
value (bounded-debugging convention — stop once the evidence is
sufficient to decide). The closest available evidence remains the
Stage 9 GPU smoke test (§3): real-device `cudaMalloc`/kernel-launch
round trips, bitwise-correct against the host reference, 8/8 PASS, no
observed memory errors. A dedicated `compute-sanitizer` pass on the
smoke test binary (cheap, single-GPU, no C512 density) remains a
reasonable follow-up if this port is revisited, but is not required to
reach the Stage 14/§10 decision below.

## 6. Stage 13 — C32 integration correctness

Covered by §4 above (jobs 58628036/58628037): CUDA arm dispatch correct
(36/36 possible calls dispatch CUDA, 1/42 architecturally-expected
cold-start fallback per rank), Fortran control arm correct (42/42
Fortran, escape hatch verified), both jobs `COMPLETED`/`ExitCode 0:0`,
no FATAL/NaN/Inf, field agreement at round-off for all host-crossing
fields.

## 7. Stage 14 — C512 integration correctness: DEVICE OUT-OF-MEMORY BLOCKER

Job 58628363 (`p99_state_to_fv_ab5.sbatch`, same node `warpa003`, MPS
ON, 6 ranks/1 A100, authoritative C512/L10 production config, same
methodology as P92/P89), `sacct`: job and all 10 srun steps
`COMPLETED`, `ExitCode 0:0`, total elapsed 00:14:54 (well inside the
01:10:00 request; each run's own `rc` was captured by the wrapper, not
by `sacct`, which only reflects the wrapper shell exiting cleanly).

**Every Arm-A (Fortran baseline) run succeeded**, all 5 pairs:

| Run | DYN_DATE_COUNT | Elapsed |
|---|---|---|
| pair1_A | 30 | 00:02:16 |
| pair2_A | 30 | 00:02:15 |
| pair3_A | 30 | 00:02:15 |
| pair4_A | 30 | 00:02:14 |
| pair5_A | 30 | 00:02:15 |

**Every Arm-B (CUDA State_To_FV) run FAILED, all 5 pairs**, all with
the identical FATAL, all ranks, always within the first ~37-38s
(before the first `DYN Date` print — `DYN_DATE_COUNT=0` for all 5):

```
FATAL from PE <n>: CUDA dyn_core backend: fv3_gpu_fv_tp_2d_hord8 failed: out of memory
Abort(1) ... application called MPI_Abort(MPI_COMM_WORLD, 1) - process <n>
```

**Root cause, established from GPU memory telemetry, not assumed:**
the `nvidia-smi` polling loop shows the **Fortran-only baseline arm
alone** (Arm A, with every other production-default CUDA path already
active — REMAP_UV/PT_DELZ/REMAP_Q/REMAP_PRESSUREVARS) drives the
shared A100 to **40265 MiB used / 40334 MiB free of 40960 MiB total —
only ~69-73 MiB of headroom** at this density (6 ranks/1 GPU under
MPS, one full 512x512x10 tile per rank). This device is already
memory-saturated by existing production CUDA state *before* this
port's new buffers are even added. Adding Phase A/C's new persistent,
grow-only, per-rank `StateInputView` buffers (`state_u`, `state_v`,
`state_w`, `state_pt`, `state_pkz`, `state_dz`, `state_pe`, plus the
`ps` scratch buffer — per-rank estimate ≈151MB × 6 independent
per-rank CUDA contexts ≈ 0.9GB aggregate, per the Stage 9 (P98)
data-volume estimate) has nowhere to go. The FATAL's specific call
site (`fv3_gpu_fv_tp_2d_hord8`'s allocator) is most plausibly just the
first allocation call that happens to trip the now-exhausted budget in
program order — **not** evidence that the hord==8 CUDA tracer path
executes in production (it does not; production uses
`hord_mt/vt/tm=6`, and P83 already closed CUDA tracer production
optimization as a separate, unrelated matter). The aggregate telemetry
alone (69MiB free before this port's ~0.9GB new demand) is sufficient
to explain the failure regardless of exactly which allocation call
trips first.

This is a genuine **capacity/validation blocker at C512 production
density**, not a kernel-correctness defect: Stage 9's GPU smoke test
(§3) and Stage 13's C32 integration (§4/§6) both confirm the kernels
themselves are bitwise-correct on real hardware at lower density. The
failure is specific to the combination of (a) one full tile per rank,
6 ranks sharing 1 A100 via MPS, and (b) the cumulative persistent
device-memory footprint of every already-shipped production CUDA path
plus this port's new buffers.

## 8. Stage 15 — Same-node paired performance (5 pairs): NOT MEASURABLE

No CUDA-arm run completed a single `DYN_CORE`/`State_To_FV` cycle
before crashing (§7) — there is no CUDA-arm timer data of any kind to
pair against the Fortran baseline. The Fortran-only baseline itself
was stable and internally consistent across all 5 repeats (MAPL
Profiler report, mean-per-DYN_CORE-call basis): `TOTAL` 132.06-133.64
(range ~1.2%), `DYN_CORE` 25.52-25.58, `FV_DYN_LOOP` 48.36-49.96,
`TRACER_2D` 12.84-13.94, `REMAPPING` 11.96-13.28,
`REMAP_TOTALENERGYCON` 3.99-5.35 — consistent run-to-run, confirming
the baseline harness itself is not the source of the failure. (Note:
these absolute magnitudes are not directly compared against Prompt
94's C512 figures in CURRENT_STATE.md §5 — reconciling the two would
require confirming both drew from the same profiler report section/
call-count basis, which is moot here since no CUDA-arm number exists
to pair against either way.)

**No performance gate classification is possible.** The Stage
15/16 objective (measure a CUDA-vs-Fortran `TOTAL` delta) cannot be
executed until Stage 14's device-memory blocker is resolved by a
separate, explicitly-authorized follow-on (see §11).

## 9. Stage 16 — Attribution reconciliation

Moot — there is no CUDA-arm timer data to reconcile against a
predicted-net estimate (§8).

## 10. Performance gate classification and decision

Gate not applicable — no CUDA-arm C512 timing data exists to classify
(§8). **Decision: VALIDATION BLOCKED**, not KEEP / KEEP AS
INFRASTRUCTURE / DISABLE (those all presuppose a completed run).
Production default remains **Fortran**, unchanged, consistent with
every prior stage of this task and the task's binding constraint that
CUDA must not be default-enabled regardless of outcome.

## 11. Residency follow-up question

**Is there a broader persistent-residency opportunity here? Answer:
NO new opportunity is exposed by this port's own design — but YES, a
broader capacity constraint is exposed that any future port at this
density must budget against.** Phase A/C's device-residency choice
(§0/§2 — leaving `pe`/`peln`/`pk`/`delp`/`pt`/`delz`/`pkz`
device-resident, feeding `fv_dynamics()` directly) was and remains the
architecturally correct design; it does not, by itself, change the
model's aggregate device-memory footprint versus a hypothetical
host-round-trip design; the ~0.9GB new persistent-buffer demand is an
inherent cost of doing this compute on the GPU at all, not a) defect
in the residency choice, and b) not reducible without shrinking what
the port actually needs to hold. The real, generalizable finding is
that **the C512/6-ranks-per-GPU density (already flagged in
CURRENT_STATE.md §7 as memory-tight for CUDA-aware MPI) has, as of
this task, only ~69-73 MiB of headroom left on a 40GB A100 given the
sum of every already-shipped production-default CUDA path's own
persistent buffers.** Any future CUDA port considered for this density
— not just `State_To_FV` — should budget its expected persistent
device-memory footprint against this near-zero headroom *before*
implementation, and a per-port memory audit across the already-shipped
production defaults (REMAP_UV, PT/DELZ, REMAP_Q, REMAP_PRESSUREVARS)
may itself be a worthwhile, low-risk follow-on to quantify exactly how
much of the 40GB budget each already consumes. Reducing ranks-per-GPU
(2 GPUs/3-ranks-per-GPU is already validated correct for the separate
CUDA-aware-MPI transport, CURRENT_STATE.md §5) is a plausible
precondition for reviving this port, rather than shrinking
`State_To_FV`'s own buffer set, which is already minimal (Stage 3 of
P98: only the fields with no existing FieldRegistry identity get new
buffers; everything else reuses existing storage). No implementation
of any of this is performed here — answer only, per task scope.

## Summary

- IMPLEMENTED: YES (Phase A + Phase C CUDA kernels, selector, bridge)
- COMPILED: YES
- GPU EXECUTED: YES (Stage 9 smoke test; Stage 13 C32 integration)
- NUMERICALLY VALIDATED: YES at C32 (Stage 9 8/8 bitwise; Stage 13 C32
  round-off agreement). **NOT validated at C512** — every C512 CUDA-arm
  run crashed with a device out-of-memory FATAL before completing a
  single cycle (§7).
- PERFORMANCE VALIDATED: NO — not measurable; no CUDA-arm C512 timing
  data exists (§8).
- Production default: FORTRAN (unchanged)
- No commit/push: YES

## Final verdict

**STATE_TO_FV CUDA VALIDATION BLOCKED — device out-of-memory at C512
production density (6 ranks/1 A100 under MPS): the Fortran-only
baseline (with every other already-shipped production CUDA path
active) already consumes 40265/40334 MiB of the 40GB A100, leaving
only ~69-73 MiB headroom before this port's own new persistent Phase
A/C buffers (~0.9GB aggregate across 6 per-rank contexts) are even
added; every one of 5 CUDA-arm C512 runs aborted via FATAL/MPI_Abort
within ~37-38s, before a single DYN_CORE cycle completed, so no
C512 numerical validation or performance comparison is possible in
this task. Kernel correctness at lower density (C32 integration,
Stage 9 GPU smoke test) is unaffected — this is a device-memory
capacity blocker at C512 density, not a kernel defect.**

**Update 2026-09-29 (Prompt 100):** a dedicated device-memory
accounting/buffer-reuse study confirmed this port's own ~906 MB
aggregate addition is fully recoverable in principle (safe temporal
alias into the existing, lifetime-disjoint, ~6x-larger
`WorkspaceId::vertical` arena — no duplicate storage, no numerical
change), but the pre-existing baseline headroom (~69-73 MiB, without
this port at all) is already ~440 MiB short of the required >=512 MiB
safety margin on its own. **No `State_To_FV`-scoped fix — including
the best one identified — can reach the required target**, so none
was implemented. **Decision: STATE_TO_FV STILL MEMORY-BLOCKED —
BROADER DEVICE-RESIDENCY ARCHITECTURE REQUIRED**, not another
isolated kernel port. See
`docs/performance/p100_state_to_fv_device_memory_reuse_2026-09-29.md`.

**Update 2026-09-29 (Prompt 101):** the broader per-port/arena memory
audit Prompt 100 called for was executed. Static accounting of every
named production allocation (FieldRegistry, `WorkspaceLayout`, REMAP_Q/
REMAP_PRESSUREVARS buffers; PT/DELZ and REMAP_UV confirmed to add zero
dedicated buffers) explains only ≈46.6% of the live-measured converged
C512 footprint — a ≈21.5 GiB gap remains, traced to the first
production dispatch call but not attributed among 4 candidate causes.
**Decision: RECONCILIATION INCOMPLETE — no memory fix selected.**
`State_To_FV` remains CLOSED — STILL MEMORY-BLOCKED. See
`docs/performance/p101_production_gpu_memory_inventory_2026-09-29.md`.

**Update 2026-09-29 (Prompt 102/103):** the per-rank/per-context memory
architecture study (P102) and the executed 2-GPU C512 validation (P103)
together resolved the memory-blocked status at 1-GPU/6-ranks density by
showing 2-GPU/3-ranks-per-GPU has ~19.7 GiB/GPU free — ~40x this port's
~440 MiB shortfall. `State_To_FV` memory feasibility **CONFIRMED at
2-GPU density**.

**Update 2026-09-29 (Prompt 104):** enabled and measured `State_To_FV`
CUDA at 2-GPU/3-ranks-per-GPU. CUDA dispatch of both phases CONFIRMED
(indirect converging evidence — branch selection, ABI success, +440.4
MiB/GPU device-memory footprint matching P100's prediction, and a
distinct/reproducible timing signature; no raw dispatch counter was
implemented). Direct speedup **1.633x mean / 38.75% reduction** (3/3
paired A/B runs, stddev 0.006x — highly reproducible). TOTAL end-to-end
+1.01% mean (MEANINGFUL, boundary). **No field-level correctness
comparison was possible at C512/2-GPU** (the short harness produced no
HISTORY output) — this port's only field-level validation remains the
C32 bitwise result above, now with an open gap at C512/2-GPU. **Decision:
KEEP OPT-IN** (`RUN_CUDA_FV3_STATE_TO_FV_ENABLE` unchanged, still
opt-in) pending a follow-up correctness pair with HISTORY output
configured. No source/production-default change. See
`docs/performance/p104_state_to_fv_2gpu_validation_performance_2026-09-29.md`.

**Update 2026-09-29 (Prompt 105):** closed the field-level correctness
gap. The Prompt 104 correctness pair's HISTORY output existed all
along — Prompt 104's check inspected only the top-level rundir and
missed the actual `scratch_*/` execution subdirectory where MAPL
History writes its `.nc4` files; no new job was needed. Comparison via
`hs_netcdf_inspect.py compare-backends` at C512/2-GPU/3-ranks-per-GPU:
**T and PS BITWISE identical**, **U and V ROUND-OFF CLEAN** (max_rel
1e-8 to 1.7e-6, no NaN divergence, no boundary-clustered
halo/layout-pattern discrepancy) across every pressure level checked.
NUMERICALLY VALIDATED at C512/2-GPU is now YES. Combined with Prompt
104's already-established GPU EXECUTED and PERFORMANCE VALIDATED
(1.633x direct, TOTAL +1.01%) status at this density, all three legs of
the validation hierarchy are now satisfied. **Decision: READY FOR
DEFAULT ENABLE** — recommended as a separate, dedicated follow-up task;
default not changed in Prompt 105. See
`docs/performance/p105_state_to_fv_c512_2gpu_correctness_2026-09-29.md`.
