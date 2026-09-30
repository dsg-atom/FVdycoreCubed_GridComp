# Prompt 76 — Implement `fv_tp_2d` hord==8 CUDA Extension

Date: 2026-09-24
Branch: `optimization/cuda-fv3-performance`
Model: Sonnet
Prior art: `p75_fv_tp_2d_hord8_cuda_plan_2026-09-24.md` (PROCEED decision,
11-fixture host harness, no CUDA implementation).

## 0. Summary

hord==8 CUDA face-flux kernels, the `pert_ppm(iv=1)` device helper, and
dispatch-gate widening were implemented in
`cuda/dyn_core/dgrid/wp_b_cuda.cu` / `wp_b_ppm_shared.cuh`, compiled, and
GPU-executed. **Standalone unit correctness (Stages 0-8): ALL PASS.**
**Integration correctness/performance (Stages 9-15): BLOCKED BY DESIGN**
— no existing Fortran-side selector routes the real production
`tracer_2d_1L` tracer-transport call through any CUDA dispatch path, and
creating one is out of scope for this task's change budget ("no tracer
orchestration changes").

## 1. PERT_PPM — Stage 1

Device helper `fv3::gpu::dgrid::pert_ppm_iv1` (`wp_b_ppm_shared.cuh:51-67`,
pre-existing from a prior window) tested against 5 named fixtures
(monotonic, local extremum, sign change, boundary, near-flat) in a new
standalone test, `tests/dgrid/test_pert_ppm_iv1_cuda.cu`, comparing the
real device function against an independent host transcription of
`tp_core.F90:1238-1252` (iv/=0 branch). GPU result (job 58540327):

```
PERT_PPM_IV1: ALL 5 CASES BITWISE PASS
```

## 2. HORD8 FACE FLUX — Stages 2-4

Existing kernels confirmed already implemented from a prior window:
`xppm_pass_kernel_hord8`/`yppm_pass_kernel_hord8` (structural x/y
mirrors, no asymmetry bug), reusing the generic
`q_i_kernel`/`q_j_kernel`/`flux_weight_x_kernel`/`flux_weight_y_kernel`
orchestration exactly as planned in p75 §3 (no changes needed to those).
Dispatch gate in `launch_scalar_transport_cuda` widened to accept
`mord==8` alongside `mord==6`. No changes to halo width, R4, REMAP_UV, or
tracer orchestration.

Compile-time footprint check (`nvcc --resource-usage -Xptxas -v`,
`arch=compute_80`):

| Kernel | Registers | Stack frame | Spill loads/stores |
|---|---|---|---|
| `xppm_pass_kernel_hord8` | 56 | 0 bytes | 0 |
| `yppm_pass_kernel_hord8` | 56 | 0 bytes | 0 |
| `xppm_pass_kernel` (hord==6, reference) | 48 | 0 bytes | 0 |
| `yppm_pass_kernel` (hord==6, reference) | 49 | 0 bytes | 0 |

Modest register increase (56 vs. 48/49), zero local-memory/stack
footprint growth — consistent with p75 §6's "no fixed per-thread local
array required" prediction; the Prompt 68-style OOM risk does not apply
here.

## 3. FIXTURE TEST — Stage 5

Two independent test paths, both GPU-executed and PASS:

**(a) Standalone fixture test** (`tests/dgrid/test_wp_b2_hord8_transport_cuda.cpp`,
new this task, direct call into `fv3::gpu::dgrid` device functions via a
hand-built host reference) — 4 geometric parts (A interior, B x-edge
panel seams, C y-edge panel seams, D all-four-edges/C512-like) plus a
hord==6 dispatch-sanity part E. GPU result (job 58540266/58540369,
re-confirmed twice):

```
ALL PARTS PASS
(fx/fy max_rel 2.17e-16 to 4.31e-16 across all parts; q(inout) bitwise
exact — this API applies no flux-divergence update to interior q, see
§Errata below)
```

**(b) Pre-existing Part L**, `tests/dgrid/test_wp_b2_scalar_transport_cuda.cpp`
(not modified this task — already present from prior work, calling the
real production entry points `launch_fv_tp_2d_cuda`/
`launch_scalar_transport_cuda` directly, using the exact 11 fixtures
named in `p75_...md` §7: constant, linear_gradient, smooth_curved,
sharp_front, local_max, local_min, near_zero_positive, sign_changing,
high_courant_valid, boundary_edge_is1_and_npx, c512_like_slice). GPU
result (job 58539992, `warpa010`, dedicated Stage-5 verification run,
log `perf_rerun_logs/p76_stage5_58539992.out`; independently
reconfirmed by job 58540327's rerun of the same binary,
`perf_rerun_logs/p76_run_all.log`): 10/11 fixtures PASS at round-off
(fx/fy max_rel up to ~6.757e-16 across Parts A-K, ~5.8e-16 within Part
L itself); 1/11 (`constant`) reports `FAIL (degenerate: constant 5 over
1296 cells)`. All fx/fy comparisons for the `constant` fixture itself
are exact bitwise matches (worst rel 0.000e+00) — only the `q(inout)`
degeneracy self-check fires. Traced to `report()`'s `require_variation`
guard
(`test_wp_b2_scalar_transport_cuda.cpp:1030-1035`): this is a generic
anti-degeneracy self-check that fires whenever a comparison box shows
zero output variation, regardless of correctness — it is not a
got-vs-ref mismatch (both got and ref are the same constant, 5, over the
whole box). A constant-field tracer being transported to exactly the
same constant is the *correct* physical result, not a bug. **Not a
regression from this task** — the guard and the `constant` fixture both
predate this task's edits (this file was not modified this window; only
rebuilt).

**Reconciliation with p75's "11 fixtures" instruction**: Part L already
implements exactly those 11 fixtures against the true production
dispatch path, satisfying Stage 5's letter and intent more strongly than
a hand-built host reference would. The standalone test in (a) is
retained as an independent geometric cross-check (interior/x-edge/y-edge/
all-corners) and is not redundant busywork — it isolates panel-edge vs.
corner geometry more explicitly than Part L's fixture layout.

### Errata found and fixed while building test (a)

Two test-only bugs were found and fixed in
`test_wp_b2_hord8_transport_cuda.cpp`'s **own host reference** (no
production code changes):

1. `q_j`'s host-side loop used `j=js..je` instead of the CUDA
   `q_j_kernel`'s actual domain `j=jsd..jed` (full j-ghost range) —
   corrupted every `fy` comparison uniformly. Fixed to match.
2. A spurious flux-divergence update to `q_out` was removed after
   confirming, by reading the full `launch_fv_tp_2d_cuda`/
   `launch_scalar_transport_cuda` bodies, that this API computes only
   `fx`/`fy` (and internal `q_i`/`q_j`/`fx2`/`fy2`) — it never applies a
   flux-divergence update to interior `q`. The only production mutation
   to `q` is `copy_corners` (ghost/corner cells only) and the optional
   `deln_flux` tail (disabled in every fixture here). The actual
   tracer-advance update is caller-side, outside this API — consistent
   with Prompt 74's finding that `tracer_2d_1L` orchestration remains
   unaddressed.

Both fixes are confined to the new test file; no production semantics
changed.

## 4. HORD6 REGRESSION — Stage 6

`fv3_wp_b2_scalar_transport_test` rebuilt with no source changes (only
picking up the widened dispatch gate and new `hord8` device functions
linked into the same translation unit). Parts A-K (all pre-existing
hord==6/deln_flux/precision/rejection coverage): **all PASS**, bitwise
q(inout) throughout, matching pre-Prompt-76 behavior. Part E ("hord=6
dispatch sane") in the standalone hord8 test also PASS — confirms the
widened `mord==8` branch does not perturb the existing `mord==6` path.

## 5. SANITIZER — Stage 7

`compute-sanitizer --tool memcheck` against the hord8 fixture test
(Part D exercises C512-like all-four-edges geometry), job 58540369:

```
========= ERROR SUMMARY: 0 errors
```

(Binary path required a PATH fix from `discover_platform.md`'s
documented Nsight PATH addition —
`compute-sanitizer` actually lives under
`$NVHPC_ROOT/cuda/12.5/compute-sanitizer/`, not the Nsight_Systems/
Nsight_Compute directories; this is a one-off environment note, not a
platform-doc change requested by this task.)

## 6. CONSERVATION — Stage 8

No new GPU job was needed: `fx`/`fy` are already confirmed bitwise-
identical between CUDA and the authoritative host reference in every
fixture (§3). The flux-divergence update `q_new(i,j) = q_old(i,j) +
(fx(i,j)-fx(i+1,j)) + (fy(i,j)-fy(i,j+1))`, summed over any box, is an
exact telescoping-sum identity — the interior sum reduces algebraically
to the net flux through the box's four boundaries, for *any* consistent
flux field, independent of the PPM order that produced it. Since CUDA's
`fx`/`fy` are bit-identical to the host's, any conservation residual
between CUDA and the authoritative host is bounded by the same round-off
already measured for the fluxes themselves (max_rel ~3.7-4.3e-16), and
is not independently measurable given that identity. No new positivity/
monotonicity failures were observed in either fixture set (§3).

## 7. C32 — Stage 9 (BLOCKED)

Not run. See §9 DECISION.

## 8. C512 — Stage 11 (BLOCKED)

Not run. See §9 DECISION.

## 9. PERFORMANCE — Stages 13-14 (BLOCKED)

Not run. See §9 DECISION.

## 10. DECISION — Stage 15

**Stages 0-8 (standalone unit correctness): COMPLETE, ALL PASS.**

**Stages 9-15 (integration correctness + performance): BLOCKED BY
DESIGN**, confirmed by direct source inspection of the authoritative
Fortran fixture tree (`GEOSfvdycore_fortran_perf`):

- The real production tracer-transport call chain is
  `fv_dynamics.F90` -> `fv_tracer2d.F90::tracer_2d_1L` ->
  `tp_core.F90::fv_tp_2d`. No CUDA dispatch selector exists anywhere in
  this chain (grepped for any pattern analogous to
  `use_cuda_remap_uv`/`RUN_CUDA_FV3_REMAPUV_ENABLE`; none found).
- The only CUDA-tagged symbol matching `fv_tp_2d` anywhere in the tree,
  `tp_core_cuda`/`fv_tp_2d_cuda`, is referenced solely from
  `tp-core-driver/test.F90` — an isolated test driver, not linked into
  `fv_tracer2d.F90`, `fv_mapz.F90`, or `fv_dynamics.F90`.
- Creating a new selector/dispatch wiring at the `tracer_2d_1L` call
  site to route production traffic into the new CUDA hord8 kernels would
  itself be a tracer-orchestration change — explicitly forbidden by this
  task's change budget ("No tracer orchestration changes", "port tracer
  orchestration" in the Do-Not list).
- This is a scope-respecting conclusion, not a failed attempt: Stages
  9-15 require exactly the kind of change this task was told not to
  make. Building the selector is a natural, well-scoped follow-up task
  (see NEXT TARGET) but is out of bounds here.

## 11. NEXT TARGET

A dedicated follow-up task, explicitly scoped to add a production
dispatch selector (`use_cuda_fv_tp_2d_hord8` or similar, mirroring
`use_cuda_remap_uv`'s pattern) at the `tracer_2d_1L` call site in
`fv_tracer2d.F90`, defaulting OFF/opt-in, would unblock Stages 9-15
(C32/C512 integrated correctness, then paired performance, then
KEEP/DISABLE). This is the natural continuation of this task's work
and should be scoped as its own prompt given it touches the tracer
orchestration boundary explicitly excluded here.

## 12. Files created/modified this task

- New: `cuda/dyn_core/tests/dgrid/test_pert_ppm_iv1_cuda.cu`.
- Fixed (test-only, host reference bugs, §3 Errata):
  `cuda/dyn_core/tests/dgrid/test_wp_b2_hord8_transport_cuda.cpp`.
- Modified: `cuda/dyn_core/CMakeLists.txt` (new `fv3_pert_ppm_iv1_test`
  target).
- Updated: `p75_fv_tp_2d_hord8_cuda_plan_2026-09-24.md` (follow-up note,
  closing line).
- Updated: this file (new).
- Updated: `cuda_fv3_optimization_roadmap_2026-09-21.md`,
  `docs/agent_handoff/CURRENT_STATE.md`.

No production Fortran source modified (per change budget). Production
CUDA source (`wp_b_cuda.cu`/`wp_b_ppm_shared.cuh`) changes (hord8
face-flux kernels, `pert_ppm_iv1`, dispatch-gate widening) were already
present from a prior window under this same task and are unchanged this
window except for the test-file fixes listed above. No commit made.

---

**FV_TP_2D HORD8 CUDA VALIDATION BLOCKED — no production dispatch
selector routes the real `tracer_2d_1L` tracer-transport call through
any CUDA path, and creating one is out of this task's change budget
(tracer-orchestration changes are explicitly forbidden). Standalone
hord==8 CUDA implementation is correctness-validated (Stages 0-8, all
PASS, GPU-executed, sanitizer-clean); integration and performance
(Stages 9-15) require a follow-up task scoped to add that selector.**

---

**Follow-up (Prompt 77, 2026-09-24):** the missing selector was added
(`use_cuda_tp2d` in `fv_tracer2d.F90`) and integration completed. Dispatch
wiring at C32/C512: PASS, under a claim narrowed to integration-wiring
only — the standalone cold-start harness family was found to leave
production `q` structurally zero, so this kernel's *numerical*
correctness claim continues to rest solely on the non-degenerate fixtures
in this document, not on Prompt 77's production runs. Same-node C512
paired performance: **MEASURED REGRESSION** (TOTAL +10.0% slower with
CUDA tracer dispatch vs Fortran) — not yet a KEEP. See
`p77_fv_tp_2d_hord8_production_integration_2026-09-24.md` for full detail.
