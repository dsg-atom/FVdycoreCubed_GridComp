# Milestone 3A.11S — H_GZ_FIRST Ordering Correction

Branch: `cuda/gpu-bringup`
Base commit: `78c07e5` (`78c07e5b5b999bc2809e23e32e8a179d73e5114e`, "Record milestone 3A.10 commit SHA in handoff") (working tree uncommitted)
Authoritative reference: `GEOS-ESM/GFDL_atmos_cubed_sphere`, tag `geos/v3.0.0`,
SHA `ca30e26305fda43831095334c02a3f5e0b8c0c45` — verified present on disk at
`/discover/nobackup/projects/imvi-nse/jli30/fv3_reference/GFDL_atmos_cubed_sphere`
(`git log -1` confirms the exact SHA and tag).

## Headline

The H_GZ_FIRST ordering defect is **FIXED and verified against the authoritative
tagged source**. It caused a real, deterministic corruption and its removal
measurably improved repeated-step stability.

It was **not the only remaining defect**. Repeated execution now survives longer
but still fails, and the residual failure has been isolated to a **third,
distinct** defect in a different component. The performance benchmark remains
BLOCKED.

Failure progression across the three fixes:

| State | First failing call |
|---|---|
| 3A.11R start | call #2 |
| + benchmark `delz` fixture fix (3A.11R) | call #4 |
| + H_GZ_FIRST ordering fix (this milestone) | **call #7** |

## 1. Why the race hypothesis was rejected

Established in 3A.11R and unchanged: three independent runs bitwise identical
across 5400 checkpoints; `racecheck` 0 hazards; maximal per-stage
synchronization did not remove the NaN; corrupted cells had a fixed spatial
signature identical on all six tiles. The apparent nondeterminism was an
artifact of `debug_check_finite_stage` syncing only `compute_stream` and then
issuing a blocking `cudaMemcpy` on the legacy null stream, which is unordered
with respect to the non-blocking `transfer_stream` and so sampled fields
mid-unpack. See `milestone_3a11r_halo_sync_race.md`.

## 2. Authoritative Fortran ordering (§1)

From `model/dyn_core.F90` at the tagged SHA. The exchange is **split-phase**:

| # | Line | Operation | Routine | Inputs | Outputs | Halo dep | Next consumer |
|---|---|---|---|---|---|---|---|
| 1 | :427 | start w exchange | `start_group_halo_update` i_pack(7) | `w` | — | — | :507 |
| 2 | :433 | `if (it==1)` gate | — | — | — | — | — |
| 3 | :445-455 | **gz reconstruction**: `gz(npz+1)=zs`; `gz(k)=gz(k+1)-delz(k)` over **is:ie, js:je (INTERIOR ONLY)** | inline | `delz`, `zs` | `gz` interior | — | step 4 |
| 4 | **:459** | **START** gz exchange | `start_group_halo_update` i_pack(5) | `gz` **interior (just rebuilt)** | in-flight | — | :552 |
| 5 | :504 | complete u,v | i_pack(8) | — | `u`,`v` halo | H_DWIND_IN | :516 |
| 6 | :507 | complete w | i_pack(7) | — | `w` halo | H_W_IN | :516 |
| 7 | :516 | C-grid predictor | `c_sw` | — | `ut`,`vt`,`delpc`,`ptc` | — | :577 |
| 8 | :528 | start divgd | i_pack(3) | `divgd` | — | (nord>0) | :636 |
| 9 | **:552** | **COMPLETE** gz exchange, `if (it==1)` | `complete_group_halo_update` i_pack(5) | — | **`gz` halo** | — | step 10 |
| 10 | :557-563 | `zh(i,j,k) = gz(i,j,k)` over **isd:ied, jsd:jed (FULL HALO BOX)**, it==1 | inline | `gz` incl. halo | `zh` full box | **needs #9** | :577, `update_dz_d` |
| 11 | :566-573 | `gz = zh` over full halo box, it>1 | inline | `zh` | `gz` | — | :577 |
| 12 | :577 | `update_dz_c` | `nh_utils` | `ut`,`vt`,`gz`,`dp_ref`,`zs`,`area` | `gz`,`ws3` | — | :583 |
| 13 | :583 | `Riem_Solver_C` | `nh_utils` | `delpc`,`ptc`,`gz`,`ws3`,`q_con`,`cappa` | `gz`,`pkc` | — | :614 `p_grad_c` |

Answers to the §1 questions:

- **Where gz interior becomes authoritative:** step 3 (:445-455), *before* the
  exchange starts.
- **What H_GZ_FIRST exchanges:** the `gz` array's halo, packed from the
  freshly reconstructed interior.
- **What must precede H_GZ_FIRST:** the gz reconstruction from `delz`/`zs`.
- **What must follow it:** `zh <- gz` over the full halo box (:557-563), then
  `update_dz_c`, then `Riem_Solver_C`.
- **Does zh participate in the split:** YES, decisively. `zh`'s halo is seeded
  entirely from `gz`'s exchanged halo at :557-563, and `zh` is later consumed by
  `update_dz_d`. `pkc` and `delz` do not participate in this split (`pkc` is a
  `Riem_Solver_C` output; `delz` is an input to the reconstruction).
- **Frequency (§7):** `it == 1` **only** — the gate at :433 covers both the
  reconstruction and the START, and :552's COMPLETE is independently gated
  `if (it==1)`. **The existing `it==1` behavior is source-confirmed correct and
  was preserved.**

Note the split-phase START/COMPLETE straddles `c_sw`, so in Fortran the transfer
overlaps the predictor. The CUDA runtime's `resolve_halo` is a single blocking
pack/transfer/unpack rather than a split start/complete, so the ordering
*constraint* is reproduced exactly while the *overlap* is not. That is a
performance property, not a correctness one, and is out of scope here (§4).

## 3. Previous incorrect CUDA ordering (§2)

`fv3_gpu_runtime.cpp` substep loop, through 3A.11R:

```
resolve(H_GZ_FIRST)            <-- packs gz interior from the PREVIOUS step
run(cgrid_vertical_solver)     <-- gz_from_delz_kernel rebuilds it INSIDE here
```

Mapping:

| CUDA operation | Authoritative op | Previous position | Required position |
|---|---|---|---|
| `gz_from_delz_kernel` | :445-455 | inside solver, **after** resolve | **before** resolve |
| `require(H_GZ_FIRST)` | :552 | inside solver (already correct) | unchanged |
| `gz_zh_sync_kernel` | :557-573 | inside solver, after require | unchanged |
| `update_dz_c` | :577 | inside solver | unchanged |
| `Riem_Solver_C` | :583 | inside solver | unchanged |

The numerical stage was **already correctly structured internally** — its
`require(FV3_GPU_H_GZ_FIRST)` sat at exactly the right point, between the
reconstruction and the gz/zh sync, with accurate documentation. The defect was
entirely in runtime orchestration: because the runtime pre-resolved the group
before invoking the opaque stage, `require()` found it "ready" while the data it
carried predated the reconstruction.

Consequence: every tile's `gz` halo was one top-level step stale. :557-563 then
broadcast that stale halo into `zh` across the full halo box, where it persisted
for all later substeps, and `update_dz_c` differenced `gz2` across the resulting
interior/halo discontinuity until its denominator
`(area + xfx_diff + yfx_diff)` crossed zero.

## 4. The false runtime comment (§6) — CORRECTED

`fv3_gpu_runtime.cpp` previously justified the early resolve with:

> "On the only topology resolve_halo will accept, the exchange copies nothing,
> so resolving before rather than inside the stage skips no operation."

**False.** On the real six-tile `grid_type == 0` cubed sphere all four edge
neighbours are populated and the exchange performs real device-to-device data
movement. The claim held only for the degenerate single-tile topology earlier
tests used. Replaced with a description of the real topology, the actual failure
mechanism, and a pointer to this document. The HALO STATE TRANSITION TABLE entry
for H_GZ_FIRST was updated to record the new producer stage.

No other source comment or handoff document was found asserting the same
"copies nothing" claim.

## 5. Stage split design (§3) — IMPLEMENTED

```
cgrid_vertical_pre_halo   ->   resolve(H_GZ_FIRST)   ->   cgrid_vertical_solver
   (gz_from_delz,                (real exchange)            (require, zh<-gz,
    it==1 only)                                              update_dz_c,
                                                             Riem_Solver_C)
```

Implementation, minimal and additive:

| File | Change | Numerical algorithm changed |
|---|---|---|
| `vertical/fv3_vertical_kernels.cuh` | Declare `launch_cgrid_vertical_pre_halo`; document the split and why it is required | **NO** |
| `vertical/fv3_vertical_kernels.cu` | Factor the config fence into `cgrid_vertical_config_fence` (verbatim predicates); add `launch_cgrid_vertical_pre_halo` containing the *relocated, unmodified* `gz_from_delz_kernel` launch; remove that launch from `launch_cgrid_vertical_solver`, **keeping** its `require(H_GZ_FIRST)` | **NO** |
| `include/fv3_gpu_internal.hpp` | Additive `LaunchFn cgrid_vertical_pre_halo` slot on `NumericalDispatch` | **NO** |
| `runtime/fv3_gpu_production_dispatch.cu` | Wire the new slot | **NO** |
| `runtime/fv3_gpu_runtime.cpp` | Run `cgrid_vertical_pre_halo` **before** `resolve(H_GZ_FIRST)`; correct the false comment and the state table | **NO** |

No numerical work is duplicated. `gz_from_delz_kernel` is not re-run after the
exchange. `gz_from_delz_kernel` and `gz_zh_sync_kernel` bodies are untouched.
Readiness still arises from the real exchange — no `cudaDeviceSynchronize`, no
host-wide barrier, no manual `mark_ready`, no fake readiness (§5).

The retained `require(H_GZ_FIRST)` inside the solver is deliberate: a dispatch
table that omits the pre-halo slot now fails loudly instead of silently
exchanging stale gz. Both the benchmark and `test_dispatch_smoke` obtain their
table from `fv3_build_production_dispatch`, so the new slot propagates
automatically.

## 6. H_GZ_FIRST state transition table (§7)

| Stage | State before | Action | State after |
|---|---|---|---|
| substep entry, it==1 | unknown | — | unknown |
| `cgrid_vertical_pre_halo` | unknown | `gz_from_delz_kernel` writes gz interior on `compute_stream` | unknown (producer complete) |
| `resolve_halo(H_GZ_FIRST)` | unknown | `compute_stream` --(`event_compute_done`)--> `transfer_stream`; pack; BARRIER 1; peer `pack_done` waits; D2D + unpack; `xfer_done`; `compute_stream` waits `xfer_done`; BARRIER 2 | ready |
| `cgrid_vertical_solver` `require` | ready | consumes | ready |
| `gz_zh_sync_kernel` (zh <- gz) | ready | reads gz halo, writes zh full box | ready (zh halo consistent, so H_ZH not invalidated) |
| `update_dz_c` / `Riem_Solver_C` | ready | overwrite gz interior | gz halo now stale, but not re-consumed this substep |
| substep entry, it>1 | n/a | no reconstruction, no exchange (`if (it==1)` at :433/:552) | n/a — gz restored from zh |

Frequency: **once per top-level step, on `it == 1` only.** Source-confirmed.

## 7. H_CWIND (§8) — unchanged

Not modified. Verified only for regression: it is still resolved after
`cgrid_pressure` (whose stage writes `uc`/`vc`), matching :614 `p_grad_c` ->
:618 start -> :640 complete. `NOT CAUSAL` stands.

## 8. Results

### Build
R8 Release build clean, no new warnings.

### Repeated-call stability (§9)
8x8x10, R8, 6 tiles, n_split=3, 50 calls: **FAIL — first NaN at call #7**
(previously call #4). Larger grids and R4 not run, per §9's instruction not to
proceed until the small reproducer is stable.

Re-verified after the §12 `wp_d_contract.cpp` revert (which links into the
benchmark tree too, so the reproducer had to be re-run to confirm the revert did
not perturb the result). Rebuilt `build_cuda_r8_perf`, re-ran the 50-call case:
still FAIL, **same first-NaN step and same spatial signature**:

```
[FV3DBG tile=2 step=6] it=1 cgrid_vertical_solver   gz     nan/inf=10/2475 first_idx=155
[FV3DBG tile=2 step=6] it=1 cgrid_vertical_solver   pkc    nan/inf=10/2475 first_idx=380
[FV3DBG tile=3 step=6] it=1 cgrid_vertical_solver   gz     nan/inf=10/2475 first_idx=36
[FV3DBG tile=3 step=6] it=1 cgrid_vertical_solver   pkc    nan/inf=10/2475 first_idx=261
```

`step=6` is the 7th call. 10 cells = one column over all k. `[FV3DBG-IDX]`
decodes tile 2's cells as `(3,8,1..10)`, i.e. `j=8=je`; tile 3's as `j=0=js-1`
— adjacent across the tile-2/tile-3 seam. Only 2 of 6 tiles fail first. The
revert is confirmed neutral with respect to the residual defect.

Note on log extraction: the diagnostic's field format is
`nan/inf=<nan>/<total>`, so a failing line is one whose **first** number is
nonzero (`nan/inf=10/2475`), not one containing the word "nonfinite". An earlier
grep for `nonfinite` matched nothing and briefly looked like a clean run; it was
a bad pattern, not a clean result.

### Determinism (§10)
Three independent runs of the fixed binary, 8 calls each.

Comparing only well-formed single-`fprintf` `[FV3DBG ...]` lines (the
`[FV3DBG-IDX]` cell dump issues one `fprintf` **per cell**, so with six
concurrent tile threads it garbles irreproducibly — a known instrument defect,
not a property of the computation):

| Scope | Lines per run | 1v2 diffs | 1v3 diffs |
|---|---|---|---|
| Pre-NaN, steps 0-5 | 15552 / 15552 / 15552 | **0** | **0** |
| All steps 0-7 | 18964 / 19039 / 18962 | 5523 | 5377 |

**Deterministic wherever determinism is meaningful.** Divergence begins only
after NaN enters the state, where ordering-sensitive comparisons on NaN
legitimately differ and the affected-cell count varies. Per §10 this is expected
floating-point/NaN behavior, not an independent defect.

### Sanitizers (§12)
Run on the fixed binary over the repeated six-tile path (8 calls). Sanitizer is
not on PATH under the validated module stack; absolute path required:
`/usr/local/other/nvidia/hpc_sdk/Linux_x86_64/24.7/cuda/12.5/compute-sanitizer/compute-sanitizer`

| Tool | Result |
|---|---|
| memcheck | **PASS** — `ERROR SUMMARY: 0 errors` |
| racecheck | **PASS** — `0 hazards displayed (0 errors, 0 warnings)` |
| initcheck | **ONLY PREVIOUSLY CLASSIFIED AUTHORITATIVE DEAD READS** — see below |

Two tool-invocation notes, both of which have now cost this project a run each:

1. `--track-unused-memory` is a **bare boolean flag** and takes no argument.
   `--track-unused-memory no` silently makes `no` the target application and
   yields `Target application doesn't exist or is not a valid executable` —
   which is what invalidated the 3A.11R initcheck attempt. `=no` instead errors
   with `does not take any arguments`. Omit the flag for the default.
2. `--print-limit` **must** be raised. See immediately below.

#### initcheck census — exhaustive, `--print-limit 5000`

**A 100-report sample is explicitly NOT accepted as a census.** 3A.9 records that
exactly this truncation once produced a *wrong* single-signature conclusion: its
100-sample showed only the `d_sw` path, and the raised-limit re-run revealed a
**second** path. The first pass here reproduced that trap precisely — it
truncated at 100 reports, all of them `d_sw`. It was discarded and re-run.

Exhaustive result (`ic_3a11s.sh`, log `ic_full_r8.log`,
`--grid 8 --npz 10 --n_split 3 --warmup 1 --iters 2`):
`ERROR SUMMARY: 3456 errors`, **no** `not printed` line — nothing truncated.

| findings | device kernel | host frames | 3A.9 path |
|---|---|---|---|
| 864 | `copy_corners_kernel` | `launch_copy_corners_cuda` ← `launch_del6_vt_flux_cuda` ← `launch_d_sw_cuda` ← `launch_dgrid_dynamics_cuda` | #1, `sw_core.F90:974` |
| 2592 | `copy_corners_kernel` | `launch_copy_corners_cuda` ← `launch_del6_vt_flux_cuda` ← `launch_update_dz_d` ← `launch_dgrid_height_transport_cuda` | #2, `nh_utils.F90:288` |

864 + 2592 = 3456. **One** headline form
(`Uninitialized __global__ memory read of size 8 bytes`), **one** device kernel,
**zero** other error kinds, **two** call paths — the same two 3A.9 classified.

The per-path level sets are the decisive check, and they match *exactly* what the
harness's own `nord` arrays predict under 3A.9's Fortran-derived rule that only
`nord == 1` under-fills the `ng = 3` halo (`nord == 0` never calls
`copy_corners`; `nord == 2` fills the whole halo):

| path | observed levels | harness array (`s = k % 4`, k = 0..9) | predicted `nord == 1` levels |
|---|---|---|---|
| `d_sw` (w-damping, `nord_w`) | `k ∈ {1, 5, 9}`, 288 each | `nord_w[4] = {0,1,0,2}` | k%4 == 1 → **{1,5,9}** ✓ |
| `update_dz_d` (height damping, `nord_v`) | `k ∈ {2, 6}`, 1296 each | `nord_v[4] = {0,2,1,2}` | k%4 == 2 → **{2,6}** ✓ |

Two independently-derived level sets, both predicted before being looked up, both
matching with no extra or missing level. A data-dependent uninitialized read would
not be confined to a level set determined purely by a damping-order selector.

Absolute counts differ from 3A.9's 672 (432/240) because the driving harness and
its invocation count differ; the *signature set* is what the classification turns
on, and it is unchanged.

Both paths are the same `del6_vt_flux` body at `nord == 1` with an unassigned `d2`
actual, whose contamination is provably confined to the outermost halo ring and
dead — derived index-by-index from the authoritative Fortran in 3A.9 §10, not
assumed from resemblance.

**Verdict: ONLY PREVIOUSLY CLASSIFIED AUTHORITATIVE DEAD READS. No new
signature. initcheck is NOT reported as clean** — the dead reads remain by
design, because they exist in the authoritative Fortran.

#### Side effect: 3A.11R's open hypothesis is REFUTED

3A.11R left one hypothesis untested for want of an initcheck run: that the
un-zeroed `WorkspaceId::cgrid` arena was the NaN source, since `update_dz_c`'s
`gz2`/`xfx`/`yfx`/`fx`/`fy` are `cudaMalloc`'d arena scratch, never zeroed, and
invisible to the registered-field diagnostic.

The exhaustive census reports **zero** uninitialized reads anywhere in
`update_dz_c` or any other C-grid-arena consumer — every one of the 3456 findings
is `copy_corners_kernel` under `del6_vt_flux`. The un-zeroed C-grid arena is
therefore **fully written before it is read**, and is **not** the source of the
residual NaN. This independently corroborates §9's attribution of the residual to
`Riem_Solver_C` by eliminating the one remaining alternative that the
field-level diagnostic could not see.

### Regression (§13)

**Retracted and re-run.** An earlier draft of this document claimed
"R8 CTest 19/19 PASS, R4 CTest 19/19 PASS". That claim was **invalid** and has
been withdrawn. Recorded here deliberately, because the failure mode is easy to
repeat:

`val_3a11s.sh` does `cd $R/build_cuda_r4 && ctest`. It **never rebuilds**. Audit
of object timestamps against the modified sources:

| build tree | `.o` files newer than the source edit |
|---|---|
| `build_cuda_r8_perf` (benchmark) | 20 of 47 — **rebuilt** |
| `build_cuda_r8` (CTest) | **0 of 47 — entirely stale** |
| `build_cuda_r4` (CTest) | **0 of 47 — entirely stale** |

So the 19/19 was produced by binaries that **do not contain the H_GZ_FIRST
fix**, and said nothing about regression. The R8 figure additionally had no
supporting log at all — only one `tests passed` line exists across every job
output, and it belongs to the R4 section. A green CTest against a stale tree is
worse than no result, because it reads as evidence.

#### Regression, re-run properly

Both CTest trees rebuilt from the modified source (`cmake --build`, exit 0, no new
warnings), then `ctest` on a `gpu_a100` node:

| Tree | First run (as found) | After reverting the `wp_d_contract` change |
|---|---|---|
| `build_cuda_r8` | 18/19 — `fv3_wp_d_contract` aborted | **19/19 PASS** |
| `build_cuda_r4` | 18/19 — `fv3_wp_d_contract` aborted | **19/19 PASS** |

The one failure was **not** caused by the H_GZ_FIRST fix — see §12. Every GPU
integration test passed in both runs, including `fv3_integrated_smoke`,
`fv3_dispatch_smoke`, `fv3_halo_exchange_multitile`, and all five WP-B tests.

**Final: R8 CTest 19/19 PASS, R4 CTest 19/19 PASS** against trees that actually
contain the stage split. No regression in 3A.4 height transport, 3A.5 scalar
transport, 3A.6 momentum, 3A.7 divergence, 3A.8 d_sw, 3A.9 integrated runtime,
or 3A.10 halo exchange.

## 12. An unrelated working-tree change was WRONG and has been reverted

The rebuilt CTest run immediately failed `fv3_wp_d_contract` in both precisions —
a failure the stale trees had hidden. It was **not** from this milestone's work.

**File:** `cuda/dyn_core/operators/wp_d_contract.cpp` (uncommitted, inherited,
classified "UNRELATED" in an earlier draft). It rewrote the `pe`/`peln` bounds
check from an axis-swapped form to plain `covers_3d`, on the stated rationale that
axis1 = y and axis2 = level "matching every other field here".

**That rationale is wrong.** FV3 dimensions `pe` and `peln` as **(i, k, j)** — the
level axis really is axis 1. Authoritative source, tag `geos/v3.0.0`:

```fortran
! dyn_core.F90:202-203
real, intent(inout):: pe  (bd%is-1:bd%ie+1, npz+1, bd%js-1:bd%je+1)
real, intent(inout):: peln(bd%is  :bd%ie,   npz+1, bd%js  :bd%je  )
! nh_core.F90:87-88 agrees
```

`pe`/`peln` are genuinely special in FV3; they are the exception the original
check was written for, not an oversight. `test_wp_d_contract.cpp:27-31` encodes
that deliberately — it special-cases exactly these two fields with
`extent[1] = 4 = npz+1` on the middle axis. The pre-existing check and the test
agreed with each other and with the Fortran. The change broke all three.

**Reverted** (`git checkout --`). The rejected version is preserved at
`/tmp/wp_d_contract_REJECTED_axis_change.cpp.bak` rather than destroyed.
After reverting, both suites are 19/19.

The real defect it was chasing is elsewhere: the symptom it cited — a false
"descriptor does not cover required bounds" at 16x16 but not 8x8 — is explained by
the **benchmark harness** allocating `pe`/`peln` as (i, j, k). At 8x8 the
mis-ordered axis2 extent (11) coincidentally exceeds the required y range (10) and
the check passes; at 16x16 the y range grows to 18 and it fails. **The harness is
what needs fixing, not the contract.**

### ESCALATION — a separate, latent inconsistency, deliberately NOT changed

`fv3_vertical_kernels.cu:358-359` writes `pe_out.at(i, j, k)` and
`peln_out.at(i, j, k)`, i.e. it treats axis2 as the level axis — the opposite of
the (i, k, j) descriptor layout the contract and the authoritative Fortran both
specify. If those descriptors are ever bound to real FV3 `pe`/`peln` storage, that
kernel writes transposed.

This is a **pre-existing** condition, independent of this milestone, and it is
potentially a numerical defect rather than a mechanical one. Per the standing rule
("if a fix requires changing scientific semantics, stop and escalate instead of
silently making the change") it is **escalated, not fixed here**. It needs its own
milestone to determine whether the CUDA path deliberately transposes `pe`/`peln`
at the boundary or whether the kernel indexing is simply wrong.

Status: `IMPLEMENTED` — not `NUMERICALLY VALIDATED`. No test currently exercises
`pe`/`peln` content through this kernel with a real (i, k, j) descriptor, which is
why it has stayed latent.

## 9. Residual defect — THIRD, DISTINCT, isolated to `Riem_Solver_C`

New signature after the ordering fix, and the evidence localizing it:

- First NaN in the whole run: `step=6` (call #7), `it=1`, stage
  `cgrid_vertical_solver`, fields `gz` and `pkc`, 10 cells = one column, all k.
- Steps 1-5 are entirely clean (0 NaN lines).
- At step 6 the `halo_in` and `cgrid_predictor` checkpoints are clean, including
  **`delz`, `zh`, `w`, `delp`** — all of which the diagnostic covers.
- Spatial signature changed from the SE corner to a **single mid-edge column**:
  `(i=3, j=8=je)` on tile 2 and `(i=4, j=0=js-1)` on tile 3 — adjacent across
  the tile-2/tile-3 seam — and only 2 of 6 tiles fail first.

Attribution: `gz_from_delz_kernel` computes `gz(k) = gz(k+1) - delz(k)` from
finite `delz` and `zs`, so it cannot manufacture a NaN — and `delz` is clean at
entry. `update_dz_c` outputs both `gz` and `ws3`, and **`ws3` is clean**, which
it would not be if `update_dz_c`'s division had degenerated (`ws3` is
`(zs - gz(km))/dt`). `Riem_Solver_C` outputs exactly `gz` **and** `pkc` — the two
fields that failed, together. Therefore the residual NaN originates in
**`Riem_Solver_C`**, not in the gz reconstruction, not in `update_dz_c`, and not
in halo ordering.

Reducing the acoustic timestep 4x (`n_split` 3 -> 12, a CLI argument, no rebuild)
does **not** prevent the blowup, so it is not a simple acoustic-CFL limit. Note
this test is not fully clean: 50 calls at n_split=12 executes 600 substeps versus
150, so per-substep growth is aggravated rather than relieved.

Not investigated further in this milestone, and deliberately not "fixed" —
§4 forbids rewriting `Riem_Solver_C` numerics without a separately proven defect,
and no such proof exists yet. Two candidate explanations remain open and are
**not** discriminated:

1. A genuine defect in the C-grid Riemann solver (e.g. `p_fac` handling, layer
   collapse guard, or the `a_imp = 1.0` SIM1 path).
2. Progressive physical blowup of an unbalanced synthetic benchmark state. The
   harness state is not in hydrostatic or geostrophic balance — uniform `delp`,
   `pt = 280 + 10*(k%NPZ)/NPZ`, arbitrary sinusoidal 10 m/s `uc`/`vc`,
   `w = 0.01` — so a real instability over ~6 steps is plausible.

The cleanest next discriminator is to run a quiescent, hydrostatically
consistent state (`uc = vc = w = 0`, no `horiz` perturbation). If that survives
50 calls, explanation 2 holds and the harness state needs balancing; if it still
fails, explanation 1 holds and `Riem_Solver_C` has a real defect. This must be
weighed against §11's requirement for per-tile-asymmetric data (the harness
already provides `make_tile_asymmetric`), so both a balanced and an asymmetric
variant are needed.

## 10. Gate C1 reassessment (§14)

Gate C1's original evidence was a **single** integrated invocation, and this
milestone confirms that was insufficient: a single call cannot expose a defect
whose mechanism is "the halo carries the *previous* step's interior." One call
has no previous step.

Current status: the integrated path still executes end-to-end, R8 CTest is
19/19, and the corrected ordering now matches authoritative Fortran — so the
single-invocation Gate C1 claim is **not invalidated**, and is now better
founded. But Gate C1 must **not** be reported as covering repeated execution.
Repeated-step Gate C1 is **BLOCKED** pending the residual `Riem_Solver_C` issue.

## 11. Uncommitted work classification (§16)

| Item | Classification |
|---|---|
| `vertical/fv3_vertical_kernels.{cu,cuh}` stage split | **KEEP — legitimate production fix** (this milestone) |
| `include/fv3_gpu_internal.hpp` new dispatch slot | **KEEP — legitimate production fix** |
| `runtime/fv3_gpu_production_dispatch.cu` wiring | **KEEP — legitimate production fix** |
| `runtime/fv3_gpu_runtime.cpp` reorder + comment/table correction | **KEEP — legitimate production fix** |
| `runtime/fv3_gpu_runtime.cpp` `WorkspaceId::vertical` sizing fix | **UNRELATED — separate commit** |
| `runtime/fv3_gpu_runtime.cpp` `FV3_GPU_DBG_STAGE` / `debug_check_finite_stage` | **KEEP — reusable diagnostic** (gated behind `FV3_GPU_DEBUG_FAIL`); revisit once stability closes |
| `operators/wp_d_contract.cpp` axis-order "fix" | **REVERTED — proven wrong against authoritative Fortran; it broke `fv3_wp_d_contract` in both precisions. See §12.** |
| `benchmarks/run_small_perf_bench.cpp` `delz` fixture fix | **KEEP — benchmark fix** |
| `benchmarks/` harness + `CMakeLists.txt` wiring | **KEEP — benchmark fix** |
| `build_cuda_r8_perf/`, `*.log`, `*.out`, `*.stdout`, `*.sorted`, `val_3a11s.sh`, `ic_3a11s.sh`, `sanitize_3a11r.sh`, `determinism_3a11s.sh` | **REMOVE — never commit** |

Because the runtime diagnostic and the `WorkspaceId::vertical` fix share a file
with the ordering fix, a clean three-way split would need `git add -p`. Nothing
has been committed or pushed.
