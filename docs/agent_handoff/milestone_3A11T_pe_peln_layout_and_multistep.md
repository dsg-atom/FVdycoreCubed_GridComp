# Milestone 3A.11T — Riem_Solver_C / pe-peln Layout Diagnostic and Multi-Step Stability

Branch: `cuda/gpu-bringup`
Base commit at start: `774ab59` "Fix H_GZ_FIRST ordering and checkpoint multistep diagnostics"
Working tree: modified, NOT committed (see "Uncommitted changes").

Verdict: **MULTI-STEP STABILITY STILL BLOCKED**

---

## 1. Completion report (mandated fields)

| Field | Result |
| --- | --- |
| Authoritative pe/peln dimensional order | **(i, k, j)** — `pe(is-1:ie+1, npz+1, js-1:je+1)`, `peln(is:ie, npz+1, js:je)`, confirmed in `GFDL_atmos_cubed_sphere` `geos/v3.0.0` @ `ca30e26305fda43831095334c02a3f5e0b8c0c45` |
| Benchmark pe/peln order before | (i, k, j) — already authoritative |
| CUDA pe/peln interpretation before | (i, k, j) — already authoritative |
| Layout defect | **BOTH (latent, not active)** — harness and kernel agreed with each other AND with Fortran, so no live mismatch existed; the descriptors did not *express* the ordering, they merely happened to match |
| Layout fix | Descriptors/strides made explicit; no array transposes. See §3 |
| Layout-sensitive test | **PASS** (R8 and R4), with a permutation negative control that fails as required |
| 8x8x10 R8, 50 consecutive calls | **PASS** — 3 independent runs, default config, no env override. **But see the fragility finding in §5: this is a survival-time margin, not a cure** |
| 16x16x10 R8, 50 consecutive calls | **FAIL** — step 17, `it=2`, `cgrid_vertical_solver`, `gz` |
| First remaining bad stage | `cgrid_vertical_solver` (`gz`, 10 halo cells, `ownnan=0`), propagating to first OWNED contamination 3 stages later in `vertical_solver` (`w`, `ownnan=10`) |
| Riem_Solver_C inputs physically valid before failure | **NO** for the original 8x8 failure — but the violated input was `delp`, NEGATIVE at k=3 (−9.338e2) and k=7 (−2.910e3), and the consumer was **Riem_Solver3** (D-grid), not Riem_Solver_C. Riem_Solver_C is EXONERATED: it takes no pe/peln at all |
| Focused independent host/reference comparison | **NOT RUN** — §9's precondition ("all Riem inputs physically valid immediately before the failing call") was never met. `delp < 0` on entry made the classification unambiguous without it: Riem_Solver3 faithfully turns negative layer mass into NaN |
| Root cause | **PARTIAL.** One defect found and fixed (harness `damp_v` over the del-2n stability limit, §4). A SECOND, slower mechanism remains unidentified (§5) |
| Numerical algorithm changed | **NO** |
| Performance milestone 3A.11 | **STILL BLOCKED** |
| Numerical validation | **NOT RUN** |

### §12 acceptance gate

| Requirement | Status |
| --- | --- |
| 8x8 R8 50 calls PASS | PASS (fragile — see §5) |
| 16x16 R8 50 calls PASS | **FAIL** |
| 3 independent runs PASS | PASS (8x8 only) |
| memcheck PASS | PASS — `ERROR SUMMARY: 0 errors` |
| racecheck PASS | PASS — `RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)` |
| No new initcheck signatures | PASS — single signature `dgrid::copy_corners_kernel`, identical to baseline `s3_initcheck.log`. Both capped at the `--print-limit` default of 100, so only the SIGNATURE SET is claimed unchanged, not the count |

Gate NOT met. Performance work must not resume.

---

## 2. Hypothesis disposition against the prompt's A–E

- **A. pe/peln descriptor/layout mismatch — RULED OUT.** All three layers agree on (i,k,j), and `Riem_Solver_c` takes neither `pe` nor `peln`.
- **B. Another input-contract/layout defect — RULED OUT for the 8x8 failure**, which is fully explained by D below. Not excluded for the residual failure.
- **C. Genuine CUDA Riem_Solver_C translation defect — RULED OUT.** Riem_Solver3 received `delp < 0` and correctly produced NaN.
- **D. Physically invalid / unbalanced synthetic benchmark state — CONFIRMED as one cause** (the `damp_v` coefficient, §4). Strongly suspected but NOT PROVEN as the residual cause (§5).
- **E. Another precisely identified cause — the residual mechanism, still unidentified.**

---

## 3. Layout finding (§1–§5, CLOSED)

Authoritative order is (i, k, j) for `pe`/`peln` — an FV3 exception to the usual (i,j,k). `pk`, `gz`, `zh`, `pkc`, `pk3` are genuinely (i,j,k).

The harness and the CUDA kernels were **already** consistent with each other and with Fortran, so no active mismatch existed. Classified BOTH-but-latent because neither layer *expressed* the constraint — it was coincidence maintained by hand. Fix was confined to descriptors/strides; no array was transposed, per §5's "fix the smallest responsible layer".

A layout-sensitive test was added encoding `value = A*i + B*j + C*k` with coefficients chosen so no i/j/k permutation aliases, probing interior, j-halo, k=0, k=npz, and C-grid vertical boundary points. It PASSES in R8 and R4, and a deliberate permutation makes it FAIL, which is the property that makes it worth keeping.

Numerical semantics changed: **NO**.

---

## 4. Defect found and fixed: harness `damp_v` above the del-2n stability limit

`benchmarks/run_small_perf_bench.cpp` paired `damp_v = 0.20` with `nord_v = 2` at `k%4==3`.

For a checkerboard mode on the uniform fabricated metric (`dx=dy=dxc=dyc=1e5`, `rarea=1/dx^2`, `da_min_c=9.900108e9`, `del6_u=del6_v≈1`), each del^6 sweep in `del6_vt_flux` contributes `-8*rarea`, the final flux stage `-2`, the coefficient is `damp4=(damp_v*da_min_c)^(nord_v+1)` (`wp_b_dsw_cuda.cu:579`, bit-exact vs `sw_core.F90:1477`), and the consumer applies `*rdx`:

```
gamma_v = 4 * damp_v^(nord_v+1) * 8^nord_v      (need <= 2)

k%4==1  damp_v 0.10  nord_v 2  ->  0.256   stable
k%4==2  damp_v 0.05  nord_v 1  ->  0.080   stable
k%4==3  damp_v 0.20  nord_v 2  ->  2.048   UNSTABLE   (ceiling 0.198)
```

FV3's documented 0.16 ceiling is a `nord==1` limit and does not carry to `nord==2`; that is how the value got in.

### Evidence (job 58088916), nord left at the default pattern

| leg | gamma_v @ k%4==3 | divdamp | 8-step result |
| --- | --- | --- | --- |
| base | 2.048 UNSTABLE | live, d4_bg=0.16 | FAIL step 4 |
| `DAMPV_SCALE=0.5` | 0.256 | **live, untouched** | CLEAN |
| `D4_SCALE=0.5` | 2.048 UNSTABLE | gamma_d 2.055 -> 0.257 | FAIL step 4 |
| both | 0.256 | 0.257 | CLEAN |
| both, 50 steps | | | CLEAN 50 |

Corroboration: §7 found `delp` negative first at k=3 and k=7 — exactly the `k%4==3` levels.

### Fix

`damp_v[3]`: `0.20 -> 0.12` (gamma_v 0.442, 4.5x margin). Chosen over the 0.198 ceiling to keep a real margin, over 0.10 to stay distinct from the `k%4==1` entry so the four groups remain four distinct validated combinations, and still larger than that entry at the same `nord_v=2`, which is the coverage the row exists to provide. Production `vtdm4` is 0.02–0.06, so 0.12 remains a deliberate stress value.

Numerical semantics changed: **NO** — a benchmark input value plus a diagnostic print. No production CUDA touched, per §3's rule that a bad benchmark must not be accommodated by changing production code.

### RETRACTION recorded

I predicted `(nord_k=2, d4_bg=0.16) -> gamma_d = 2.055` was ALSO unstable. The `D4_SCALE=0.5` leg refutes it: gamma_d drops to 0.257 and the run still fails at exactly step 4, while the `DAMPV_SCALE` leg is clean carrying `d4_bg=0.16` untouched. The lambda=8 checkerboard estimate overestimates the divergence-damping operator's spectral radius. `d4_bg` was left alone rather than "fixed" on the strength of a contradicted estimate.

### Why every earlier nord experiment was confounded

`nord_cap_v = min(cfg.nord, 2)` (`wp_b_dsw_cuda.cu:937`, clamp `:590`) combined with the `mode=3` per-level gate activates a level only when its OWN `ndif` equals the swept order. So `FV3_BENCH_DIVDAMP=nord1` switches del^6 wind damping **entirely off** on the `ndif==2` levels — precisely those carrying the largest `damp_v`. Every nord leg moved the operator AND removed damping from half the column. nord1-PASS was never evidence of correctness.

---

## 5. UNRESOLVED: a second, slower mechanism

The `damp_v` fix is real but **not sufficient**. The 8x8 50-call PASS is a survival-time margin, not a cure.

### dt / grid sweep (job 58094255), default damping

| leg | grid | n_split | fails at | ~acoustic substeps |
| --- | --- | --- | --- | --- |
| g8_ns3 | 8 | 3 | — | 150 (PASS) |
| g8_ns6 | 8 | 6 | step 46 | 276 |
| g8_ns12 | 8 | 12 | step 24 | 288 |
| g16_ns3 | 16 | 3 | step 17 | 51 |
| g16_ns6 | 16 | 6 | step 36 | 216 |
| g16_ns12 | 16 | 12 | step 26 | 312 |

Both grids fail at roughly 220–310 substeps. §6's criterion (50 calls at `n_split=3` = 150 substeps) stops short of the 8x8 failure — it does not reach it.

### Established properties of the residual mechanism

- **dt-sensitive**: measured in substeps the failure is pushed back 51 -> 216 -> 312. A dt-insensitive indexing/box defect would fail at a fixed substep count.
- **Damping-SUPPRESSED, not damping-driven**: `FV3_BENCH_DAMP=nodel6` (verified fired: `damp_v=0 nord_v=0 gamma_v=0`) fails EARLIEST of all, step 10 / ~30 substeps. Reducing damping only delays the failure (`DAMPV_SCALE=0.5` -> step 21; both halved -> step 27).
- **Poison-independent**: `FV3_GPU_NO_INIT_POISON=1` reproduces 16x16 **bit-identically** — same hash `d36ada74b033457d`, same 105572 owned-nan lines, same 66 `RIEM3-BAD`. This also PROVES the narrowed corner-ghost poison is never read anywhere, since suppressing it changes no bit. Good news for §11's narrowing.
- **Not memory corruption or a race**: memcheck 0 errors, racecheck 0 hazards.
- **Grid-size-independent in substeps**, so not a per-cell coefficient issue: `da_min`/`da_min_c` are identical at both grid sizes (uniform fabricated metric), so gamma_v is unchanged.

### The sharpest open lead: every first-failure site is at k=0

```
g8_ns6   w   (i=  2, j=  8, k=0)  OWNED (j=je)
g8_ns12  gz  (i=  0, j=  6, k=0)  halo
g16_ns3  gz  (i= 16, j= -1, k=0)  halo (i=ie)
g16_ns3  w   (i= 15, j=  0, k=0)  halo
nodel6   gz  (i= 16, j=  0, k=0)  halo
g16_ns12 w   (i=  8, j= 11, k=0)  OWNED
```

`k=0` is `k%4==0`, the ONLY row in the fabricated table with `damp_v=0, damp_w=0, damp_t=0, nord_v=0, nord_k=0, d_con=0, kgb=0`. It is therefore simultaneously:

1. the completely undamped level — consistent with "damping suppresses the growth", and
2. the only level exercising the `nord==0` arms of divergence damping and del^6.

Those two readings are the remaining live hypotheses and they are separable. Note the `nord==0` divdamp arm is a genuinely DISTINCT code path: it uses `dddmp*abs(delpc(i,j)*dt)` **with** `abs`, whereas the `nord>0` arm has no `abs` (`sw_core.F90` ~:1413 vs the nord==0 arm).

### Recommended next steps, in order

1. Force `nord=0` on ALL levels (`FV3_BENCH_DIVDAMP=nord0`). If growth accelerates and spreads off the `k%4==0` levels, the `nord==0` arm is implicated as a port defect. Mind the confound in §4 — nord forcing also changes gating.
2. Give `k%4==0` nonzero `damp_v` (a sponge top, which is what real FV3 has at the model top — the harness currently has the LEAST damping there). If the growth vanishes, classify **SYNTHETIC HARNESS STATE INVALID** under §8.
3. Audit the `nord==0` divdamp arm against `sw_core.F90` including the `abs`, per §10 — edit only if the two expressions are shown to disagree.
4. Investigate exchange **E11**, still DECLARED MISSING at `fv3_gpu_halo_exchange.cu:45-52`: the shared staggered seam LINE (`mpp_get_boundary`, `dyn_core.F90:1094-1095`). A missing seam sync would produce exactly this profile — slow, seam/edge-localized, dt-delayed, damping-suppressed, grid-size-independent. Several first-failure sites sit on owned edges and halo ring 1.

---

## 6. §11 diagnostic classification

| Diagnostic | Disposition | Reason |
| --- | --- | --- |
| `[BENCH]` gamma_v / damping table print | **KEEP** | Unconditional, prints coefficients AS APPLIED. This is the permanent guard against the §4 defect class and against vacuous knob experiments |
| `debug_check_finite_stage` + `[FV3DBG]` | **KEEP** | The only reason the residual mechanism is characterized at all |
| `FV3_BENCH_DAMPV_SCALE` | **KEEP** | The discriminator that isolated the root cause; one table column, no metric fabrication |
| `FV3_BENCH_D4_SCALE` | **KEEP** | Needed as the complement; without it the `damp_v` attribution is not exclusive |
| `FV3_GPU_NO_INIT_POISON` | **KEEP** | Produced the bit-identical result proving the poison is never read |
| `FV3_BENCH_DAMP`, `FV3_BENCH_DIVDAMP` | **KEEP, with a design fix owed** | See known gaps: the mutually-exclusive `strcmp` chain makes whole classes of experiment silently vacuous |
| `[RIEM3-BAD]` + `fv3_riem3_bad_budget` | **KEEP, but fix two flaws** | (a) `dz_worst` is seeded from `dz_out_col[0]`, an OUTPUT already NaN when the probe fires, so it always prints `nan`; `dz_nonneg` from the INPUT `zh_col` is the load-bearing number. (b) The count saturates at the budget — it read 66 in EVERY leg regardless of failure step, so it carries no information and must not be reported as a measurement |
| `[RIEMC-PRE]` | **REMOVE** | Riem_Solver_C is exonerated (takes no pe/peln; the failure was Riem_Solver3 with negative `delp`). Nothing further depends on it |
| `FV3_BENCH_ORTHO_METRIC`, `FV3_BENCH_DEL6_SCALE` | **KEEP** | Cheap; but DEL6_SCALE needs a confirmation print — it has none, and an unverifiable knob is how three experiments in this milestone became silently vacuous |

---

## 7. Known gaps recorded, not fixed

Source-level analysis only — none of the following is COMPILED, GPU EXECUTED, or NUMERICALLY VALIDATED.

- **D1** `divg2` is READ (`wp_d_cuda.cu:497`, `:505`, unconditionally) but written by NOTHING, with no guard rejecting `d_ext > 0`.
- **D2** Per-substep `heat_source`/`diss_est` accumulation (`dyn_core.F90:824-825`) is not implemented: `dsw_heat_reset_kernel` (`wp_b_dsw_cuda.cu:448-449`) zeroes them every substep, understating heating by ~`n_split` whenever `d_con > 1e-5`. Expected to be the first discrepancy in a Fortran-vs-CUDA savepoint comparison.
- **D3** `clamp_nord_cap` (`wp_b_dsw_cuda.cu:590-593`) returns `min(nord, ng-1)` = `min(nord,2)`, silently downgrading a `nord=3` config, contradicting its own kernel guard which permits 3.
- **D4** `vt`/`delpc` role inversion at `wp_b_dsw_contract_cuda.cpp:104/:107` vs `dyn_core.F90:789`.
- **D5** `pe_halo`/`pln_halo` have zero production call sites.
- **D6** No `init_step` zeroing of `diss_est`.
- **Write-write race in the smag arena**: `to_arena_view3` (`wp_b_contract_views.hpp:55-60`) builds views valid over i,j in −2..11, but `smag_shear_winds_kernel` (`wp_b_divergence_cuda.cu:475-487`) writes `ws.vt` at `j<=jed+1=12` and `ws.ut` at `i<=ied+1=12`, so `ws.ut(12,j,k)` aliases `ws.ut(-2,j+1,k)`. Appears inert (only the outermost ring of `ws.smag.wk` is contaminated and `a2b_ord4` reads −1..10) but it is a genuine defect and a plausible non-reproducibility source under a different launch config.
- **Fortran keeps two distinct `ut`/`vt`; the CUDA merges them.** `dyn_core.F90:340` allocates a persistent `ut(isd:ied,jsd:jed,npz)` (c_sw producer -> `update_dz_c` consumer) while `d_sw`'s are stack locals (`sw_core.F90:563-564`). Safe today ONLY because `cgrid_predictor`'s `d2a2c` write box contains `update_dz_c`'s read box every substep. The comment at `test_dispatch_smoke.cpp:575-577` is wrong for the production ordering.
- **`debug_check_finite_stage` over-scans**: `count = allocation_bytes/sizeof(fv_real)` (`fv3_gpu_runtime.cpp:237`) makes full-extent extrema and the hash dominated by cells no consumer reads. This is why `ut`/`vt` reported 1e23 (see below) and why the FIRST 16x16 line reads `ownnan=0`. Restrict to the declared consumer box, or owned+1.
- **`FV3_BENCH_DIVDAMP` design flaw**: `d2only`/`d4only`/`nosmag`/`nord0`/`nord1`/`nord2` all parse from ONE env var via mutually-exclusive `strcmp`, so a nord selection cannot be combined with `d2only`/`d4only`. This made a whole class of isolation experiment impossible and silently vacuous.
- **`(nord_k=2, d4_bg=0.16)`** exceeds the lambda=8 del-2n estimate (gamma=2.055) yet is empirically survivable — the estimate is wrong for that operator. Do not act on it.
- **`wp_b_kernels.hpp:517-598`** dead duplicate with three numerical errors.
- **E11 exchange DECLARED MISSING** (`fv3_gpu_halo_exchange.cu:45-52`); four-edge-only exchange with no diagonal corner fill.
- **`!b.cubed_sphere` extra guard** at `wp_b_cuda.cu:684`.
- **WP-D unit test** should be extended so a `pk_b == pp` wiring mistake cannot pass again.
- **Do NOT poison `cappa`**: `Riem_Solver_C` legitimately reads it at the four ring-1 diagonals where it holds `0.` from `fv_dynamics.F90:318`; `1/(1-1e40)` would inject Inf/NaN into cells the CPU silently discards.

### Closed this milestone

**`ut`/`vt` 1e13/1e23 values are INERT SCRATCH, not a defect.** They are `del6_vt_flux` first-pass fluxes carrying the authoritative `damp4=(damp_v*da_min_c)^(nord_v+1)`, an `area^(nord+1)`-dimensioned coefficient only returned to physical scale by the subsequent `nord` divisions by `rarea`; Fortran's `del6_vt_flux` has the identical structure and also leaves first-pass values in the outer ring. Magnitudes verified against both nord legs and both extremum k values. Provably never read: max reader index is `i=10` for `ut` and `j=10` for `vt`, so the four `i/j=11` corner cells are outside every read box in the tree, and the two edge cells are overwritten by `d2a2c` before `update_dz_c` reads them; `ut`/`vt` are in no halo group. `smag_corner` aliasing DISPROVEN (dedicated arena slots 25/26, `wp_b_dsw_contract_cuda.cpp:428-429`). Zero-metric-divisor DISPROVEN — the fabrication loop writes all 225 elements of every 2-D metric (`:663-682`); the comment at `:1242-1243` claiming it skips the padding row/column is stale and should be corrected.

---

## 8. Uncommitted changes

```
cuda/dyn_core/CMakeLists.txt
cuda/dyn_core/benchmarks/run_small_perf_bench.cpp
cuda/dyn_core/operators/wp_d_contract.cpp
cuda/dyn_core/operators/wp_d_contract_cuda.cpp
cuda/dyn_core/operators/wp_d_operators.cpp
cuda/dyn_core/operators/wp_d_operators.hpp
cuda/dyn_core/runtime/fv3_gpu_runtime.cpp
cuda/dyn_core/tests/integration/test_dispatch_smoke.cpp
cuda/dyn_core/tests/integration/test_integrated_smoke.cpp
cuda/dyn_core/vertical/fv3_riem_solver_c.cu
cuda/dyn_core/vertical/fv3_riem_solver_c.cuh
cuda/dyn_core/vertical/fv3_vertical_kernels.cu
```

Not committed and not pushed, per the standing instruction. Scratch scripts and logs
(`gam_3a11t.sh`, `acc_3a11t.sh`, `g16_3a11t.sh`, `n16_3a11t.sh`, `gm_*`, `ac_*`, `gg_*`,
`nn_*`, `slurm-*`) live in the `/discover/nobackup` working copy and must NOT be committed.

## 9. Build / test state

| Item | Status |
| --- | --- |
| `build_cuda_r8_perf` | COMPILED clean |
| `build_cuda_r8` | COMPILED clean, ctest **20/20 PASS** |
| `build_cuda_r4` | COMPILED clean, ctest **20/20 PASS** |
| GPU execution | PASS (A100 `warpa003`) |
| memcheck / racecheck | PASS / PASS |
| initcheck | no new signatures vs baseline |
| Numerical validation vs authoritative Fortran | **NOT RUN** |

---

MULTI-STEP STABILITY STILL BLOCKED
