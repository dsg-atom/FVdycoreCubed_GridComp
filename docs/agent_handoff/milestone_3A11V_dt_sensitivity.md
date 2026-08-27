# Milestone 3A.11V — Bottom-Level Growing-Mode dt-Sensitivity Diagnostic

Branch: `cuda/gpu-bringup`
Parent commit at start: `774ab59` "Fix H_GZ_FIRST ordering and checkpoint multistep diagnostics"
Precision: R8 unless stated. Grid: cubed sphere, 6 tiles, `grid_type=0`, `hydrostatic=false`, `a_imp=1.0`.

## Verdict

The residual bottom-level growing mode is **RESOLVED**. It is a **2-delta del-2n
divergence-damping instability created by an invented benchmark damping table**,
it is **dt-independent**, and the **authoritative configuration is not exposed to
it**. A **separate** and genuine CUDA defect (two gravitational constants inside
one acoustic substep) was found, repaired, and GPU-verified; it sets the mode's
initial amplitude profile but is **not** on its causal path.

Validation levels reached this milestone:

| Item | Level |
|---|---|
| grav plumbing repair | IMPLEMENTED / COMPILED / **GPU EXECUTED** |
| grav repair vs Fortran savepoints | **NOT NUMERICALLY VALIDATED** |
| damping-table faithfulness | diagnosed + documented, **NOT repaired** |
| section 13 acceptance campaign | **NOT COMPLETE** (see Blockers) |

## 1. Primary reproducer

```
build_cuda_r8_perf/fv3_gpu_perf_bench --grid 8 --npz 10 --n_split 3 \
    --warmup 0 --iters 300 --repeats 1 --mode resident --hardware A100
env: FV3_BENCH_WIND_SCALE=0 FV3_BENCH_HORIZ_SCALE=0
     FV3_GPU_DEBUG_FAIL=1 FV3_GPU_DEBUG_CSV=1 FV3_GPU_DEBUG_TILE=1
     FV3_GPU_DEBUG_PROBE=4:5:1,4:5:5,4:5:9 FV3_GPU_DEBUG_FIELDS=u,v,delp,w
```

Seam-free (`HORIZ_SCALE=0`) so the invalid seam-discontinuous IC is excluded.
Fails at top-level call 189, `it=2 dgrid_dynamics`, field `u`, 105/2250 nonfinite.
Bit-reproducible: `L_loc` and `L_ctl` differ only in probe cells and produce an
identical failure hash `cd0f6ed639bd41` at the identical substep.

## 2. dt / n_split matrix

`step.bdt = 10.0` s and `k_split = 1` are fixed (`run_small_perf_bench.cpp:1847`),
and `p.dt = s->bdt / n_split` (`fv3_gpu_runtime.cpp:1326-1327`, citing
`dyn_core.F90:296`). Outer-step duration therefore does **not** vary with
`n_split`, so this sweep **is** a controlled-dt experiment: model time after N
calls is 10N s regardless, and acoustic dt = 10/n_split.

| n_split | acoustic dt (s) | growth / acoustic substep | growth / model second | fail substep | fail model time (s) |
|---|---|---|---|---|---|
| 1 | 10.000 | +0.072044 | +0.007204 | 525 | 5250 |
| 2 | 5.000 | +0.070958 | +0.014192 | 547 | 2735 |
| 3 | 3.333 | +0.068451 | +0.020535 | 570 | 1900 |
| 6 | 1.667 | +0.069100 | +0.041460 | 584 | 973 |
| 12 | 0.833 | +0.069467 | +0.083360 | 602 | 502 |
| 24 | 0.417 | +0.071241 | +0.170978 | 615 | 256 |

Growth per **substep** is constant to ±2.5% over a **24x** dt range. Growth per
**model second** spans 23.7x, i.e. exactly proportional to 1/dt. Failure model
time falls 5250 -> 256 s while failure substep count *rises* 525 -> 615.

**Classification: dt-INDEPENDENT per substep. NOT a CFL/timestep instability.**
Refining dt does not help and never will; it makes wall-clock failure *sooner*.

Nearly grid-independent: 16x16 gives +0.075395, fail at substep 542.

## 3. Mode amplitude and location

Machine-readable CSV throughout (`CSVPROBE` rows, `an_3a11v.py`), never awk on
formatted logs. Probe cells pre-decoded to (i,j,k) in the runtime.

Initial amplitude 2.775e-15 (`u` at `nh_pressure`). Mode located at (i=4, j=5)
at **every** level whose `nord_k > 0`.

## 4. Growth rate

Classification is by **functional form**, not failure time. amp/substep flat =
linear ramp; rising = exponential.

Two corrections to my own earlier readings are recorded here because both were
wrong in instructive ways:

- I once fitted a log-slope to the divdamp-off residual and reported
  "+0.0073/substep decaying". A linear ramp's apparent log-slope *must* decay as
  1/x. amp/substep is constant at 2.50e-15 (ns=3) and 5.85e-16 (ns=12), ratio
  4.27 ~ 12/3, i.e. constant injection per model second, ~7.2e-16. Not a mode.
- My `r>8` first/last classifier labelled `A_16` EXPONENTIAL. Its piecewise
  slopes are +0.0264, **-0.0070**, +0.0024, +0.0013, +0.0120 — sign-changing —
  and it reaches 1.95e-12 over 900 substeps where +0.0685 would reach e^61.
  It is the bounded ramp. The classifier caught an early transient.

## 5. Zero-perturbation origin (section 5)

Wind is bitwise `0.0` after `entry`, `halo_in`, `cgrid_predictor`,
`cgrid_vertical_solver`, `cgrid_pressure`; `u` first becomes nonzero at
`dgrid_dynamics`. `w` is bitwise `0.0` through `dgrid_height` and first becomes
nonzero at **`vertical_solver`**. Not roundoff — see section 12.

## 6. Physical balance of the seam-free state

**BALANCED.** The IC satisfies the SIM1 gamma-law invariant
`((rdgas/grav)*delp/(-delz)*pt)^(1/(1-akap)) == pm2` (`nh_utils.F90:1232`) to
<= 1.1e-15 relative, max residual 8.7e-11 Pa, at all ten levels, with every wind
component, `w`, `omga`, `ws`, `ws3` exactly `0.0`. It can account for
w ~ 1e-11 m/s and no more. My earlier "unbalanced synthetic IC" inference is
**withdrawn**; section 11 is therefore not the answer.

Caveat that is *not* resolved: the harness fabricates a uniform 100 km metric
with no per-cell lat/lon (`run_small_perf_bench.cpp:829-848`), so a globally
consistent cube IC is not constructible without new work (task #20). This makes
`divg_u == del6_u == sina_v` bit-identically, so a divg/del6 swap would be
invisible in this benchmark — a real coverage hole, unrelated to this milestone.

## 7. Bottom-level special operators audited (NOT modified)

`Riem_Solver3` bottom BC `dd(km) += r2g*dm(km)` (`nh_utils.F90:1109`) and
`p1(km) -= r6g*dm(km)` (`:1185`); `sim3`/`sim3p0` twins at `:962`, `:1042`;
`Riem_Solver_C` bottom BC; `ws`/`ws3` surface vertical velocity; `delz`/`zh`
update; `gz = grav*zh`; `kgb = ke_bg` at `k==npz` (`dyn_core.F90:707-711`);
`d_con` at the bottom level; `damp_v`/`damp_w`/`damp_t` at k=9.

**Outcome: no bottom-boundary operator defect was found.** The k=9 localization
turned out to be an artifact — see section 10.

## 8. Damping discriminator (section 8)

Growth per substep, seam-free, ns=3, classified by functional form:

| config | growth/substep | note |
|---|---|---|
| default | +0.068451 | |
| `nodel6` | +0.0685 | unchanged — del6 is NOT the cause |
| `dampv05` | +0.0685 | unchanged |
| `noheat` | +0.0685 | **bit-identical**, hash `d04d72b52dd68470` |
| `nosmag` | +0.0687 | unchanged (a failure-time-only reading misled me here once) |
| `nodivdamp` | +0.0070 | **mode gone** |
| `none` | +0.0070 | mode gone |
| `nord0` | +0.0073 | mode gone |
| `d2only` | +0.0071 | mode gone |
| `d4only` | +0.066046 | **mode reproduced** |

Confirms quantitatively that the mode is **not** caused by excessive del6
damping, and that disabling del6 does not help. No damping was added as a fix.

## 9. First energy-amplifying stage (section 9)

`vertical_solver` is the **only** stage that ever changes `w`/`delz`/`zh`.
`delp` is exactly x1.000000000 at every stage. `u`/`v` change only at
`dgrid_dynamics` (x1.094927877, after removing the dx=1e5 unit switch) and
`nh_pressure` (x1.000151623, after x1e5), net x1.095093893 per substep,
ln = 0.090840 — consistent with the measured +0.0685 after the halo/other
stages. **Attribution: WP-B `d_sw`, the `nord>0` divergence-damping arm.**

UNIT TRAP, load-bearing: `u`/`v` are held in `u*dx`/`v*dy` form between `d_sw`
and `nh_p_grad` (dx=dy=1e5), so raw magnitudes swing x1e5 at `dgrid_dynamics`
and x1e-5 at `nh_pressure`. Every growth series here is taken at **one** stage
(`nh_pressure`). `pkc` x4082 at `cgrid_vertical_solver` and x0.000266 at
`vertical_solver` is the `huge_r` poison/rebuild artifact (product ~1.085), not
growth.

## 10. Root-cause classification

**UNBALANCED / INVALID SYNTHETIC BENCHMARK CONFIGURATION** — specifically an
invented per-level damping table that pairs `d4_bg = 0.16` with `nord_k = 2`,
1.6% above the 2-delta del-2n stability limit. **Not** a CUDA/driver defect in
the damping path.

Corrected stability norm, re-derived term-by-term from the Fortran statements:

```
gamma_d4 = (8*d4_bg)^(nord+1) * (da_min/area_local)     unstable for gamma > 2
critical d4_bg = 2^(1/(nord+1))/8 = 0.2500 / 0.17678 / 0.15749  (nord 0/1/2)
```

My earlier `gamma = 4*c^(nord+1)*8^nord` was **exactly 2x too small**: I counted
a 1-D mode, whereas `divergence_corner` contributes 2 **per direction**. The
harness's own comment recording "gamma_d == 2.055 is contradicted by
measurement" was contradicted for that reason, and is now superseded in source.

Evidence, strongest first:

1. **Parameter-free threshold prediction confirmed at two different nord.**
   Only the narrow `r = da_min/A in [0.9608, 1.0]` band is fitted.

   | | predicted s* | measured s* |
   |---|---|---|
   | nord=1 | [1.105, 1.127] | **(1.10, 1.15]** |
   | nord=2 | [0.984, 0.998] | **(0.98, 1.00]** |

   The competing hypothesis "the ported operator is effectively too strong"
   (my old formula plus a fitted uniform factor F=1.28, which also fit the
   earlier wider brackets) requires s* = 1.221 at nord=1 and is **refuted** —
   s=1.15 already fails, at step 104. This is what decides **config-class vs
   port-class**, and it is decided at nord1, not nord2.
2. **Index-exact audit of the CUDA `divg_d` nord loop against
   `sw_core.F90:1359-1423`: 7/7 MATCH.** Not a transcription defect.
3. **The authoritative configuration never forms the pairing.** GEOS sets
   `nord=2` (`FV_StateMod.F90:681`) *and* `d4_bg_top = d4_bg_bot = 0.12`
   (`:683,687`) -> `gamma_d4 = 0.885`, a 2.26x margin. Raw `fv_arrays.F90`
   defaults pair 0.16 with `nord=1` -> 1.638. `(0.16, nord_k=2)` is reachable
   only via a hand-written namelist and is an invention of this harness.
4. **The authoritative config is inside gamma < 2 at every level**, tightest at
   k=0 (gamma = 1.600, 80% of the limit, and *unconditional* because
   `d2_bg_k1 = 0.2` equals the `min(0.20,...)` Smagorinsky cap).
5. **The dt matrix** (section 2) excludes a timestep instability.
6. **The mode is at every `nord_k>0` level at the same rate**, not just k=9:

   | probe | nord_k | growth/substep |
   |---|---|---|
   | k=1 | 2 | +0.076811 |
   | k=5 | 2 | +0.078099 |
   | k=9 | 2 | +0.078682 |
   | k=0 | 0 | +0.025938 (amplitude 8 orders lower) |
   | k=3 | 1 | +0.009441 |
   | k=7 | 1 | +0.014466 |

   This answers the strongest objection to the diagnosis. The *rate* is uniform
   across critical levels; only the *amplitude* is bottom-weighted (2.6e-05 at
   k=9 vs 4.9e-13 at k=1 at the same substep). So "the mode is localized at the
   bottom level" was an artifact of where I probed plus k=9 holding 37.3% of
   column mass, so `delp` reaches zero there first. My original control probe sat
   at k=4, which is `nord_k=0` — a non-amplifying level — and therefore could not
   have detected a sibling critical level.
7. **The authoritative coefficient is stable.** With `d4_bg = 0.12` at the
   critical group and `nord=2` (the authoritative pairing), all legs are bounded
   ramps, none exponential: 8x8 x3 runs 300 calls, 16x16 300 calls, 32x32 60
   calls. The three 8x8 repeats agree to every printed digit at every sample.

Incomplete part of the norm, recorded so it is not lost: on the `nord>0` branch a
**second additive** del-2 + Smagorinsky term acts on the same corner divergence
(`sw_core.F90:1418`, `vort = damp2*delpc + dd8*divg_d`), contributing
`+8*max(d2_bg, min(0.20, dddmp*|div*dt|))*(da_min_c/area_c)`. It does **not**
affect any threshold above, because `d2_bg == 0.0` at exactly the critical
`k%4==1` group and a quiescent seam-free run leaves the Smagorinsky part ~0
(per-group gamma at nord=2, s=1: 1.018 / **2.035** / 1.603 / 0.497, so s=1 is
the unique critical group). It would matter for any other group or a divergent
flow, and it is the only route by which the authoritative GEOS config can exceed
2 (k=9: 1.600 + 0.885 = 2.485, requiring `|divergence| >= 0.3 /s` at dt=10/3 s,
orders of magnitude beyond synoptic).

## 11. Second, independent defect — REPAIRED (section 12)

**Two gravitational constants inside one acoustic substep**, which the
authoritative Fortran cannot do (`nh_core.F90:100` and `nh_utils.F90:348` both
take the single `constants_mod` grav).

- `fv3_vertical_column.hpp:703` hardcoded `T(9.8)`; `riem_solver3_column` had
  **no grav parameter at all**, and `fv3_vertical_kernels.cu` had zero grav
  references — a missing plumbing path, not a mistyped constant.
- `Riem_Solver_C` (`fv3_riem_solver_c.cu:404`) and `gz = grav*zh`
  (`wp_d_contract_cuda.cpp:42`) do use `cfg.grav = 9.80665`.

Because `dm` is the coefficient of the very gamma-law invariant SIM1 inverts
(`nh_utils.F90:1234`), the ratio f = 1.00067857 became a spurious perturbation
pressure `pm2*(f^gama - 1)`: 1.08 Pa at k=0 rising to **73.0 Pa at k=9** —
bottom-weighted — destroying an exact rest state.

Measured, and reproduced to 10 significant digits in host double:
w = **+9.206721742e-02** m/s at the first vertical solve from bitwise-zero wind.

Repair: plumbed `cfg.grav` through `riem_solver3_column`, `sim3_column`,
`sim3p0_column` (r2g = grav/2, r6g = grav/6 per `nh_utils.F90:934-935,1082-1083`)
and the kernel launch, plus a `cfg.grav > 0` validation guard mirroring
`fv3_riem_solver_c.cu:272`. **Numerical semantics changed: YES — a repair toward
the authoritative Fortran, which uses one grav throughout.** The `sim3`/`sim3p0`
sites are IMPLEMENTED + COMPILED only; they are dead at the mandated `a_imp=1.0`.

GPU-verified this milestone with **no** harness knob, `cfg.grav = 9.80665`:

```
it=1 entry .. dgrid_height    w ownmin/ownmax = +0.0 / +0.0   (7 stages, bitwise)
it=1 vertical_solver          w ownmin/ownmax = -2.0492356432e-13 / +3.3736068573e-13
```

An 11-order reduction from +9.207e-02. The rest state is preserved to roundoff.

**This defect is NOT on the instability's causal path**, and my earlier
"coherent two-defect picture" claiming it was is **refuted by my own
pre-registered test**: forcing the two gravities to agree shrank the injected
perturbation by 12 orders (1e-1 -> 1e-13) yet moved failure only 570 -> 569
substeps and left the rate at +0.0685. An exponential at that rate needs 403
extra substeps per 12 orders. What it *does* explain is the bottom-weighted
initial **amplitude profile** in section 10 item 6.

The authoritative **value** of `constants_mod` grav is **NOT VERIFIED** — FMS
`constants_mod` is absent from this filesystem, FMS historically used 9.80,
GEOS/MAPL uses 9.80665. That does not affect the fix: the defect was the
*disagreement*, and using the single configured value is correct either way.

## 12. Third finding — a missing translation layer (no code change)

Nothing under `cuda/` converts `flagstruct` scalars into the ten per-level
damping arrays. `grep -r ke_bg cuda/` returns **zero** hits;
`fv3_gpu_interface_mod.F90` mentions none of `nord_k`/`d2_bg`/`d4_bg`/
`n_sponge`/`ke_bg`. The kernels **consume** the arrays correctly
(`wp_b_dsw_cuda.cu:458,474` for `kgb`/`d_con`, documented at
`wp_b_dsw_cuda.cuh:191`), and the contract already carries all ten
(`fv3_gpu_contract.cpp:56-65`). So this is a missing **translation layer**, not a
missing kernel — and this harness's invented `k%4` table is currently the *de
facto* specification.

**8 of the 10 arrays are STRUCTURALLY unreachable from the authoritative code**,
not merely mis-valued (`nord_k` with three distinct values; `nord_v=1` and
`nord_w=1` at nord=2; `damp_v=0` against a 0.01 floor; `damp_w` as an independent
array; level-varying `damp_t`; `d_con=0.5`; `kgb` nonzero at interior levels).
The full field-by-field deviation table, the authoritative npz=10 column, and the
four load-bearing ordering subtleties are recorded permanently in the
FAITHFULNESS WARNING in `run_small_perf_bench.cpp`. Task #25.

Reference caveat: `fv3_reference/GFDL_atmos_cubed_sphere` is at `ca30e26`, a
**GEOS-ESM fork** merge on `geos/v3.0.0`, not upstream GFDL. Its damping block
has `d4dmp()`, `d_ext()` and `n_zfilter` that upstream lacks. For GEOS
integration this is the correct authority, but it is not upstream GFDL and the
distinction should be stated wherever "authoritative" is claimed.

## 13. Acceptance for performance resumption — NOT MET

| criterion | status |
|---|---|
| 8x8 R8, >=100 calls | PASS at authoritative `d4_bg=0.12` (300 calls, bounded ramp) |
| 16x16 R8, >=100 calls | PASS at authoritative `d4_bg=0.12` (300 calls, bounded ramp) |
| 32x32 R8, >=50 calls | PASS at authoritative `d4_bg=0.12` (60 calls, bounded ramp) |
| 3 independent runs | PASS at 8x8 (bitwise identical trajectories) |
| delp positive, no NaN/Inf | PASS in those legs |
| memcheck | **NOT RUN this milestone** |
| racecheck | **NOT RUN this milestone** |
| initcheck signature | **NOT RUN this milestone** |

The stability windows pass **only** with `FV3_BENCH_D4_SCALE=0.75`, a diagnostic
knob, applied to a table that is still structurally unfaithful in 8 of 10 arrays.
That is evidence the diagnosis is right; it is **not** a stable benchmark
configuration. Performance must not resume on it.

## Blockers

1. Task #25 — the per-level damping translation layer. This is the real fix and
   it also gates faithful GEOS integration, not just the benchmark.
2. Sanitizer legs (memcheck / racecheck / initcheck) not re-run after the grav
   repair.
3. Task #20 — the harness has no per-cell lat/lon, so a globally consistent cube
   IC is not constructible. Deferred by this milestone's own instructions.

## Deferred, do not conflate

Seam-sensitive test (3A.11U section 4); implement E5; implement E11 (must not
mask); k=0 damping row per `dyn_core.F90:721`; D1 (`clamp_nord_cap` caps the
group loop at `ng-1==2`, so `nord_k==3` silently gets zero divergence damping)
and dead-code D2-D5; halo documentation defect at
`fv3_gpu_halo_exchange.cu:44-52` and `fv3_gpu_halo_topology.hpp:289-297`;
`delpc` aliasing deviation (CUDA binds `d_sw`'s first dummy to shared
`FieldId::delpc`, `dyn_core.F90:789` binds it to `vt` scratch — benign under the
current order, order-fragile); task #26 production foot-gun escalation.

## Reproduction

GPU jobs, all `gpu_a100` / `rome` / 1 GPU / 60G / 12 cpus / 1 task, node
`warpa003`. Scripts `dt_3a11v.sh` .. `dt6_3a11v.sh`, `loc_3a11v.sh` (scratch, not
committed): jobs 58095382, 58095438, 58095540, 58095728, 58095838, 58096002,
58096246. Analysis: `an_3a11v.py`. Rebuild/test: `rebuild_verify_3a11v.sh`.

Build state after the repair, all three trees, `make` exit 0, 0 compiler errors:
`build_cuda_r8` 28 targets, `build_cuda_r4` 28, `build_cuda_r8_perf` 29. CTest on
a login node: 20 listed / 11 passed / **9 skipped (GPU-requiring)** / 0 failed.
That is CPU-visible coverage only; it is **not** 20/20.

Caveat: jobs 58095382-58096002 predate the grav repair and must not be compared
bit-for-bit against post-repair runs. Job 58096246 postdates it (binary mtime
2026-08-27 10:46:06, verified in the job log).

## Next recommended phase

Task #25, then re-run the full section 13 campaign including sanitizers on the
faithful table, then Prompt 6 authoritative numerical validation. Performance
remains blocked.
