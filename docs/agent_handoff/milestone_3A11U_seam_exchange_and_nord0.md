# Milestone 3A.11U — Residual Multi-Step Instability: Staggered Seam Exchange / nord==0 Path

Branch: `cuda/gpu-bringup`
Parent commit at start: `774ab59` ("Fix H_GZ_FIRST ordering and checkpoint multistep diagnostics")
Predecessor handoff: `docs/agent_handoff/milestone_3A11T_pe_peln_layout_and_multistep.md`

---

## Headline

The residual multi-step instability is **not** a port defect. It is caused by an
**invalid synthetic initial condition** in the benchmark harness: the horizontal
structure of both the winds and the scalars is a function of the **per-tile local
index**, seeded identically on all six tiles. That is not a single-valued field on a
cubed sphere. The IC therefore carries genuine discontinuities across all twelve panel
edges, the seam exchange correctly transports the neighbour's differing value inward,
and the resulting real ~22 m/s seam shear drives `delp` monotonically toward zero until
`log()` of a negative thickness produces the first NaN.

This is §7 option **C** (invalid synthetic state), not §7 option A (missing staggered
seam exchange) and not §7 option B (nord==0 semantics). Both A and B were investigated
to closure and both are **ruled out as the cause** — though A did turn up three genuine
authoritative-fidelity gaps, recorded below, which must be fixed on their own merits.

The proof is quantitative rather than circumstantial: see "The 0.069% confirmation".

---

## The 0.069% confirmation

The mechanism was not merely observed, it was **predicted numerically before being
measured**, from cube topology plus thermodynamics, with both endpoints independently
derived.

Ring-1 `pkc` excess scales as `r^gama - 1`, with

```
r    = (1 + hp_owned) / (1 + hp_ghost)
gama = 1 / (1 - akap) = 1 / (1 - 2/7) = 1.4
```

Applying `seam_along_index(t,n,reversed)` (`fv3_gpu_halo_topology.hpp:356-360`) and the
parity rule "odd tiles rotate on WEST and NORTH, even tiles on EAST and SOUTH"
(`fv3_gpu_halo_topology.cpp:35-36`):

| tiles | aligned edge | destination | source | `hp_owned - hp_ghost` | max at | predicted site |
|---|---|---|---|---|---|---|
| odd (1,3,5) | SOUTH | `(i=t, j=0)` | `(i=t, j=8)` | `0.034026 * sin(0.35(t+2))` | `t=2` | **(2,0)** |
| even (2,4,6) | WEST | `(i=0, j=t)` | `(i=8, j=t)` | `0.019900 * cos(0.35(t+2))` | `t=1` | **(0,1)** |

Predicted odd/even excess ratio: **3.45464**
Measured: `4182.3132 / 1211.4443` = **3.45227**
Agreement: **0.069%**

Both endpoints — the cube geometry and the thermodynamics — are fully accounted for.
Nothing is left over for a kernel defect to explain.

---

## The discriminator experiment (Slurm job 58095070)

`FV3_BENCH_HORIZ_SCALE` was added to the harness to scale every local-index horizontal
perturbation. Default is `1.0`, so the default configuration is bit-unchanged.

| leg | configuration | `delp` distinct hashes | ring-1 `pkc` excess | 50 calls |
|---|---|---|---|---|
| 8x8 | `WIND=0 HORIZ=0` | **1 / 150** | **exactly 0.0 at all 900** | clean |
| 16x16 | `WIND=0 HORIZ=0` | **1 / 150** | **exactly 0.0 at all 900** | **clean** |
| 16x16 | `HORIZ=0`, default winds | 150 / 150 (expected) | up to `1.36e4` | **clean** |
| 16x16 | untouched default | 59 / 150 | up to `5.00e3` | **FAIL step 17** |

Two things follow.

**The regression guard is exact.** The untouched default reproduces the known 3A.11T
failure bit-identically: step 17, tile 4, `it=2`, `cgrid_vertical_solver`, `gz`,
`nan/inf=10/5819`, hash `d36ada74b033457d`, 105572 owned-nan lines, 66 `RIEM3-BAD`.
The harness edits changed the default configuration by not one bit.

**Leg 3 is the new information.** With the scalar discontinuity removed but the *full
default winds retained*, 16x16 survives 50 calls. So the step-17 failure requires
**both** seam-discontinuous winds **and** seam-discontinuous scalars.

### The invariant — twice overstated, and now stated correctly

This invariant was written down twice in too strong a form. Both versions are recorded
because the pattern of error matters more than either version.

**Version 1, withdrawn:** "every field stays bit-constant forever at rest". The uniform
legs show `pkc` with 150 distinct hashes over 900 tile-checkpoints. But 900/150 = exactly
6 per hash, i.e. all six tiles are bit-identical at every checkpoint and the field
evolves only *vertically*. That is admissible: a horizontally uniform column may still
evolve vertically if it is not in exact nonhydrostatic balance, and `pkc` legitimately
does.

**Version 2, also withdrawn:** "`delp` is bit-constant, unconditionally". This held
exactly for 50 steps at 16x16 and 32x32 (1 distinct hash over 150 checkpoints) — and then
**failed at 100 steps**:

| leg | first `delp` hash change |
|---|---|
| 8x8, 100 iters | checkpoint 183 = **step 60, `it=3`** |
| 16x16, 100 iters | checkpoint 175 = **step 58, `it=1`** |

The drift is real but tiny and is **not** the instability: over 100 steps the printed min
and max *values* never move off `5.848931925e+02` / `3.690426555e+04`, no cell goes
nonfinite, and min owned `delp` stays exactly `584.893193`. What moves is the
**argmin/argmax** — `ownmin ...@72` → `@85`, i.e. `(3,1,0)` → `(16,1,0)`. Because the
field is horizontally uniform, very many cells are tied at the min, so the argmin is
arbitrary among ties and moves as soon as a few cells shift by one ulp.

It is **not** a race either: the three independent 8x8 100-step runs produced
**byte-identical hash sequences** (a==b==c), and racecheck reports 0 hazards. So it is
deterministic arithmetic, and the "exactly zero" argument had a hole.

**The hole, located by measurement rather than argued around: the winds are never exactly
zero.** `FV3_BENCH_WIND_SCALE=0` zeroes the wind *seed*, but the column is deliberately
not in exact nonhydrostatic balance, so it rings vertically from step 0 — `w` is already
`9.2e-02` at step 0 and oscillates in a bounded range `3e-3 .. 6e-1`. Owned `|u|` is
`4.7e-10` at step 0 and grows with an amplifying envelope:

```
step  1   9.98e-10        step 58   5.76e-06
step 20   5.90e-09        step 60   9.12e-06
step 50   9.20e-12        step 65   2.87e-05
step 55   2.90e-06        step 80   8.93e-04
                          step 90   8.85e-03
```

(Oscillatory step to step, hence the non-monotone samples; the *envelope* is what grows,
roughly an order of magnitude per ten steps in the late phase.) `delp`'s update is strict
flux form, so a nonzero wind of any magnitude makes the flux divergence nonzero. `delp`
stays bit-constant only for as long as the amplified roundoff remains below `delp`'s ULP,
which is ~58-60 steps. The onset is a threshold crossing, not an event.

The invariant is therefore **bounded in time**, and only this much is safe to assert:

> With all winds and `w` exactly zero *at the instant considered*, all scalars functions
> of `k` alone, uniform metrics and `phis = 0`: every horizontal difference is exactly
> `0.0`; `p_grad_c`'s numerator is `A*B + (-A)*B` and IEEE subtraction is exactly
> antisymmetric, hence exactly `0.0`; every flux is exactly `0`. Therefore `delp` does
> not change on that substep, ring-1 `pkc` excess is exactly `0.0`, and all six tiles
> stay bit-identical. This does **not** extend across many substeps, because the
> unbalanced column does not stay at rest.

Ring-1 `pkc` excess did remain **exactly 0.0 at every checkpoint of every uniform leg**
(0 of 1800 nonzero at 8x8/100 and 16x16/100; 0 of 900 at 16x16/50 and 32x32/50). That
part of the invariant survives unqualified.

### NEW OPEN ITEM: an amplifying mode at k=9, present with a valid uniform IC

The growing `|u|` envelope above is localised: the owned extremum sits at
**`(4,5,k=9)`** — the bottom level — for 1869 of 2500 checkpoints (k=4 for 627, k=0 for
only 4). This matters for two reasons.

1. **It is not seam-driven.** The uniform legs have no seam discontinuity at all, so this
   mode is a property of the discrete operator plus this column, not of the invalid IC.
2. **It coincides with the default run's failure site.** §6 found the default 16x16
   `delp` degradation pinned at `(16,1,k=9)` — the same bottom level.

That suggests a unified reading: **an amplifying mode at k=9 exists independently, and
the invalid IC sets its initial amplitude and therefore the time to failure** (~17 steps
from a ~22 m/s seam shear, versus >100 steps from roundoff). If that is right, fixing the
IC alone raises the time to failure without removing the mode, and the root cause has two
components rather than one.

This is **not yet established** — it is a measured lead, and it is the highest-value next
investigation. Note k=9 is damped (`k%4 = 1` → `nord_k=2`, `nord_v=2`, `damp_v=0.10`,
`d4_bg=0.16`, `d_con=1.0`), so an absent-damping explanation does not fit it as written.

### Why the uniform leg is not the fix

At `HORIZ_SCALE=0 WIND_SCALE=0` all six tiles are identical and every exchanged quantity
is uniform. The leg is therefore **blind** to neighbour-mapping, stagger and sign
defects — precisely the class §4 exists to catch. Reporting its pass as multi-step
stability restored would be reducing the test's stringency, which §9 forbids.

---

## §1/§2 — `mpp_get_boundary` traced authoritatively

Authoritative tree: `GFDL_atmos_cubed_sphere`, `geos/v3.0.0`, verified at
`ca30e26305fda43831095334c02a3f5e0b8c0c45`.

**E11** is a shared staggered seam **line** reconciliation, `dyn_core.F90:1091-1105`:

- fields: D-grid `u` (NORTH-staggered) and `v` (EAST-staggered);
- halo width: **ring `h=0` only** — the shared seam line itself, not a ring of ghosts;
- timing: **only when `it == n_split`**, mutually exclusive with the `it /= n_split`
  D-grid halo update at `:1108-1109`;
- buffers: `real ebuffer(npy+2,npz)`, `real nbuffer(npx+2,npz)` — **2-D**, declared at
  `dyn_core.F90:250-251`;
- consumer: `nh_p_grad` / the pressure-gradient update immediately following;
- its own comment scopes it to *"Prevent accumulation of rounding errors at overlapped
  domain edges"*.

### Mapping table

| Fortran exchange | CUDA halo group/service | implemented? | same staggering? | same transform? | same timing? | consumer protected? |
|---|---|---|---|---|---|---|
| E5 `q_con,cappa,delp,pt`, completed `dyn_core.F90:476` before `c_sw`/`Riem_Solver_C` on `it==1` | none; `H_DSCALAR` carries the same four fields (`fv3_gpu_halo_exchange.hpp:220-229`) | **NO** | n/a (CENTER, matches) | n/a | **NO** — resolves at `fv3_gpu_runtime.cpp:1569`, *after* `dgrid_dynamics` | **NO** |
| E11 `mpp_get_boundary(u,v)`, ring `h=0`, `it==n_split` | none | **NO** | — | — | — | **NO** (`pressure_u_kernel` `wp_d_cuda.cu:459-461`, `pressure_v_kernel` `:467-469`) |
| `i_pack(5)` gz, `it==1` | `H_GZ_FIRST`, `fv3_gpu_runtime.cpp:1561` | YES | YES (CENTER scalar, `fv3_gpu_halo_exchange.hpp:191-199`) | YES | YES | YES |

### §2 verdict: a NEW halo role/group is required

E11 is **not representable by the current halo descriptor**:

- `halo_width` is a **ring count**, and every consumer maps it to `h = blockIdx.z + 1`,
  so ring `h=0` is **structurally unreachable**;
- `slot_offset` uses `(h-1)`, so `h=0` would index **negatively**;
- there is **no per-field edge mask** (E11 exchanges only specific edges);
- there is **no send-only / receive-only asymmetry** (E11 is an ownership
  reconciliation, not a symmetric ghost fill);
- **no `it == n_split` gate exists anywhere in the port**.

The existing index algebra does, however, already generalise correctly to `h=0` — so the
work is descriptor and gating work, not new index mathematics. Separately, FMS's
`mpp_do_get_boundary` rotation/ownership convention is **not traced anywhere in the
port** and must be before implementation.

### E11 is ruled out as the instability cause

On magnitude grounds. Its own comment scopes it to *rounding* error accumulation; the
measured seam disagreement in the failing configuration is **~22 m/s**. A rounding-level
reconciliation cannot be the origin of a 22 m/s discontinuity, and implementing it here
would mask the real cause — which §9 forbids.

---

## §5 — nord==0 path reconciled: MATCH (verdict PARTIAL)

Side-by-side comparison of the authoritative `nord_k == 0` branch against the CUDA path
gives a **term-by-term match**:

- `dddmp * abs(delpc * dt)` — identical;
- `rarea_c` applied **before** the limiter — identical;
- `da_min_c` **outside** the `max` — identical;
- the `0.20` cap — identical;
- `da_min` in `dd8` — identical;
- all five index boxes L1–L5 — identical;
- all four corner sign corrections — identical;
- the full update ordering — identical.

Verdict is recorded as PARTIAL rather than MATCH only because the reconciliation covers
the `nord_k == 0` branch as instructed and does not extend to the `nord_k >= 1` group
loop. No change was made to this branch, per §5's instruction.

**Ruled out as the instability cause.**

---

## §6 — first `delp` degradation, measured

`dgrid_dynamics` is the **only** stage that changes `delp`; every other stage leaves
min/max bit-identical.

16x16 owned minimum trajectory:

```
584.786  ...  561.16 (step 14)  ->  332.1 (15)  ->  181.6 (16)  ->  -505.2 (17)
```

At step 17 SIM1 takes `log()` of a negative thickness and produces the first NaN.

- **First degradation stage:** `dgrid_dynamics` (D-grid flux-form `delp` update).
- **First degradation location:** owned maximum pinned at **tile 4, (i=16, j=1, k=9)** —
  the owned SE cube corner, bottom level — growing monotonically every substep.

The 8x8 leg that **passes** 50 calls has the **identical signature** at `(8,1,9)`:
36971 → 38536, +4.2%, monotone, never recovering, at a nearly identical absolute rate.
So **the 8x8 pass is survival headroom, not stability.** This restates and confirms the
retraction already recorded in 3A.11T.

---

## §7 — k=0 synthetic state: STATE VALID, DAMPING ROW INVALID

k=0 satisfies every authoritative invariant:

- `delp = 584.893 Pa > 0`;
- `pe(1) = ptop = 1000 Pa`;
- `delz = -3773.68 m`;
- SIM1-balanced.

However the **damping row** at k=0 is not authoritative. The harness gives k=0
`damp_v=0, damp_w=0, damp_t=0, nord_v=0, nord_k=0, d_con=0, kgb=0` — i.e. it damps the
model top *less* than the interior. Authoritative FV3 never does this:

- `dyn_core.F90:743-751` pairs every `nord_* = 0` with a **strictly positive**
  coefficient;
- `dyn_core.F90:721` floors `damp_vt(k) = max(0.01, min(vtdm4, d4_bg_top/4.0))` at
  **every** k.

**The reasoning previously given for dismissing this row was wrong and is withdrawn.** It
was: "damping of exactly zero wind is exactly zero, so the k=0 row cannot contribute a
single bit in the uniform legs." That rests on the winds being exactly zero, which the
measurement above shows they never are — owned `|u|` is `4.7e-10` at step 0 and grows.
Damping of a nonzero wind is nonzero, so the argument does not hold.

The row is still **not implicated**, but for a different and now measured reason: the
amplifying mode sits at **k=9**, not k=0. The owned `|u|` extremum is at k=9 for 1869 of
2500 checkpoints and at k=0 for only 4. So the absent k=0 damping is not where the growth
lives.

It remains a harness fidelity defect to be fixed on its own merits, justified from
`dyn_core.F90:721`, and verified not merely to delay the failure (§9).

Per §7's instruction, k=0 was **not** "fixed just because it is the failure level".

---

## Genuine port defects found (none of them the instability driver)

### E5 missing — the strongest of the three

The `q_con / cappa / delp / pt` halo started at `fv_dynamics.F90:620,622,626,627`
(`i_pack(1)`) and completed at `dyn_core.F90:476` on `it == 1`, **before** `c_sw`
(`:517`) and `Riem_Solver_C` (`:583`), has **no CUDA group**. `H_DSCALAR` carries the
same four fields but resolves at `fv3_gpu_runtime.cpp:1569`, *after* `dgrid_dynamics`.

This asymmetry is exactly what confined the observed ring-1 excess to `it=1`: `gz` **is**
exchanged before the stage (`:1561`) while `delpc`/`ptc` is not, so the two columns are
mutually inconsistent on `it=1` only. Silent under this IC because it is uniform per
tile; **would bite with a real IC.**

### E11 missing

See §1/§2 above. Authoritative-fidelity item; ruled out as the cause.

### `gz` seed units wrong by `grav` — FIXED this milestone

The harness seeded `h.gz = grav * zh_col` while `h.zh = zh_col`. `gz` carries **two
different unit conventions inside a single substep**:

| site | expression | units |
|---|---|---|
| `dyn_core.F90:433-455` (`it==1` fill) | `gz(npz+1)=zs`; `gz(k)=gz(k+1)-delz(k)` | **metres** (no `grav`) |
| `update_dz_c` (`nh_utils.F90:188-203`) | `ws=(zs-gz(km+1))*rdt` | metres |
| `Riem_Solver_c` tail (`nh_utils.F90:421-426`) | `gz(km+1)=hs`; `gz(k)=gz(k+1)-dz2(k)*grav` | **grav·height on output** |

The last is per its own comment, *"Compute Height * grav (for p-gradient
computation)"*. The seed must match the **first** convention, i.e. metres, matching `zh`
(`dyn_core.F90:552-560` copies `zh = gz` over the full box on `it==1`).

The old `grav` factor was mostly invisible because the `it==1` fill overwrites `gz`'s
interior and the edge exchange overwrites rings 1-3 — but **neither reaches the four
diagonal halo corner blocks**, so exactly those cells kept a value 9.8x too large. This
is the root of the previously-unexplained `gz`/`zh` x`grav` corner recurrence, which had
only been characterised as a device-residency artefact.

Numerical semantics changed: **NO** for every cell any consumer reads in this
configuration; the corrected cells are ones the authoritative Fortran never reads either,
because it reallocates `gz` per `init_step`.

### D1 latent — `clamp_nord_cap` gating hole

`clamp_nord_cap(opt.nord, b.ng)` (`wp_b_dsw_cuda.cu:936`, `:590-593`) caps the group
loop at `ng-1 == 2`, so `nord_k == 3` silently receives **zero** divergence damping while
still returning `FV3_GPU_SUCCESS`. Not live at `cfg.nord = 2`, but it fails silently
rather than loudly, and `docs/architecture/dyn_core_cuda_supported_config.md:71` cites
dead code as evidence that `nord=1-3` is supported.

### Smaller items

- `in_box` lacks a `k` bound test (`wp_b_dsw_cuda.cu:46-52`).
- `lim_fac` absent from the CUDA scalar transport path — inert at `|hord| = 6`, where
  both sides hard-code `3.*abs(b0) < abs(bl-br)`.
- `update_dz_c_copy_gz`'s launch grid never reaches `i = isd` or `j = jsd`, so `gz2`'s
  outermost ring is stale versus Fortran's `is-ng..ie+ng`. Unread in this configuration.

### Documentation defect — contradicted with specifics

Three inaccuracies at `fv3_gpu_halo_exchange.cu:44-52` and
`fv3_gpu_halo_topology.hpp:289-297`:

1. the "never into the field array itself" claim — E11 writes into the field;
2. the "neither overwrites the other" claim;
3. "1-D buffers" — `dyn_core.F90:250-251` declares `real ebuffer(npy+2,npz)`,
   `real nbuffer(npx+2,npz)`, which are **2-D**.

`docs/agent_handoff/milestone_3a10_gpu_halo_exchange.md:86` is already correct.

---

## Source changes made this milestone

All confined to `cuda/dyn_core/benchmarks/run_small_perf_bench.cpp`, i.e. **the harness
only**. No numerical kernel was touched.

1. `FV3_BENCH_HORIZ_SCALE` knob plus `bench_horiz_scale()`, scaling every local-index
   horizontal perturbation: `horiz_pert` (reaching `pt`, `delz`, `zh`, `gz`), the metric
   `horiz` (reaching `area`, `rarea_c`, `f0`), `heat_source`, `ptc`, `z_rat`, `omga`.
   Default `1.0` — the default configuration is bit-unchanged, verified.
2. The comment block above `bench_wind_scale()` corrected. It previously presented a
   two-way PORT DEFECT / STATE-DRIVEN dichotomy that was **incomplete**, and reading it
   as complete produced a wrong conclusion that had to be retracted. The correction is
   written into the source so the next reader cannot repeat it.
3. `gz` seed units corrected from `grav * zh_col` to `zh_col`.

---

## Method failures worth carrying forward

**A port defect was declared proven on an unverified premise — twice in this milestone.**
The premise was that `WIND_SCALE=0` gives a horizontally uniform state. It does not: it
gives zero *wind* and nothing more, because the scalars carry their own independent
local-index structure. An entire `H_GZ_FIRST` rotation-bug hypothesis was built on it and
had to be retracted outright.

Because this was the **second** instance of the same class of error (the first being
`make_tile_asymmetric`), the response was not to re-read the one line expected to matter
but to run a **complete grep census** of `std::sin|std::cos|horiz_pert|_pert|perturb`.
That found **six** unscaled local-index sources, not one.

> Never assert a state property from the lines you expect to control it. Enumerate every
> writer.

The same class of error recurred a third time, in the *analysis* rather than the premise:
a probe built to answer "are the winds exactly zero?" used fixed `awk` column indices
`$11`/`$12` for `min=`/`max=`. The real positions are **`$10` = `min=`, `$11` = `max=`,
`$12` = `ownnan=`**, and **post-loop-stage lines carry no `it=` token so every column
shifts by one**. The probe printed the self-contradictory line
`u NONZERO from step=0 ... first magnitude 0.000000e+00` and a wind trajectory table of
all zeros — which, taken at face value, would have "confirmed" the exactly-zero premise
that was in fact false. Re-extracting the same saved log by **matching tokens by name**
(`/^min=/`, `/^ownmax=/`, and identifying the field as the token before `nan/inf=`)
reversed the conclusion. Parse `[FV3DBG]` by token name, never by column number; and treat
a self-contradictory output line as a parser bug, not as data.

Other traps confirmed live, worth preserving:

- Tiles are numbered **1–6** in `[FV3DBG]`, not 0–5. An `awk` predicate of `tile=0`
  silently reports "0 distinct hashes over 0 checkpoints" and reads as a clean pass.
- initcheck can write its report to **stdout** and then hang for tens of minutes in
  teardown without ever emitting `ERROR SUMMARY`. Collect the signature set from the
  report itself rather than waiting for a summary line that may never come.
- Signed zero: `ws * (-7.0 + ...)` at `ws=0` yields `-0.0`, a different bit pattern from
  `+0.0`, so **hashes differ while min/max stay 0**. Use extrema to discriminate zero.
- `[FV3DBG]`/`[BENCH]` go to **stderr**; `[RIEM3-BAD]`/`[RIEMC-PRE]` device `printf` go
  to **stdout**.
- `grep nonfinite` matches nothing — the field is `nan/inf=<n>/<total>`.
- `[FV3DBG]` prints **unconditionally** per field per stage under
  `FV3_GPU_DEBUG_FAIL=1`; the `fprintf` is outside the `bad_count > 0` guard.
- Never classify a run from its first reported line: the 16x16 failure's first line has
  `ownnan=0` and reads as benign, while the run accumulates **105572** lines with
  `ownnan >= 1`.
- `compute-sanitizer` is **not on PATH** under this module set. Use
  `/usr/local/other/nvidia/hpc_sdk/Linux_x86_64/24.7/cuda/12.5/compute-sanitizer/compute-sanitizer`
  with an `[ -x ]` precondition, or all three tools exit 127 and read as clean.
- `--print-limit` truncates initcheck at 100 hits, so a raw count is a **sample, not a
  census**. Only the **signature set** is comparable across runs.
- `dz_worst` in `[RIEM3-BAD]` is seeded from an already-NaN output and always prints
  `nan`; `dz_nonneg` is the load-bearing field. `RIEM3-BAD lines: 66` is the print budget
  saturating and carries **no** information.

---

## Remaining work

1. **Investigate the amplifying k=9 mode** (see the open item above). Highest value,
   because if it is real then item 2 alone will not restore stability — it will only move
   the failure later. Suggested discriminator, the same one that worked in 3A.11T: run the
   uniform leg to 200-300 steps, fit the envelope growth rate, and test whether the rate
   depends on `n_split`. A rate that moves with `dt` is a CFL/marginal-stability property
   of the configuration; a `dt`-insensitive rate is a port defect. Then check whether the
   mode survives with `d_con=0` and with `damp_v` scaled at k=9, and whether the
   authoritative Fortran shows the same ringing for the same column — an unbalanced column
   may legitimately ring.
2. **Rebuild the benchmark IC on a globally consistent cube coordinate** so it is
   single-valued across seams *and* retains seam sensitivity. This is the fix for the
   proven component. Currently blocked: the harness has no real cube geometry — it
   fabricates uniform 100 km metrics (`dx = dy = 1e5`, `area = 1e10`) with no per-cell
   lat/lon.
2. **§4 seam-sensitive test** — asymmetric per-tile values; all six tiles; all four
   edges; reversed seams; vector component swap/sign; exact staggered destination
   indices; k=0 explicit; must **fail** if the seam exchange is disabled. Note the
   uniform leg is explicitly blind to this class.
3. **Implement E5** — the missing pre-`c_sw` scalar halo. Now has an observable
   consequence and should be treated as its own port defect.
4. **Implement E11** — requires a new halo role/group carrying ring index 0, a per-field
   edge mask, send/receive asymmetry, an `h=0`-safe buffer offset, `it == n_split`
   gating, and a traced `mpp_do_get_boundary` rotation/ownership convention. Must **not**
   be used to mask the instability (§9).
5. Fix the k=0 damping row from `dyn_core.F90:721`; verify it does not merely delay the
   failure.
6. Fix or document D1, the dead code D2–D5 in `wp_b_kernels.hpp`, and the supported-config
   claim that cites dead code.
7. Fix the halo documentation defect.

---

## §12 Completion report

| field | result |
|---|---|
| E11 `mpp_get_boundary` mapping | **COMPLETE** |
| Missing staggered seam exchange | **CONFIRMED missing — but RULED OUT as the instability cause** |
| Staggered exchange implementation | **NOT REQUIRED for this instability** (authoritative-fidelity item; implementing it here would mask the real cause, §9) |
| Seam-sensitive test | **NOT RUN** (§4 deferred; the uniform leg is blind to this class) |
| nord==0 Fortran/CUDA reconciliation | **PARTIAL** — term-by-term MATCH on the `nord_k==0` branch; `nord_k>=1` group loop not covered |
| First `delp` degradation stage | `dgrid_dynamics` (D-grid flux-form `delp` update) |
| First `delp` degradation location | tile 4, `(i=16, j=1, k=9)` — owned SE cube corner, bottom level |
| k=0 synthetic state | **VALID** state; damping row **INVALID** but not causal |
| 16x16 R8 50-call | **FAIL** on the untouched default (step 17, bit-identical to 3A.11T); **PASS** only at `HORIZ_SCALE=0`, a weakened configuration |
| 8x8 R8 100-call | **PASS** at `HORIZ_SCALE=0`, three independent runs, byte-identical hash sequences; `delp` positive throughout (min `584.893193`) |
| 32x32 R8 50-call | **PASS** at `HORIZ_SCALE=0`; `delp` bit-constant (1 hash / 150), ring-1 `pkc` excess exactly 0.0 |
| memcheck | **PASS** — 0 errors |
| racecheck | **PASS** — 0 hazards displayed, 0 errors, 0 warnings |
| initcheck | **ONLY PREVIOUSLY CLASSIFIED DEAD READS** — single signature `dgrid::copy_corners_kernel` (via `launch_copy_corners_cuda` ← `launch_del6_vt_flux_cuda` ← `launch_d_sw_cuda`), identical to the baseline recorded at 3A.11T line 40. **No new signature.** All 100 reported hits carry that one signature and saturate the default `--print-limit` of 100, so only the signature set is claimed unchanged, not the count; the run did not reach an `ERROR SUMMARY` line |
| R8 CTest | **20/20 PASS** (rebuilt after the harness change) |
| R4 CTest | **20/20 PASS** (rebuilt after the harness change) |
| Root cause | **Primary, proven: invalid synthetic IC.** Horizontal structure of both winds and scalars is a function of the per-tile local index, seeded identically on all six tiles, hence not single-valued on a cubed sphere. Genuine discontinuities across all twelve panel edges; the seam exchange correctly transports them inward; the resulting real ~22 m/s seam shear drives `delp` monotonically negative until `log()` of a negative thickness NaNs. Confirmed to **0.069%** against an a-priori prediction. **Second component, measured but not established: an amplifying mode at k=9** that is present even with a valid uniform IC and no seam discontinuity at all, and which sits at the same bottom level as the default run's `delp` degradation. If real, the IC sets the mode's initial amplitude and hence the time to failure rather than being the whole cause. Recorded as the highest-value next investigation. |
| Numerical algorithm changed | **NO** |
| Performance milestone | **STILL BLOCKED** |
| Numerical validation | **NOT RUN** |

The §11 gate is **not** met. The §8-mandated known failing case is the untouched default
16x16 at `n_split=3`, and it still fails at step 17. The passes above are at
`HORIZ_SCALE=0`, which is a *weakened* configuration; presenting them as stability
restored would be reducing the test's stringency, which §9 forbids.

**MULTI-STEP STABILITY STILL BLOCKED**
