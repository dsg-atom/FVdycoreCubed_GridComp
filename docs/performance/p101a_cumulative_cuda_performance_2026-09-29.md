# Prompt 101A — Consolidate End-to-End CUDA Performance Improvement So Far

## 0. Task

Quantify how much faster the current production CUDA configuration
(PT/DELZ + REMAP_UV + REMAP_Q + REMAP_PRESSUREVARS, all CUDA) is vs. the
best available pre-port/earlier-production baseline, under a comparable
C512 configuration. No source changes. No commit/push. No memory-cleanup
or multi-GPU work. No R4. No Nsight Compute.

Constraint respected throughout: State_To_FV, TRACER_2D, and
REMAP_TOTALENERGYCON remain Fortran (production default) in every run
below — none were toggled or enabled.

## 1. Stage 0 — Historical Chronology (C512, DYN_CORE/fv_timing family)

| Prompt | Job | Node | Binary sha256 (first 8) | TOTAL ms/call | FV_DYN_LOOP ms/call | REMAPPING ms/call | Active CUDA remap ports | Notes |
|---|---|---|---|---|---|---|---|---|
| 48 | 58468244 | warpa004 | (Fortran/CUDA base dyncore, pre-P49) | not measured | not measured | not measured | none | Only `DYN_CORE` (fv_timing) measured: Fortran 6374.02, CUDA 6373.86 ms/call, ratio 1.00. `DYN_CORE` is a **sibling** of `REMAPPING` (P69 finding) — structurally blind to all 4 remap ports. |
| 69 | 58530987 | warpa004 | `17a60c19...` | A 4930.87 / B 4573.08 | A 2141.28 / B 1786.66 | A 1050.67 / B 696.43 | PT/DELZ only (A=Fortran, B=CUDA) | PT/DELZ closure paired A/B, 3 pairs. TOTAL improvement 7.26%, FV_DYN_LOOP 16.56%, REMAPPING 33.72%. |
| 71 | 58533089 | warpa004 | (pre-REMAP_Q/PRESSUREVARS) | A 15288.87 / B 15136.63 | A 2378.60 / B 2243.83 | A 715.87 / B 563.57 | PT/DELZ CUDA (default) + REMAP_UV toggled (A=Fortran, B=CUDA) | REMAP_UV closure paired A/B, 3 pairs. Different TOTAL timer basis than P69 (whole-job-normalized, not directly comparable digit-for-digit) — improvement percentages are the portable figures: TOTAL 1.0%, FV_DYN_LOOP 5.7%, REMAPPING 21.3%. |
| 89 | 58570647 | warpa003 | `eabd1c54...` | mean Δ 2.654% / median 1.722% | mean Δ 5.95% | mean Δ 12.67% | PT/DELZ+REMAP_UV CUDA (default) + REMAP_Q toggled (A=Fortran, B=CUDA) | REMAP_Q closure paired A/B, 5 pairs. REMAP_Q direct speedup mean 3.247x. 60-call confirmation (job 58571255): TOTAL 1.41%, FV_DYN_LOOP 2.49%, REMAPPING 8.02%, REMAP_Q 69.8%/3.31x. |
| 90 | 58593086 | warpa004 | (pre-PRESSUREVARS) | 3932.4 | 1706.8 | 499.6 | PT/DELZ+REMAP_UV+REMAP_Q CUDA (default), PRESSUREVARS Fortran | Single-arm hotspot profile, 60 calls. Intermediate state — 3 of 4 ports CUDA. |
| 93 | 58617749 (5-pair) / 58618735 (60-call) | warpa003 | `3d4364e6...` | mean Δ 1.805% / median 1.633% | +4.086% (60-call) | +10.887% (60-call) | PT/DELZ+REMAP_UV+REMAP_Q CUDA (default) + PRESSUREVARS toggled (A=Fortran, B=CUDA) | REMAP_PRESSUREVARS closure paired A/B, 5 pairs + 1 long pair. Direct speedup mean 3.498x (5-pair), 3.194x (60-call). |
| 94 | 58619479 | warpa003 | `b482b647...` | **3836.74** | **1619.80** | **414.05** | **All 4 CUDA (current production defaults)** | Single-arm post-PRESSUREVARS hotspot profile, 60 calls. `DYN_CORE` 845.85, `TRACER_2D` 453.32 (Fortran), `REMAP_TOTALENERGYCON` 150.16 (Fortran, now dominant `REMAPPING` sub-timer). |
| 96 | (reuses P94 data) | warpa003 | `b482b647...` | 3836.74 | 1619.80 (fully explained by DYN_CORE+TRACER_2D+REMAPPING) | 414.05 | same as P94 | Decomposition only — no new run. Found `DYN_EPILOGUE` (688.67 ms/call, MAPL, Fortran, no MPI) as the largest previously-unexamined cost, **outside** `FV_DYN_LOOP`. |
| **101A** | **58638414** | **warpa010** | **`2339366b...`** | **A 5095.05 / B 4419.08 (mean)** | **A 2299.57 / B 1633.59 (mean)** | **A 1077.49 / B 415.92 (mean)** | **All 4 toggled together** (A=all 4 Fortran, B=all 4 CUDA=current production) | **This task's own clean same-node A/B — see §3.** 5 pairs, alternating order. |

**Key structural finding carried forward from P69/P96**: `DYN_CORE`
(fv_timing) is a sibling of `REMAPPING`, not a parent — Prompt 48's
`DYN_CORE`-only comparison cannot detect any of the 4 remap ports'
effect at all, regardless of node.

## 2. Stage 1 — Baseline Comparability Assessment

**Conclusion: no single historical run is directly comparable to the
current production configuration.** Two independent, compounding
confounds:

1. **Metric mismatch.** Prompt 48 (the only true same-node,
   alternating-order, Fortran-vs-CUDA C512 comparison) measured only
   `DYN_CORE`. Per Prompt 69's finding (confirmed again in Prompt 96),
   `DYN_CORE` is a **sibling** of `REMAPPING` in the timer tree, not its
   parent — none of PT/DELZ, REMAP_UV, REMAP_Q, or REMAP_PRESSUREVARS
   (which all execute inside `REMAPPING`) can move Prompt 48's
   `DYN_CORE` number in either direction. Prompt 48 is therefore
   structurally blind to the exact effect this task must quantify.
2. **Node mismatch.** Prompt 48 ran on `warpa004`; the current-production
   reference (Prompt 94, job 58619479) ran on `warpa003`. Per
   `CURRENT_STATE.md` §8, node-to-node A100 variance at C512 exceeds 2x
   in absolute terms — an uncontrolled cross-node comparison of any
   timer would be unreliable even if the metric mismatch above did not
   exist.

No other historical run pairs a zero-remap-port state against the
current all-4-CUDA state on the same node with the same binary lineage.
Per Stage 1's own instruction ("If no single historical run is fully
comparable, state that clearly"), this is now stated: **Stage 3's
historical-comparison table below is reported for context only and is
NOT used as the primary answer.** Stage 4's clean same-node A/B (§3) is
the primary answer.

## 3. Stage 2 — Current Production Reference (verified)

Job **58619479**, node `warpa003`, binary sha256
`b482c647...` — full: `b482b64703f0db7652b41310131815bef48f610d4b8ed208e7efef5d89b3ec96`
(confirmed via `docs/performance/p94_post_pressurevars_cuda_hotspot_profile_2026-09-28.md`).
60 calls, rc=0. All 4 ports CUDA (production default, nothing overridden).
State_To_FV: Fortran (default, no ENABLE set — P99/P100 memory-blocked).
TOTAL 3836.74 ms/call, FV_DYN_LOOP 1619.80 ms/call, REMAPPING 414.05
ms/call, DYN_CORE 845.85 ms/call, TRACER_2D 453.32 ms/call (Fortran).

## 4. Stage 3 — Direct Historical Comparison (context only, see §2's caveat)

No valid apples-to-apples historical baseline exists for TOTAL/
FV_DYN_LOOP/REMAPPING (§2). The only true "zero remap ports" C512
same-node measurement (Prompt 48) reported `DYN_CORE` only:

| Metric | Prompt 48 baseline | Prompt 94 current | Comparable? |
|---|---|---|---|
| `DYN_CORE` (fv_timing) | 6373.86 ms/call (CUDA, `warpa004`, pre-P49, zero remap ports) | 845.85 ms/call (`warpa003`, all 4 ports CUDA) | **NO** — different node, and `DYN_CORE` is unaffected by any of the 4 ports (sibling of `REMAPPING`); the ~7.5x apparent gap is not attributable to this task's ports at all (P37/P47/P48 already showed `DYN_CORE` alone swings >2x from node hardware variance). |
| TOTAL / FV_DYN_LOOP / REMAPPING | not measured at Prompt 48 | 3836.74 / 1619.80 / 414.05 ms/call | **N/A** — no Prompt 48 figures exist for these timers. |

**No historical TOTAL/FV_DYN_LOOP/REMAPPING speedup or improvement
percentage is reported from this table** — the required inputs do not
exist on the baseline side. This confirms Stage 4 (a fresh same-node A/B
reconstructing the baseline on the current codebase) is not merely
preferred but necessary.

## 5. Stage 4/5 — Same-Node A/B (the primary measurement)

Single Slurm allocation, job **58638414**, node **`warpa010`**, 1 A100,
6 MPI ranks (Intel MPI 2021.13), R8, C512, MPS ON,
`RUN_CUDA_FV3_FUSE_HALO_EDGES=YES`/`RUN_CUDA_FV3_FUSE_CORNER_FILL=YES`
(established best config), Graph OFF. Binary sha256
`2339366b10885c4532f08efa089d4e3b0a062403397f0c64c6a396b186e2064d`
(current authoritative build — includes Prompt 99's `State_To_FV` code,
which stays Fortran/disabled throughout since no `_ENABLE` var was set).

**Arm A** (reconstructed pre-port baseline, all 4 forced Fortran):
`RUN_CUDA_FV3_PTDELZ_FORCE_FORTRAN=1`,
`RUN_CUDA_FV3_REMAPUV_FORCE_FORTRAN=1`,
`RUN_CUDA_FV3_REMAPQ_FORCE_FORTRAN=1`,
`RUN_CUDA_FV3_PRESSUREVARS_FORCE_FORTRAN=1`.

**Arm B** (current production defaults): all four unset (default CUDA
per P69/P84A/P89/P93 closures).

Both arms: TRACER_2D Fortran (no `_ENABLE` set, P83 STOP), State_To_FV
Fortran (no `_ENABLE` set, P99/P100 memory-blocked),
REMAP_TOTALENERGYCON Fortran (no CUDA implementation exists, P95).

5 pairs, alternating order A1 B1 / B2 A2 / A3 B3 / B4 A4 / A5 B5 (matches
P69/P89/P93 methodology). All 10 arm-runs: rc=0, 30/30 `DYN Date`
completions, zero FATAL/OOM/Abort. Provenance verified every run:
`sha=2339366b...` matches `expected=2339366b...` in all 10
`P101A_PROVENANCE` lines. Selectors verified every run: Arm A prints
`PTDELZ_FF=1 REMAPUV_FF=1 REMAPQ_FF=1 PRESSUREVARS_FF=1`; Arm B prints
all four as `unset` — exactly as designed, no unexpected fallback.

### Per-pair timings (ms/call = accumulated seconds × 1000 / 30)

| Pair | Order | A TOTAL | B TOTAL | Improvement | A FV_DYN_LOOP | B FV_DYN_LOOP | Improvement | A REMAPPING | B REMAPPING | Improvement |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | A1 B1 | 5144.10 | 4428.54 | 13.910% | 2303.37 | 1641.24 | 28.746% | 1083.44 | 420.16 | 61.220% |
| 2 | B2 A2 | 5093.37 | 4407.49 | 13.466% | 2310.67 | 1621.39 | 29.830% | 1086.36 | 404.38 | 62.777% |
| 3 | A3 B3 | 5053.49 | 4412.01 | 12.694% | 2270.75 | 1627.57 | 28.325% | 1051.51 | 410.86 | 60.926% |
| 4 | B4 A4 | 5109.69 | 4412.56 | 13.643% | 2319.63 | 1627.15 | 29.853% | 1094.62 | 413.26 | 62.246% |
| 5 | A5 B5 | 5074.58 | 4434.80 | 12.607% | 2293.43 | 1650.59 | 28.030% | 1071.50 | 430.94 | 59.781% |

CUDA (Arm B) wins TOTAL, FV_DYN_LOOP, and REMAPPING in **5 of 5 pairs**,
regardless of alternating order (no order bias visible — A-first and
B-first pairs interleave with no systematic drift).

## 6. Stage 6 — Aggregate Statistics (the primary answer)

| Metric | Mean improvement | Median improvement | StdDev | Min | Max | Mean speedup (A/B) | Median speedup |
|---|---|---|---|---|---|---|---|
| **TOTAL** | **13.264%** | **13.466%** | 0.583 | 12.607% | 13.910% | **1.1530x** | 1.1556x |
| **FV_DYN_LOOP** | **28.957%** | **28.746%** | 0.847 | 28.030% | 29.853% | **1.4078x** | 1.4034x |
| **REMAPPING** | **61.390%** | **61.220%** | 1.171 | 59.781% | 62.777% | **2.5919x** | 2.5787x |

Tight spreads (CoV 1.3–4.4%) — no STOP-condition-level noise, no
node-order effect, no per-pair outlier.

**This is the primary answer**: on the current codebase, on one node, in
one Slurm allocation, the four retained CUDA remap ports acting together
produce a **13.3% mean TOTAL improvement (1.153x cumulative speedup)**,
a **29.0% FV_DYN_LOOP improvement (1.408x speedup)**, and a **61.4%
REMAPPING improvement (2.592x speedup)** relative to the same codebase
with all four forced back to Fortran.

## 7. Stage 7 — Individual Port Contributions

| Port | Local Fortran (ms/call) | Local CUDA (ms/call) | Direct speedup | Validated application effect (TOTAL, at time of closure) | Source |
|---|---|---|---|---|---|
| PT/DELZ | 542.28 (REMAP_GETT+REMAP_T+REMAP_NH combined) | 177.52 | **3.06x** | −7.26% (job 58530987, 3-pair) | P69 |
| REMAP_UV | ~169.3–169.5 | ~16.5–16.9 | **~10.23x** | +~1.0% (job 58533089/58538549) | P71/P84A |
| REMAP_Q | ~117–125 | ~36–38 | **3.19–3.31x** (mean 3.25x, 5-pair) | −1.51% to −1.72% (median), mean −2.65% incl. one node-variance outlier; 60-call confirm −1.41% | P89 |
| REMAP_PRESSUREVARS | 131.95–132.33 | 38–41.31 | **3.19–3.50x** (mean 3.50x, 5-pair) | +1.805% mean / +1.633% median (5-pair); +1.676% (60-call) | P93 |

**Do not sum these percentages.** Each was measured as an isolated
toggle against a *different, evolving* production baseline (PT/DELZ
against zero-ports Fortran; REMAP_UV against PT/DELZ-already-CUDA;
REMAP_Q against PT/DELZ+REMAP_UV-already-CUDA; REMAP_PRESSUREVARS
against PT/DELZ+REMAP_UV+REMAP_Q-already-CUDA) — the same ms/call of CPU
time cannot be "saved" by two different ports counted independently.
The only valid cumulative figure is Stage 6's direct four-at-once
same-node A/B (13.3% TOTAL, 29.0% FV_DYN_LOOP, 61.4% REMAPPING).

Sanity check: naively summing the four TOTAL percentages above
(7.26+1.0+1.51+1.805 ≈ 11.6%, or with REMAP_Q's mean 2.65%: ≈12.7%) lands
close to but not equal to Stage 6's measured 13.3% — the near-agreement
is coincidental (each historical run's "1 − remaining ports' fraction of
TOTAL" denominator differs), not validation that summing is a sound
method. Stage 6's direct measurement is authoritative; the individual
figures are diagnostic only.

## 8. Stage 8 — Current Bottleneck Shift

**Before the four ports** (reconstructed via Stage 4's Arm A on the
current codebase, and consistent with Prompt 69's pre-port profile):
`REMAPPING` was ~1077 ms/call (mean, Arm A), dominated by PT/DELZ's three
routines (REMAP_GETT/T/NH, then Fortran, ~542 ms/call combined) plus
REMAP_UV (~169 ms), REMAP_Q (~117-125 ms), and REMAP_PRESSUREVARS
(~132 ms) — all Fortran, all inside `FV_DYN_LOOP`, and collectively the
single largest addressable local-CPU cost in the dyn_core loop.

**After the four ports** (Prompt 94/96, current production, job
58619479): `REMAPPING` dropped to 414.05 ms/call (Stage 6 confirms
−61.4% relative to the reconstructed baseline). Inside the now-shrunk
`REMAPPING`, `REMAP_TOTALENERGYCON` (150.16 ms/call, Fortran, ~89%
`mpp_global_sum`-communication-bound per P95) is now the single largest
sub-timer — larger than any of the four already-CUDA routines. Outside
`REMAPPING` but still inside `FV_DYN_LOOP`: `DYN_CORE` (845.85 ms/call,
already GPU-ported, Gate C2, opaque) and `TRACER_2D` (453.32 ms/call,
Fortran, CUDA track CLOSED NO-GO per P83). Outside `FV_DYN_LOOP`
entirely, at the MAPL level (Prompt 96's finding): `DYN_EPILOGUE`
(688.67 ms/call, Fortran, MAPL-dominated, no MPI) is now a
previously-unexamined cost larger than the entire post-port `REMAPPING`
block.

**Summary**: the four ports did exactly what they were designed to do —
they took the dominant local-CPU remap bottleneck (~1077 ms/call, ~21%
of pre-port TOTAL) and cut it to 414 ms/call. The bottleneck has moved
to costs the current ports cannot reach: a communication-bound
reduction (`REMAP_TOTALENERGYCON`), a structurally-rejected tracer path
(`TRACER_2D`), an already-GPU-resident subsystem (`DYN_CORE`), and an
external MAPL-level cost (`DYN_EPILOGUE`) never profiled before Prompt
96.

## 9. Stage 9 — GPU Memory Cost

Current production configuration (all 4 ports CUDA, State_To_FV
Fortran/disabled — the actual shipped default) uses **≈40.26–40.33 GiB
of 40.34/40.96 GiB total, leaving only ≈69–73 MiB free** (established by
Prompt 99's OOM discovery and reconfirmed by Prompt 101's diagnostic job
58638073 — both fully reverted, not re-measured in this task since doing
so would require the same barrier-synchronized instrumentation Prompt
101 already flagged as a separate, not-yet-executed follow-up). This
task's own job (58638414) confirms the idle pre-run baseline (4 MiB
used / 40334 MiB free) is unchanged from prior runs, consistent with the
converged footprint being front-loaded at the first production dispatch
call rather than accumulating further during this task's 10 short runs.

**State_To_FV's CUDA implementation is explicitly excluded from this
figure** — it is not enabled by default (Fortran remains the production
choice, P99/P100, STILL MEMORY-BLOCKED) and was not enabled in any run
in this task.

**Net picture**: performance improved substantially (13.3% TOTAL, 2.6x
REMAPPING), but memory headroom is now exhausted at the same time — the
two facts are not in tension (the 4 ports added negligible dedicated
memory per Prompt 101's audit: REMAP_Q/PRESSUREVARS ≈82 MiB/rank
combined; PT/DELZ and REMAP_UV add zero), but production is now
memory-constrained for any *further* port (e.g. `State_To_FV`)
regardless of that next port's own performance merits.

## 10. Stage 10 — Summary Metrics

```
CURRENT CUMULATIVE TOTAL IMPROVEMENT:  13.264% (mean), 13.466% (median)
CURRENT TOTAL SPEEDUP:                 1.153x (mean), 1.156x (median)
FV_DYN_LOOP IMPROVEMENT:                28.957% (mean), 28.746% (median)
REMAPPING IMPROVEMENT:                  61.390% (mean), 61.220% (median)
FASTEST LOCAL PORT:                     REMAP_UV, ~10.23x direct speedup
CURRENT GPU MEMORY:                     ~40.3 GiB used / 40.96 GiB total (~70 MiB free)
PRODUCTION CUDA DEFAULTS:               PT/DELZ, REMAP_UV, REMAP_Q, REMAP_PRESSUREVARS
```

### 10a. Addendum — reused as Prompt 102's performance baseline

Prompt 102 (`docs/performance/p102_multigpu_scaling_architecture_2026-09-29.md`,
a multi-GPU/device-memory architecture study, no numerical/default changes)
reused this file's Stage 10 figures verbatim as its input "PERFORMANCE
BASELINE" and additionally measured, single-node, C512 memory at 1- and
2-ranks-per-GPU density (6746 MiB, 13450 MiB) to resolve Prompt 101's open
per-rank-vs-fixed-per-GPU memory-scaling question (per-rank dominates,
HIGH confidence). No figure in this document was changed or superseded by
that study — it remains the current cumulative-speedup reference.

## 11. Compliance

- No numerical/scientific code modified.
- No production defaults changed (all 4 ports remain CUDA-default after
  this task; the sbatch script only sets environment variables for the
  duration of each arm's run, then unsets them).
- State_To_FV CUDA: not enabled in any run.
- TRACER_2D CUDA: not enabled in any run.
- MPI layout: unchanged (6 ranks/1 A100, Intel MPI 2021.13, same as
  every prior C512 production job).
- GPU count: unchanged (1 A100).
- R4: not touched.
- Nsight Compute: not run.
- No memory-cleanup or multi-GPU work performed.
- No commit/push.
- New artifacts: this doc; `experiments/p101a_cumulative_ab5.sbatch` and
  `experiments/p101a_pair{1..5}_{A,B}/` under the `GEOSfvdycore_fortran_perf`
  checkout (job scripts/logs, not source — not committed, different repo
  from this one).

**CUDA MODERNIZATION PERFORMANCE CONSOLIDATED — CURRENT PRODUCTION SPEEDUP: 1.153x**

## Addendum (2026-09-29, Prompt 103) — Orthogonal 2-GPU Result

Prompt 103 measured a second, independent speedup axis at the same
production-default configuration: 1-GPU/6-ranks vs. 2-GPU/3-ranks-per-GPU
(job 58642794): **TOTAL 1.120x/10.70%**, bitwise-exact correctness. This is
**not directly comparable/multiplicative** with this doc's 1.153x (that
figure is CUDA-vs-Fortran at fixed 1-GPU; this new figure is 1-GPU-vs-2-GPU
at fixed CUDA-defaults) — as **DERIVED CONTEXT ONLY**, a naive compositional
estimate is ≈1.153×1.120 ≈ 1.292x vs. an all-Fortran/1-GPU baseline, not
independently measured. See
`p103_c512_2gpu_validation_performance_2026-09-29.md`.
