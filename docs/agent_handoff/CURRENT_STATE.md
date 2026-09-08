# Current Project State

_Last updated 2026-09-08 for **Prompt 21 (longer/multi-k_split R4
validation)**: §9/§10 and this lead paragraph. **Prompt 21 is COMPLETE:
PASS.** Extends Prompt 20's short-run R4 PASS to the same longer/multi-
`k_split` depth already established for R8 in Prompt 19, using the identical
two-stage ladder, same c8/npz=10/n_split=3/R8→R4/6-MPI-rank configuration,
same `dce`/`dcx` savepoints, same `compare_dcx_backends.py` comparator and
Prompt 20's R4-specific thresholds (BITWISE=0 ULP, ROUND-OFF (R4) ≤1e-6,
SMALL (R4) ≤1e-2, SIGNIFICANT (R4) >1e-2) — no new validation framework, no
new savepoints. **Stage A** (job 58282446): R4, k_split=1, extended to **72
dyn_core calls** (matching Prompt 19 Stage A exactly). Both backends rc=0,
it=72 both. Entry (`dce`) call 1 BITWISE; exit (`dcx`) sampled at calls
1/6/18/36/54/72 all ROUND-OFF/SMALL for `delp`/`pt`/`delz`/`pe` (peak max_rel
6.50e-5/4.72e-5/7.13e-6/9.20e-6 — comfortably bounded over the full 72
calls). Three fields show threshold-crossing signatures, all characterized
and explained, none a defect: **`w`** is bounded/oscillatory (max_abs
3.4e-4→1.11e-3, bulk ratio 5–15%, no trend — a cancellation-floor artifact,
not growth); **`u`/`v`** show small episodic single-cell crossings peaking
at call 48 (~1.3–1.5e-2 raw max_rel, bulk ratio flat 0.08–0.18%); **`omga`**
is the one genuine finding to carry forward — its *bulk* (RMS-ratio) metric
shows real SUPERLINEAR growth over the 72 calls (0.06% at call 1 → crosses
the R4-SIGNIFICANT max_rel line around call 19 → 4.06% bulk by call 72),
while its absolute error plateaus after call ≈43 — i.e. the growth is driven
by `omga`'s own field magnitude shrinking (a decaying diagnostic), not by
the CUDA/Fortran discrepancy itself growing without bound. Zero NaN/Inf
(benign pre-first-write `mfxL`/`cxL`/`cyL` pattern only); `delp` positive
both backends at every sampled checkpoint. **Stage B** (job 58282890): R4,
k_split=2 (the smallest genuine multi-`k_split` case, matching Prompt 19
Stage B), **12 dyn_core calls**, both backends rc=0/it=12. Compared at calls
1/2/3/6/12: `u`/`v`/`delp`/`pt`/`delz`/`pe` all comfortably ROUND-OFF/SMALL
throughout (window too short to reach Stage A's later-call `u`/`v`/`omga`
signatures). `omga` is noisy/oscillatory within this short window (bulk
ratio 6.23%/0.089%/1.19%/0.088%/0.172% at the 5 checkpoints — not a sustained
climb, since 12 calls is short relative to where Stage A's own trend first
separated from noise, around call ~19–30). **The decisive remap-boundary
check (§7/§8 of the task) is a direct comparison of the `dcx` exit at call 2
against the `dce` entry at call 3, i.e. straddling the first k_split=2 remap
step**: for both `w` and `omga`, the post-remap entry metrics are the same
order of magnitude as (and for `omga`, smaller than) the pre-remap exit
metrics (`w`: max_rel 1.5494e-02 → 1.5501e-02; `omga`: max_rel 1.4362e-03 →
9.2027e-04) — **no discontinuous jump at the remap boundary for either
field.** `omga`'s higher within-call-3 exit value (1.78e-2) therefore
develops during call 3's own dyn_core execution, not at the boundary — the
same general intra-call growth pattern already present at call 1 (which has
*no* preceding remap at all: bitwise entry → 8.95e-2 exit), proving the
remap step is not the cause. delp positive both backends at all 5
checkpoints. **Stage B: PASS.** **R4 is now validated to the same
integration depth R8 reached at Prompt 19** for this configuration. **Zero
of the 3 permitted integration fixes were needed — no source code was
modified in this task.** No regression rerun performed (no source changed).
Full detail: `docs/agent_handoff/prompt_21_longer_r4_validation.md`. Next:
extend R4 to a wider grid (the same step already queued for R8), and if R4
is ever extended past 72 calls, re-check whether `omga`'s bulk-ratio growth
continues, accelerates, or plateaus.

Before that, last updated 2026-09-08 for **Prompt 20 (R4 numerical validation)**: §7/§9/§10
and this lead paragraph. **Prompt 20 is COMPLETE: PASS.** Same standalone
short-run recipe as Prompt 17 (`p17_both.sbatch` pattern, here
`p20_r4_both.sbatch`), same c8/npz=10/n_split=3/k_split=1/6-MPI-rank
configuration, same `dce`/`dcx` backend-agnostic savepoints, same
`compare_dcx_backends.py` comparator (unchanged) — but the **CUDA and Fortran
executables were both built R4 (single precision)** instead of R8, closing
the "R4 has never been savepoint-compared" gap recorded since Gate C2.
**R4 type mapping: PASS** (Fortran precision macro selects `real4`;
`fv3_savepoint_mod.F90` now records a `precision=` token in its `.meta` files,
read by a new `savepoint_reader.py:dtype_for_precision()` — old R8-only trees
without the token still default to R8, no back-compat break; the C
ABI/CUDA-runtime side reuses the pre-existing `fv_real` typedef already
exercised by the R4 CTest suite, no change needed there). Because
`compare_dcx_backends.py`'s built-in verdict thresholds are R8-tuned
(ROUND-OFF ≤1e-13, SMALL ≤1e-8) and every R4 field trips its SIGNIFICANT
bucket by design, **R4-specific thresholds were derived** from the measured
ratio of R4 max_rel to the already-documented R8 max_rel for the same field —
that ratio clusters at 4.0e8-5.9e8 across nearly every dyn_core-exit field,
closely matching the theoretical float32/float64 epsilon ratio
`eps32/eps64 ≈ 5.369e8`, i.e. quantitative evidence the R4 differences are
proportionally-scaled single-precision round-off, not a new defect. R4
thresholds used: BITWISE = 0 ULP, ROUND-OFF SCALE (R4) ≤1e-6, SMALL NUMERICAL
DIFFERENCE (R4) ≤1e-2, SIGNIFICANT (R4) >1e-2. **First call: entry (`dce`)
BITWISE IDENTICAL; exit (`dcx`) all 17 fields ROUND-OFF SCALE or SMALL, zero
SIGNIFICANT**, worst field `w` at max_rel=7.421176e-03 (1.4 orders below the
R4-SIGNIFICANT cutoff). **6-call repeated-call run: PASS**, worst field `w`
trend 7.421176e-03 (call 1) → 9.299323e-03 (call 3) → 1.020307e-02 (call 6) —
mild bounded growth (~37% over 6 calls), no acceleration, still under the R4
1e-2 SIGNIFICANT bound at call 6 (flagged as the figure to watch first if this
envelope is extended). Zero NaN/Inf beyond the already-documented R8-precedented
benign pre-first-write `mfxL`/`cxL`/`cyL` entries at `dce`; `delp` positive on
both backends at calls 1 and 6, all 6 ranks. **R4 MPI transport: PASS**, 23/23
R4 CTest including `fv3_halo_mpi_transport`, plus 6/6 repeats under
`srun --ntasks=6 --cpus-per-task=2` (the first `--ntasks=1` attempt hit an
OpenMPI "not enough slots" `srun`-resourcing artifact, same class as the
Prompt 16 record, not a code defect — fixed by requesting 6 tasks). **The
shared runtime source (`dyn_core.F90`/`fv_dynamics.F90` in the fixture's nested
`@fvdycore`) shows as modified but was diffed directly and confirmed to be the
pre-existing Prompt 17/18 backend-integration and savepoint instrumentation,
unchanged since Prompt 19's R8 PASS** — `dyn_core.F90`'s diff is
savepoint-dump calls plus one argument-only `real()` cast on
`mfx`/`mfy`/`cx`/`cy` (does not touch the model's own `real(kind=8)`
declarations or feed back into model state); `fv_dynamics.F90`'s diff is
exactly Design A, already R8-validated through Prompt 19. No source in either
repository was modified during this task, so a fresh R8 regression job was
judged **NOT REQUIRED** (the shared source is unchanged since the last R8
PASS). **2 of 3 permitted substantial R4 fixes consumed** (the savepoint
precision-metadata token; the `mfx`/`mfy`/`cx`/`cy` `real()` cast), 1
remaining. **Numerical algorithm changed: NO.** Scope: this validates R4 only
at the same short-run envelope as Prompt 17 (c8/npz=10/n_split=3/k_split=1/6
calls) — it does **not** extend to Prompt 19's longer/multi-`k_split` R8
ladder, nor to any wider grid. Full detail:
`docs/agent_handoff/prompt_20_r4_numerical_validation.md`. Next: extend R4 to
a longer multi-call/multi-`k_split` run matching Prompt 19's R8 ladder
(watching the `w` field's growth trend), then R4 becomes eligible for the same
wider-grid step already queued for R8.

Before that, last updated 2026-09-04 for **Prompt 19 (longer standalone validation)**:
§7/§9/§10 and this lead paragraph. **Prompt 19 is COMPLETE: PASS.** Two-stage
validation ladder, same c8/npz=10/n_split=3/R8/6-rank configuration
throughout, same executable/allocation per backend, no new validation
framework (reused `p17_both.sbatch`'s mechanics and
`tools/savepoints/compare_dcx_backends.py` unchanged). **Stage A** extended
`k_split=1` from 6 to **72 dyn_core calls** (`JOB_SGMT` 3 days): entry call 1
BITWISE identical on all 13 primary state fields (6/6 ranks); every sampled
exit checkpoint (calls 1/6/18/36/54/72) ROUND-OFF SCALE or SMALL NUMERICAL
DIFFERENCE with **zero SIGNIFICANT differences**; worst discrepancy `w`
max_rel=5.116168e-11 at call 72; all fields FLAT or SUBLINEAR growth, no
acceleration. **Stage B** used **k_split=2** — the smallest value that
exercises `fv_dynamics.F90`'s `do n_map=1,k_split` loop more than once per
heartbeat, confirmed unambiguous directly from `fv3.j:373-374` (hard-stop
condition 2 did not trigger) — over 6 heartbeats = **12 dyn_core calls**
(job 58246786, both backends rc=0, savepoints reach `it12` on both sides).
Comparison at entry call 1 (BITWISE, 6/6 ranks), entry call 3 (the first
dyn_core entry immediately after the first heartbeat's remap step — **no
anomalous jump**, all fields ROUND-OFF/SMALL, worst `w` max_rel=2.69e-11),
and exit calls 1/2/6/12 (all ROUND-OFF/SMALL, **zero SIGNIFICANT**, worst `w`
max_rel=3.527908e-11 at call 6 — smaller than Stage A's 72-call worst,
consistent with call-count-driven accumulation rather than a k_split-specific
effect). `delp` positivity explicitly re-checked at call 12: min
9957.5868/9962.5076 Pa, identical on both backends, 6/6 ranks. The recurring
`mfxL`/`mfyL`/`cxL`/`cyL` pre-first-write flux-capacitor anomaly (benign,
established in Prompt 17/18) reproduces identically in both stages and never
recurs at any exit checkpoint. **Zero of the 3 permitted integration fixes
were needed — no source code was modified.** Hard-stop condition (5), "both
Stage A and Stage B pass," is the terminal state reached; no escalation to
Opus. No CTest rerun performed (no source changed); the Prompt 18 record
(24/24 MPI transport, Intel MPI 5/5, R8 CTest 23/23) stands unchanged by
inference. Full detail:
`docs/agent_handoff/prompt_19_longer_standalone_validation.md`. Next
validation envelope: R4 savepoint comparison (still only compiled/ctest-clean,
never savepoint-compared), then wider grids, then k_split values matching the
real production heartbeat/physics cadence beyond this minimal k_split=2
probe.

Before that, last updated 2026-09-04 for **Prompt 18 (six-rank MPI halo transport
determinism)**: §1/§7/§9 item 16b/§10 and the new Prompt 18 lead paragraph
below, plus three inline premise corrections inside the Prompt 17 "NEW OPEN
DEFECT" block. **Prompt 18 is COMPLETE and it CLOSES the last blocker on longer
standalone validation**: the intermittent `fv3_halo_mpi_transport` failure was a
CUDA stream-ordering defect in the TEST HARNESS (not the transport, not an MPI
contract violation, not an OpenMPI bug), is fixed, and the transport now passes
**24/24 consecutive OpenMPI runs** at 16692 checks / 0 failures, with Intel MPI
5/5 PASS (was 5/5 FAIL) and R8 CTest 23/23. Read that paragraph first if you are
resuming; §9 item 17 is the new RESUME HERE. Before that, last updated
2026-09-04 for **Prompt 16 (standalone CUDA integration,
resumed)**: §1/§9/§10 and the lead paragraph immediately below. **Prompt 16 is
PARTIAL: the public C ABI is now complete and verified, but a second,
distinct halo-ownership ambiguity was found at the real `fv_dynamics.F90`
call site and the pass ESCALATES on it** — see the new lead paragraph just
below and §9 item 16. This does not reopen Prompt 15 (still CLOSED) or Gate
C2 (still CLOSED); it is a new, narrower blocker one layer up the call stack,
outside anything any prior prompt had read. Before that, last updated
2026-09-04 for **Prompt 15 (cross-process MPI halo transport)**:
§1/§3/§4/§5/§7/§8/§9/§10 and the lead paragraph immediately below. **Prompt 15
is COMPLETE and it CLOSES Prompt 14's blocker.** Before that, last updated
2026-09-04 for **Prompt 14 (standalone CUDA production
integration attempt)**: §1/§9/§10 and this summary, plus this new lead
paragraph. **Prompt 14 is BLOCKED/ESCALATED, not completed** — see the next
paragraph. Before that, last updated 2026-09-04 for **Prompt 13
(architecture-level optimization — CUDA Graph replay repair)**:
§1/§6/§8/§9/§10 and this summary. Previously
regenerated from Git + handoffs on 2026-08-31 (§1/§4/§5/§6/§7/§8/§9/§10
updated for the Prompt 6B **`omga` exit finalization** pass). Result: **the
dyn_core exit `omga` now VALIDATES at SMALL NUMERICAL DIFFERENCE on all six
tiles** (worst max_abs 4.875354e-14, guarded max_rel 1.505773e-12, rms
≤1.073e-14), down from max_abs ≈4.7e-02 / max_rel ≈1.0 — a ~1e12× residual
reduction. **All 13 dyn_core-exit fields now validate, all 15 in-loop boundaries
still validate, and GATE C2 IS CLOSED: the CUDA dyn_core is NUMERICALLY
VALIDATED against authoritative Fortran** for the c8 / npz=10 / n_split=3 /
k_split=1 / R8 / six-tile production first-run configuration. One fix applied:
the previously **entirely absent** omega finalization path
(`dyn_core.F90:503-517` `pem` snapshot, `:1211` `(pe−pem)*rdt`, `:1218`
`adv_pe` → `:1535-1636`) is now implemented on both the host and device sides,
with `en1/en2/ec1/ec2` **derived** from the already-authoritative B-grid corner
geometry (proven, not invented — see §4) and `a2b_ord2` **reused** rather than
rewritten. **Numerical algorithm changed: NO.** The §6 performance baseline has since been
**RE-ESTABLISHED post-Gate-C2 (Prompt 7, 2026-09-03)**: CUDA resident/transfer
medians retaken on all three grids, Fortran comparison REUSED (unchanged
fixture/config), and one bounded nsys profiling pass confirms the
launch/orchestration-bound classification still holds (17,730 kernel
launches/dyn_core call; no single kernel >6.3% of kernel time). No
optimization was performed in that pass. A **CUDA Graphs prototype (Prompt 8,
2026-09-03) was then attempted and is BLOCKED**: capturing the
`dgrid_dynamics` stage succeeds mechanically, but a single graph-mode
dyn_core call diverges numerically from the non-graph call on all 13 exit
fields, all 6 tiles (wind fields `u`/`v`/`ua`/`va`/`w`/`omga` at max_rel
≈1.0–2.0, i.e. uncorrelated — far beyond Gate C2 tolerance), and sustained
replay across benchmark repeats crashes with an illegal memory access in the
halo-transfer path that corrupts one tile's GPU context for the rest of the
job. Per the task's own instructions this triggered a STOP (tolerance was not
loosened) and both permitted autonomous fix cycles were already used on
unrelated job-script bugs, so no further fix was attempted. Full detail,
including a suspected (unconfirmed) ~8× timing confound from the job's MPI
resource configuration: `docs/agent_handoff/prompt_8_cuda_graphs_prototype.md`.
**Numerical algorithm changed: NO.** Gate C2 itself is unaffected (default,
non-graph path unchanged; R8 ctest and default smoke both PASS). A **narrow
kernel-fusion prototype (Prompt 9, 2026-09-03, job 58230026) was then
attempted and is COMPLETE, unlike Prompt 8**: fusing `c_sw`'s per-level
`fill2_4corners`+`fill_4corners` corner-fill pair (both directions, opt-in via
`--fuse-corner-fill`, default off) into one `launch_fill_corner_triplet` call
produces a `dyn_core_exit` dump **bitwise identical** to the default path on
all 13 fields × 6 tiles (78/78 IDENTICAL, 0 DIFFERS), with R8 ctest 22/22 PASS
and default smoke PASS. Performance effect is small and only partly resolved:
nsys shows kernel launches/call falling 17,730→17,370 (−2.0%) and kernel busy
time 74.0→73.0 ms/call (−1.4%), both trustworthy but modest since only 2 of
17,730 launch sites were touched; a raw wall-clock read (~10% resident-time
drop) is **not** treated as validated — repeat-to-repeat noise in that job was
comparable to the claimed delta. **Numerical algorithm changed: NO.** Full
detail: `docs/agent_handoff/prompt_9_kernel_fusion_validation.md`. Prompt 9's
own handoff was then committed (with the Prompt 6B `omga` work and Prompts 7/8)
in `b456897` ("Close Gate C2 and add initial CUDA launch optimizations") —
this was **not** authored by an agent pass; it landed on the tree between
sessions and was found and reconciled, not created, by Prompt 10 below. A
**`copy_corners_kernel` batching characterization (Prompt 10, 2026-09-04) was
then attempted and concluded NO IMPLEMENTATION IS WARRANTED**: all four
batching dimensions the task specified (tile-batching, same-field/direction
merge, cross-field merge, adjacent-call fusion) were ruled out by direct
source reading — either they require a runtime/topology redesign (six tiles
run as independent concurrent host threads with no shared launch surface) or
they cross a real, source-proven producer/consumer dependency (every
`dir=1`/`dir=2` corner-rotation pair at every loop site is separated by a
consumer kernel that reads the `dir=1` rotation before `dir=2` overwrites the
same cells). One narrower, safe-in-principle lever was found (merging the
outer per-`nord` corner calls in `del6_vt_flux`/`deln_flux_pass`, since
different `nord` passes touch disjoint vertical levels) but its ceiling is
only ≈180 of 17,730 total launches (≈1.0%) — below Prompt 9's own measured
noise floor — so it was not implemented. **No code was changed; no
Slurm job was run.** Full detail:
`docs/agent_handoff/prompt_10_copy_corners_batching.md`. A **pack/unpack
halo-edge fusion (Prompt 11, 2026-09-04, jobs 58238397/58238413) was then
attempted and is COMPLETE, with a real measured improvement, unlike Prompt
9's small effect**: `pack_edge_kernel`/`unpack_edge_kernel` already batch
level/along-position/ring-depth in their grid; the one remaining unbatched
axis was the 4-edge loop, and (unlike Prompt 10's `copy_corners` dir=1/dir=2
case) each edge's send/recv slot is disjoint with no producer/consumer
dependency between edges, so fusing all 4 edges into one launch/`(group,
field)` — opt-in via `--fuse-halo-edges`, new `pack_edge4_kernel`/
`unpack_edge4_kernel` in `fv3_gpu_halo_exchange.cu`, default off — is provably
safe reordering. Produces a `dyn_core_exit` dump **bitwise identical** to the
default path on all 13 fields × 6 tiles (78/78 IDENTICAL, 0 DIFFERS), same for
the combination with Prompt 9's `--fuse-corner-fill`, with R8 ctest PASS and
both default/optimized smoke PASS. Measured whole-dyn_core-call kernel
launches 17,730→16,074 (**−9.34%**), kernel busy time 74.0→67.44 ms/call
(**−8.86%**), and a reproducible 32×32/R8 resident wall-clock improvement of
**−20.28%** (29.20→23.28 ms/call median of 5 repeats, spread ≤0.33 ms —
comfortably above the noise floor that made Prompt 9's wall-clock claim
unusable). Combining both fusions measured very slightly slower than
`--fuse-halo-edges` alone (23.86 vs 23.28 ms/call) — recorded, not chased
further; still bitwise identical. **Numerical algorithm changed: NO.** Full
detail: `docs/agent_handoff/prompt_11_next_launch_target.md`. An
**`a2b4_global_corner_kernel` compute-efficiency tuning pass (Prompt 12,
2026-09-04, job 58238626) was then attempted and is COMPLETE, at MODERATE
value**: the pre-optimization kernel launched exactly 1 CUDA block of 128
threads (only 10 active, one A100 SM of 108 used) and serialized up to 4
physical-corner branches per active thread, each issuing 3
`a2b_extrapolate` calls (6 `a2b_great_circle` transcendental evaluations) —
up to 24 serialized evaluations per thread when all 4 corners are active.
Restructured the launch to `dim3(4, levels, 1)` with 1 thread/block
(`blockIdx.x` selects corner via `switch`, `blockIdx.y` selects level,
mirroring the codebase's own `grid1(n,levels)` idiom), spreading the 4
corners across independently-schedulable blocks with **unchanged per-corner
formula and summation order** — bitwise-identical by construction. Section 2
ncu profiling (`--kernel-name a2b4_global_corner_kernel`, both before/after)
returned `ERR_NVGPUCTRPERM` (GPU performance-counter access denied to this
account on this Discover allocation) — a **confirmed platform permission
constraint**, not a flag error; the Section 3 bottleneck classification
(SERIAL/LOW-PARALLELISM primary, OCCUPANCY-BOUND secondary) is therefore
source-reading inference, not ncu-measured. Validated `dyn_core_exit` dump
**156/156 IDENTICAL** (13 fields × 6 tiles × {default, `--fuse-halo-edges`}),
R8 ctest 22/22 PASS (both before/after builds), both smokes PASS. Measured
32×32/R8/resident whole-dyn_core-call improvement: 22.743409→22.255549
ms/call (**−2.15%**, MODERATE per the task's own ≥3%/1–3%/<1% buckets).
**Numerical algorithm changed: NO.** Terminal recommendation: further local
compute-tuning on this one kernel is not justified without ncu access; next
optimization step should be either requesting GPU performance-counter
permission from NCCS, or shifting to kernel-time/occupancy tuning of the
individually more expensive `riem_solver3_kernel`/`riem_solver_c_kernel`
(explicitly deferred architectural work). CUDA Graphs remained explicitly
out of scope for this pass. Full detail:
`docs/agent_handoff/prompt_12_a2b4_kernel_tuning.md`. An
**architecture-level optimization pass (Prompt 13, 2026-09-04, job 58238910)
was then run and is COMPLETE — it SUPERSEDES Prompt 8's BLOCKED CUDA Graphs
status**: the two candidate directions (A, repair CUDA Graph replay safety for
a bounded capture region; B, reduce host launch/orchestration overhead by
explicit stage/tile batching without graphs) were both studied from source,
and **A was found to be a LOCAL defect, not a graph-semantics problem**. Root
cause, verified directly in `fv3_gpu_runtime.cpp`: **the Prompt 8 capture arm
never launched what it captured.** Stream-captured work is *recorded, not
executed*; the arm did BeginCapture → run → EndCapture → Instantiate with no
`cudaGraphLaunch`, and the `} else` bound the fallback non-captured
`run(dispatch.dgrid_dynamics)` to non-graph mode only — so one entire `d_sw`
substep (transport, divergence/vorticity damping, KE/`heat_source`
generation) was silently dropped on the first substep of the first
`fv3_gpu_dyn_core_step` per context per repeat. That exactly explains Prompt
8's O(1) *uncorrelated* wind divergence (max_rel ≈1.0–2.0) without any
device-pointer/parameter-binding hypothesis; graph-node parameter updates
required were affirmatively measured as **ZERO**. Fixed with ~15 lines across
2 files (capture arm now launches the freshly instantiated exec, plus a
capture-time `bdt`/`n_split` invariance guard for the one by-value scalar,
`DSwOptions::dt`); capture region unchanged (`dgrid_dynamics` only, halo
exchange **not** captured). Validated: `dyn_core_exit` dump **78/78 fields ×
tiles BITWISE IDENTICAL** to the non-graph path at 1 call (the capture
occurrence — precisely what Prompt 8 got wrong) and again at 5 calls (1
capture + 14 replays/tile); a 600-call / 1,800-substep / 1,770-replay stress
ran clean with **no illegal memory access** (Prompt 8's replay crash did not
reproduce); R8 ctest 22/22 PASS; default, graph, and graph+`--fuse-halo-edges`
smokes PASS. Measured 32×32/R8/resident (same job, same node, same binary
±`--cuda-graph`): 21.660791 → **20.546787 ms/call (−5.14%)**, with whole-call
`cudaLaunchKernel` count 16,182 → **11,067/call (−31.6%)**. **Numerical
algorithm changed: NO**; graph mode stays opt-in (`--cuda-graph`, default
off). The pass's most consequential *finding* is negative and forward-looking:
**−31.6% launches bought only −5.14% wall clock**, breaking the
launch-latency model that fit Prompts 7 and 11 to 87–92%, and
`cudaStreamSynchronize` total time rose 0.44 → 135.5 ms — the workload has
crossed from **host-launch-bound to GPU-execution-bound**, which retires
Direction B's expected value (its −35.3% launch reduction would now be worth
~5%, not ~30%). Full detail:
`docs/agent_handoff/prompt_13_architectural_optimization.md`. Read
`CURRENT_STATE.md` first; avoid re-reading older milestone handoffs unless a
specific open item below requires it.

**Prompt 18 (2026-09-04) — six-rank MPI halo transport determinism: COMPLETE.
The intermittent `fv3_halo_mpi_transport` failure is ROOT-CAUSED, FIXED, and the
transport is now DETERMINISTIC at 24/24 consecutive OpenMPI runs. This CLOSES
§9 item 16b and LIFTS the gate on longer standalone validation.** Full detail:
`docs/agent_handoff/prompt_18_mpi_transport_determinism.md`. The essentials:

- **The defect was in the TEST HARNESS, not the transport, and it is NOT an MPI
  contract violation and NOT an OpenMPI bug.** It is a CUDA stream-ordering
  defect: `test_halo_mpi_transport.cu` creates its exchange stream with
  `cudaStreamCreateWithFlags(..., cudaStreamNonBlocking)`, which by definition
  does **not** implicitly synchronize with the legacy default stream — and the
  test issued two of its own device-buffer setup operations on the legacy
  stream. (1) `cudaMemset(send/recv, 0xFF, ...)`: a plain `cudaMemset` on
  *device* memory is asynchronous w.r.t. the host and lands in the legacy
  stream, so the poison could land *after* the transport's staging H2D — re-
  poisoning a whole recv slot after correct data had already arrived. (2)
  `TileArray::create()`'s synchronous `cudaMemcpy` H2D from **pageable** memory
  returns once the source is staged for DMA, not once the device write has
  landed, and also lands in the legacy stream — so the pack could read the
  field array before it was populated. Both fixed by enqueueing on the
  exchange stream (`cudaMemsetAsync`/`cudaMemcpyAsync`).
- **Proven by intervention, not just by observation.** The racy form is
  retained behind `FV3_HALO_TEST_LEGACY_SETUP=1`, so the defect is
  reproducible on demand: on the **post-fix** binary, 24/24 default repeats
  PASS while 3 of 6 legacy-control repeats FAIL (job 58246343). Same binary,
  same node, same MPI — only the stream ordering of the test's own setup
  differs.
- **Three premises in the task prompt and in §10 below were measured WRONG**,
  and all three corrections are recorded inline in the Prompt 17 "NEW OPEN
  DEFECT" block: (a) "Intel MPI 5/5 PASS / OpenMPI intermittent" was
  **inverted** — pre-fix Intel MPI 2021.13 was **5/5 FAIL** and OpenMPI 4.1.6
  was 3/5 PASS; (b) the failing value is **not** 0.0 — it is the test's own
  `0xFF` poison, which is NaN, and `std::llround(NaN)` → `LLONG_MIN` decodes
  through the provenance oracle to the *identical* text `tile 0 (-8,-8,0)`, so
  the old output could not tell poison from a genuine zero (`classify()` now
  reports `POISON-REMAINS` / `ZERO` / `WRONG-VALUE` distinctly); (c) the blast
  radius is **not one cell** — with the first-failure suppression removed, a
  failing occurrence loses a **whole field/edge message**: all 72 written cells
  (3 rings × 6 along-positions × 4 levels) of a 105-element recv slot, census
  `poison 105, zero 0, other 0`, at a correctly-computed slot offset. The
  "always `h=1 a=1 k=1`" signature was an artifact of that suppression.
- **The transport itself was audited and cleared** (§3/§4/§5/§7 of the task),
  so the next pass need not redo any of it: pack→D2H is genuinely
  stream-ordered (`fv3_gpu_runtime.cpp:912-948`, all on `d.transfer_stream`);
  request storage is a fixed `std::array<MPI_Request, 2*4*max_halo_group_fields>`
  value-initialized `{}`, **not** a `std::vector`, so the reallocation
  hypothesis the task flagged is **retired**; the tag space is symmetric
  between sender and receiver and collision-free within and across exchanges
  (max tag 207, inside the guaranteed `MPI_TAG_UB ≥ 32767`); and every recv
  slot provably lies inside the byte range the bulk H2D overwrites (max
  `slot_base` + one slot == `group_buffer_elements()`), which the measured
  offsets then confirmed empirically (315 = 3×105, 105 = 1×105).
- **One substantial production fix, in the same defect class** (1 of 3
  permitted; test instrumentation is §9-exempt):
  `fv3_gpu_runtime.cpp:1290-1291`, the attach-time halo-buffer clears in
  `attach_halo_transport`, were plain legacy-stream `cudaMemset` and are now
  `cudaMemsetAsync` on `d.transfer_stream` — the same stream every
  pack/transfer/unpack uses. Previously only *probably* ordered before the
  first pack (much host work and two MPI collectives intervene), never
  *provably* so; now a stream guarantee at zero cost.
- **Acceptance, all measured.** OpenMPI 4.1.6, exact Prompt-15 layout (1 node /
  6 ranks / 1 A100 / rome / R8): **24/24 consecutive PASS**, `checks: 16692
  failures: 0` in every run, no poison remains, no MPI errors, no CUDA errors
  (job 58246343 A). **R8 CTest 23/23** (same job, phase C) — up from Prompt
  17's 22/23. **Intel MPI 2021.13: 5/5 PASS** (job 58246352), where this exact
  configuration was 5/5 FAIL pre-fix — the strongest cross-stack evidence, and
  only consistent with an MPI-independent mechanism.
- **Standalone regression re-run and unchanged** (job 58246353, both backends
  from the same executable in one allocation, c8/npz=10/n_split=3/k_split=1/R8/
  6 ranks): both arms rc=0, **6 dyn_core calls** each; dyn_core **ENTRY call 1
  BITWISE IDENTICAL on all 13 state fields, 6/6 ranks**; dyn_core EXIT **102
  verdicts (17 fields × 6 calls) = 78 ROUND-OFF SCALE + 24 SMALL NUMERICAL
  DIFFERENCE, zero SIGNIFICANT, zero NONFINITE**, worst `max_rel`
  **2.034576e-11** (`w`, call 5) with no divergence trend across calls
  (1.716/1.463/1.889/1.956/2.035/2.008 e-11); owned-box min `delp` positive and
  **identical to all printed digits on both arms**, 9967.182763 → 9957.518684
  Pa, nonfinite=0; zero halo-transport diagnostics. Every figure **reproduces
  Prompt 17's record to the digit**, so neither the test fix nor the production
  hardening perturbed the numerics.
- **Numerical algorithm changed: NO.** Nothing was committed or pushed.
- Lesson worth carrying forward: **a plain `cudaMemset` or a synchronous
  `cudaMemcpy` is not ordered against a `cudaStreamNonBlocking` stream.** Both
  are legacy-stream operations, and `cudaStreamNonBlocking` exists precisely to
  opt out of the implicit legacy synchronization. Any setup, clear, or seed of
  a buffer that a non-blocking stream will later read must be enqueued *on that
  stream*. The intermittency, the stack-dependence, and the "unwritten" reading
  all followed from this one hazard — and the "OpenMPI-specific" framing it
  produced sent two prior passes looking in the transport.

**Prompt 17 (2026-09-04) — FMS halo ownership resolved; STANDALONE CUDA
BACKEND NOW RUNS IN THE REAL `StandAlone_FV3_Dycore.x`.** This closes the
Prompt 16 blocker (§9 item 16) and carries Prompt 14 through to a working
dual-backend standalone executable. Read
`prompt_14_standalone_cuda_integration.md` §8 for the full record; the
essentials:

- **FMS ownership contract, established from source (not assumed).**
  `mpp_start_group_update` **packs, posts and increments
  `num_nonblock_group_update` immediately**; only `mpp_complete_group_update`
  decrements it and rewinds the shared buffer cursor. A started-but-never-
  completed handle therefore leaks a stack slot **and** MPI requests on every
  `k_split` iteration — so "start it and ignore it" is illegal, and the
  question had a definite answer rather than being a matter of taste.
- **Design A adopted: with the CUDA backend active, `fv_dynamics` does NOT
  ISSUE the `i_pack(1)`/`i_pack(8)` starts at all** (rather than starting and
  abandoning them). Both are **DUPLICATE** of exchanges the CUDA dyn_core
  already performs before the same consumers (`i_pack(1)` delp/pt/q_con ⇒
  `H_DSCALAR`; `i_pack(8)` u/v DGRID_NE ⇒ `H_DWIND_IN`). `i_pack(10)` (tracer
  halo) is **DISTINCT REQUIRED** — it is deliberately left outstanding across
  `dyn_core`'s exit and is still issued on the CUDA path. The start/complete
  balance was counted, not eyeballed: with the CUDA branch taken, every
  started handle is completed and none leaks.
- **The default Fortran path is source-semantically unchanged** — the skip is
  inside an `if (use_cuda_dyncore)` arm only.
- **Result: six MPI ranks, one tile per rank, `FV_NX=1`/`FV_NY=6`, c8/npz=10/
  n_split=3/k_split=1/R8, 6 dyn_core calls, both backends from the SAME
  executable in ONE Slurm allocation (job 58244118), differing only in
  `AGCM.rc`'s `RUN_CUDA_FV3`.** Both arms exit 0 in 25 s.
  **dyn_core ENTRY, call 1: BITWISE IDENTICAL, 6/6 ranks, all 13 state
  fields.** dyn_core EXIT: **all 17 fields ROUND-OFF SCALE or SMALL NUMERICAL
  DIFFERENCE at every sampled call** (1/2/3/6), worst `max_rel` 2.008e-11
  (always `w`, its own cancellation floor), with slow near-linear growth
  (`u` 3.5e-13 → 8.6e-13) — round-off accumulation, not divergence. Zero
  NaN/Inf over 6 calls × 6 ranks × 13 fields; owned-box min `delp`
  9967.18 → 9957.52 Pa.
- **The repeated-call divergence was a real defect and is fixed: `pe`'s one
  halo ring.** Call 1 was bitwise but call 2 was not (`pt` 1.032e-04 with the
  *same* max at every level, `u` 1.845e+00 at exactly `j=js`/`j=je+1`, `v` at
  exactly `i=is`/`i=ie+1`). Root cause: Fortran fills `pe`'s single halo ring
  inside `dyn_core` via `pe_halo` (`dyn_core.F90:1504-1532`, called at `:1011`
  under `remap_step`), CUDA never wrote it, and
  `Lagrangian_to_Eulerian`'s D-grid wind/flux remap **reads it**
  (`fv_mapz.F90:686/693` for u/mfy/cy, `:729/740` for v/mfx/cx) — exactly the
  measured footprint. **This corrects §10's standing claim that the ring is
  "proven INERT downstream of dyn_core".** Fixed by reproducing `pe_halo`'s
  two loops on the host after the D2H (`fv3_gpu_interface_pe_halo`); proven
  bit-for-bit equivalent because no `delp` write exists between
  `dyn_core.F90:1011` and `:1375`, so the ring is exactly
  `cumsum(ptop, exit delp ring)` and `delp`'s ring is already host-resident.
  No added transfer, no added kernel, `pe_halo` itself being `private`
  (`:142`). The secondary `pt`/`pkz` signature is **explained**: `consv_te =
  1.0`, so corrupted seam winds corrupted total energy, whose correction
  redistributes a *uniform* `pt` offset — which is why `pt` differed
  identically at every `k`.
- Other boundary-completion gaps closed in the same fix: the four flux
  capacitors `mfx`/`mfy`/`cx`/`cy` and `ws` were never copied out
  (`fv3_gpu_registry_export_flux_capacitors`/`_ws`); `z_rat`, `ut`/`vt` and
  `peln` descriptors had wrong bounds. Recorded honestly: the `ws` copy-out
  changed nothing measurably.
- **Numerical algorithm changed: NO.** **3 of 3 permitted substantial fixes
  consumed** (Design A start-skip; bridge/registry/ABI plumbing incl. the
  copy-outs and descriptor corrections; `fv3_gpu_interface_pe_halo`) — any
  further substantial change is a §12(4) stop.
- **Regression (job 58244300, single node / 6 tasks / 1 GPU): R8 CTest
  22/23.** The one failure is `fv3_halo_mpi_transport`. **[Prompt 18: now
  23/23, job 58246343 phase C — `100% tests passed, 0 tests failed out of 23`.]**
- **NEW OPEN DEFECT, found by that regression and characterized** — **RESOLVED
  by Prompt 18 (2026-09-04): it was a CUDA stream-ordering defect in the
  transport TEST's own device-buffer setup, not in the transport, and not an MPI
  contract violation. Three of the bullets below were measured wrong and are
  corrected inline.** The block is retained for its evidence trail; read
  `prompt_18_mpi_transport_determinism.md` for the resolution. The framing was
  right in one respect — the failure was PRE-EXISTING, NOT caused by Prompt 17,
  and Prompt 15's "16692 checks / 0 failures" was indeed a passing sample rather
  than a determinism guarantee. Evidence as measured in Prompt 17:
  - Intel MPI 2021.13 build (`build_cuda_r8_intelmpi`, job 58244365, 5
    repeats): **5/5 FAIL**, 1–2 failed checks each. **[Prompt 18: post-fix this
    exact configuration is 5/5 PASS, job 58246352. Note also that the task
    prompt's premise "Intel MPI: 5/5 PASS / OpenMPI intermittent" was
    INVERTED — Intel MPI was the *worse* stack pre-fix. The rate difference
    between stacks is timing, not semantics.]**
  - **Prompt 15's own OpenMPI 4.1.6 build (`build_cuda_r8_mpi`, job 58244385,
    5 repeats): 3/5 PASS, 2/5 FAIL** — that build directory and every
    transport source predate this pass entirely, which rules this pass out as
    the cause on provenance grounds (`fv3_gpu_halo_{exchange.cu,exchange.hpp,
    transport.hpp,mpi.cpp}` and the test itself are byte-unchanged since
    Prompt 15; this pass's edits are confined to the ABI/registry/bridge).
  - Signature is stable across every occurrence: the destination cell is
    always at **`h=1 a=1 k=1`** (first ring, first reported along-position,
    first level) of some `(edge, field)` block, always holds **exactly 0.0**
    (which decodes as `tile 0 (-8,-8,0)`, i.e. **unwritten**), and the
    afflicted `(rank, tile, edge, field)` **moves between runs** — observed on
    tiles 1/3/5, edges N and E, fields 0 and 1, in `H_DWIND_IN` and the
    `edge E field 0` wind slots. Scalar groups were never observed failing.
    **[Prompt 18 CORRECTION, measured: the value is NOT 0.0 — it is the test's
    own `0xFF` poison, which is NaN, and `std::llround(NaN)` → `LLONG_MIN`
    decodes through the oracle to the *identical* text `tile 0 (-8,-8,0)`, so
    the old output could not distinguish poison from a genuine zero. The
    "always `h=1 a=1 k=1`" and "scalar groups never fail" observations are both
    artifacts of the first-failure suppression corrected in the next bullet:
    with it removed, whole blocks fail, `h=1 a=1 k=1` is simply the first cell
    visited, and `H_DSCALAR` fails too.]**
  - **The true blast radius per occurrence is UNKNOWN and may be larger than
    one cell**, because the test reports only the first failure per
    `(edge, field)` and then `goto next_field` (`test_halo_mpi_transport.cu:328`)
    — the check-count drop (16692 → 16621/16609/16538) is exactly those
    skipped cells, so an entire ~71-cell block may be missing rather than one
    cell. **Do not assume it is one cell.** **[Prompt 18: correct instinct, now
    MEASURED. The suppression was removed; a failing occurrence loses a WHOLE
    field/edge message — all 72 written cells (3 rings × 6 along × 4 levels) of
    a 105-element recv slot, census `poison 105, zero 0, other 0` — at a
    correctly-computed slot offset, and `checks:` stays at the full 16692.]**
  - Ruled out by direct reading, so the next pass need not redo it: the
    send-side ordering in `fv3_gpu_halo_mpi.cpp:206-216` is correct — the D2H
    is `cudaMemcpyAsync` on the *caller's* stream (hence stream-ordered behind
    the pack the runtime already enqueued), and the wait is
    `cudaEventRecord`/`cudaEventSynchronize` on that same stream's event before
    any `MPI_Isend`. The pack→D2H race is therefore **not** the lead.
  - Recommended first diagnostics for the next pass (both cheap, both
    instrumentation): (1) remove the `goto next_field` suppression to learn the
    true failed-cell count per block — one cell vs a whole block points at very
    different mechanisms; (2) dump `slot_base`/`used_bytes` per `(group, edge,
    field)` on both donor and receiver to test a slot-offset/size disagreement.
  - **Scope of what this does and does not invalidate.** It does **not**
    retract Prompt 15's dyn_core-exit result (78/78 bitwise identical at 1 and
    5 calls) or this pass's standalone comparison (all 17 exit fields
    ROUND-OFF/SMALL over 6 calls) — those are measured outcomes and stand. It
    **does** retract the implied claim that the six-rank transport is
    deterministically correct, and it must be resolved before any longer
    validation run, because the same transport is what the standalone CUDA
    backend uses. See §9 item 18. **[Prompt 18: resolved — the transport itself
    was never the defect, so the retraction is itself retracted; the six-rank
    transport is now demonstrated deterministic at 24/24 consecutive OpenMPI
    runs. The longer-validation gate is lifted.]**

**Prompt 14 (2026-09-04) — standalone CUDA production integration: ATTEMPTED,
BLOCKED, ESCALATED. No source changed, no build run.** Task was to integrate
the Gate-C2-validated CUDA `dyn_core` backend, opt-in, into the real
`StandAlone_FV3_Dycore.x` executable under its real production MPI
decomposition (`FV_NX=1`/`FV_NY=6`, six separate MPI ranks, one tile per
rank). §1/§2 (working-tree confirmation and production call-boundary
mapping) were completed by direct source reading of
`fv_dynamics.F90:645-655` (the real `dyn_core` call site, inside the
`do n_map=1,k_split` loop) and `dyn_core.F90:166-263` (full signature); a
condensed field-by-field interface table is in
`docs/agent_handoff/prompt_14_standalone_cuda_integration.md` §2. Work then
stopped at a **genuine, source-confirmed architectural blocker**, found while
verifying the task's own §12 warning to check (not assume) MPI/tile mapping:
**the only implemented CUDA halo exchange transport,
`fv3::gpu::halo::HaloPeerGroup` (`cuda/dyn_core/include/fv3_gpu_halo_peer.hpp`,
self-described in its own header comment as "Model B — a single process
representing all six cubed-sphere tiles"), is single-process/six-host-thread
only** — it rendezvouses with a process-local `std::mutex`/
`std::condition_variable` and copies directly between raw device pointers
that are only valid within one address space. Confirmed by direct grep
that **no real MPI API call exists anywhere in `cuda/dyn_core/runtime` or
`cuda/dyn_core/include`** (the only two `MPI_*` string matches in
`fv3_gpu_runtime.cpp`, lines 829/1259, are comments, not code), and that
`fv3_gpu_create_desc::communicator_token` — the one public-ABI field that
looks purpose-built for a real communicator handle — is set and read
nowhere in the codebase outside its own struct definition. This is not a
new defect: `docs/agent_handoff/milestone_3a10_gpu_halo_exchange.md`
already documents Model B as a deliberate placeholder at design time
("Model B does **not** establish: MPI rank mapping, MPI request lifecycle",
line 471; "`HaloPeerGroup` is the only piece an MPI transport replaces",
line 955). **Running the real six-MPI-process production configuration
therefore requires implementing a new cross-process halo transport
first** (e.g. the host-staged real-MPI "Stage 1" already anticipated by
`project_architecture.md`'s Communication Evolution section, never yet
built for the CUDA side) — a genuine runtime redesign, not fixable within
this task's 3-substantial-integration-fix budget, and matching its own hard-
stop conditions (3) and (6). Escalated to Opus per the task's own
instruction rather than attempted on Sonnet. A secondary, non-blocking
finding worth reusing later: the public ABI (`fv3_gpu_api.h`) already has a
clean, Fortran-safe, opaque-handle lifecycle (create/topology/step/
finalize) built with this exact integration in mind, but per-run
`NumericalConfig` installation, real (`TopologyView`-based) topology
installation, and halo-peer-group attachment are currently only reachable
through internal, non-ABI C++ calls (`install_config`/`set_topology`/
`attach_halo_peer_group` in the benchmark and `fv3_gpu_internal.hpp`) — a
public ABI extension for these will be needed regardless of which halo-
transport direction is chosen. Full detail, the complete interface table,
and the two candidate transport directions (host-staged MPI vs. CUDA-aware
MPI/GPU-direct):
`docs/agent_handoff/prompt_14_standalone_cuda_integration.md`.

**Prompt 15 (2026-09-04) — cross-process MPI halo transport: COMPLETE. This
CLOSES the Prompt 14 blocker.** The CUDA dyn_core now has a real
cross-process halo transport, and it is **IMPLEMENTED, COMPILED, GPU EXECUTED
across six separate MPI processes, and NUMERICALLY VALIDATED** against the
in-process `HaloPeerGroup` path. Structure: one new interface
(`include/fv3_gpu_halo_transport.hpp`, `class HaloTransport`) whose only
transport-dependent surface is **`transfer()`** (make donors' packed bytes
readable, fill my recv) and **`release_send()`**, because the proven exchange
ordering is `compute → pack → record pack_done → [transfer] → unpack → record
xfer_done → order consumer → [release_send]` and nothing else depends on how
bytes cross. Rotation, sign, slot addressing, ring iteration, `dest_edges`
filtering and the pack/unpack kernels are **shared and untouched** — the MPI
backend contains **no edge-rotation formula of its own**. The in-process path
was adapted, not replaced (`PeerHaloTransport`, a thin adapter over
`HaloPeerGroup`); **no functionality was removed from `HaloPeerGroup`** and
both transports remain available with the default unchanged. Backend
(`runtime/fv3_gpu_halo_mpi.cpp`, `name() == "mpi_host_staged"`): two pinned
`cudaHostAlloc` staging buffers sized **once at attach** from the max over all
live groups (both memset at attach — the unused slot tail is literally put on
the wire); `transfer()` = one bulk D2H of the group's contiguous send region →
`cudaEventRecord`/`cudaEventSynchronize` on a `cudaEventDisableTiming` event
(**strictly narrower than a stream sync, and not a device sync — there is no
`cudaDeviceSynchronize` anywhere in the backend**) → `MPI_Irecv`/`MPI_Isend` ×
(4 edges × fields) with `tag = ((group*4)+edge)*max_halo_group_fields + field`
(max 207, inside the guaranteed `MPI_TAG_UB ≥ 32767`) → one `MPI_Waitall` →
one bulk H2D; persistent device residency preserved (halo slots only, never
model state). `release_send()` is a justified no-op (`MPI_Waitall` already
completed). A cross-rank `agree_on_layout` (cached per-group MIN/MAX
`MPI_Allreduce` on field count) guards positional field compaction, which is
configuration-dependent (`H_DIVGD` only at `nord>0`, `H_HEAT` only at
`d_con>0`) and would otherwise let two ranks read each other's buffers at the
wrong offset. Public API (`include/fv3_gpu_communication.h`):
`FV3_GPU_HALO_TRANSPORT_{NONE,IN_PROCESS_PEER,MPI_HOST_STAGED}`,
`fv3_gpu_attach_mpi_halo_transport`, `fv3_gpu_detach_halo_transport`,
`fv3_gpu_halo_transport_kind_of` — **no C++ or MPI type in the ABI**; the
communicator crosses as the **Fortran INTEGER handle** (`MPI_Comm` is `int` in
MPICH but an opaque pointer in Open MPI, so it cannot appear in a stable C
ABI) and is converted with `MPI_Comm_f2c`. Built without
`FV3_DYN_CORE_ENABLE_MPI_HALO` (default OFF), the attach entry point returns
`FV3_GPU_UNSUPPORTED_CONFIGURATION` rather than failing to link. Rank↔tile
mapping is the identity **rank r ↔ tile r+1** and is **verified, not trusted**
(allgathered tile map with duplicate/range checks; per-neighbour
`tile_of_rank_[nb.rank] == nb.tile`) — confirmed at runtime by 12 banner lines
`rank 0 owns tile 1` … `rank 5 owns tile 6`. Evidence: **six-rank transport
test (`tests/integration/test_halo_mpi_transport.cu`, `mpirun -np 6`, one tile
per rank) PASS with `checks: 16692  failures: 0`** using
`test_halo_exchange_multitile.cu`'s encode/decode provenance oracle (a
received value proves *which donor cell* it came from) and a **LOCAL,
non-circular** sign oracle, covering scalar (`H_W_IN`), **both** wind
staggerings (`H_DWIND_IN` DGRID_NE **and** `H_CWIND` CGRID_NE — the transposed
slots, which is the pair that distinguishes a slot-keyed from a
staggering-keyed sign), the 4-field width-3 multi-ring group (`H_DSCALAR`),
and **E11** (`H_DWIND_SEAM`, `ring_begin=0`, owned destinations). Then the
decisive result: **the six-MPI-process dyn_core exit is BITWISE IDENTICAL to
the single-process `HaloPeerGroup` path — 78/78 files (13 exit fields × 6
tiles), `DIFFERS=0 MISSING=0`, at 1 dyn_core call AND at 5 calls / 15
substeps** (job 58240534, `warpa003`), from the **same binary, same flags,
same authoritative Gate-C2 SP0 state/geometry/config, same node, same job —
the only difference being `--mpi-halo`**. Because the in-process path is the
one Gate C2 validated against authoritative Fortran, **the Gate C2 numerical
result carries over to the MPI transport unchanged for this configuration**,
with no new Fortran run needed. Regression (job 58240280): R8 ctest **22/22
PASS with MPI OFF** and **23/23 with MPI ON** (new `fv3_halo_mpi_transport`
passes in 1.87 s), plus a single-process transport smoke confirming the
`PeerHaloTransport` refactor left the default path working end to end. Zero
NaN/Inf anywhere. **Numerical algorithm changed: NO.** **Zero of the 3
permitted transport fixes consumed.** No performance matrix was run and the
smoke wall-clock numbers are explicitly **not** a performance claim. Full
detail, including the per-group contract table, the corner classification
(nothing corner-specific is sent — the diagonal corner halo cells have no
donor tile, and FV3's corner *rotations* are intra-tile device operations),
and the explicit E11 mapping:
`docs/agent_handoff/prompt_15_mpi_halo_transport.md`.

**Prompt 16 (2026-09-04) — standalone CUDA production integration, resumed
from the Prompt 15 unblock: PARTIAL, ESCALATED on a NEW finding.** Resumed
exactly where §9 item 15 (above) said to. §1 (working trees) reused unchanged
from Prompt 14. **§2 (public C ABI) is now COMPLETE and verified**: the two
previously-unimplemented ABI functions declared in `fv3_gpu_api.h` —
`fv3_gpu_set_numerical_config` and `fv3_gpu_install_production_dispatch` —
now have real implementations, and §3/§4 (geometry/topology installation)
needed **no new ABI**, since `fv3_gpu_set_cubed_sphere_topology` (Milestone
3A.10) and the existing state/field-descriptor path already cover it. A real
CMake constraint governed placement: `fv3_gpu_install_production_dispatch`
must call `fv3_build_production_dispatch()`, which lives only in the
`fv3_production_dispatch` static-library target, and that target already
links **PUBLIC** against `fv3_dyn_core_gpu` — the reverse link (needed to call
it from `fv3_gpu_runtime.cpp`, part of `fv3_dyn_core_gpu`) would be a
static-library dependency cycle the CMakeLists.txt's own comment already
warns against, confirmed by an actual linker failure
(`undefined reference to fv3::gpu::fv3_build_production_dispatch`) when tried
there first. Fixed by defining it instead in
`cuda/dyn_core/runtime/fv3_gpu_production_dispatch.cu`, right after
`fv3_build_production_dispatch`'s own closing brace — **1 of 3 permitted
integration fixes consumed** (build/link-integration category). Verified: a
full clean rebuild (`build_cuda_r8_mpibench`) succeeded with zero errors;
login-node CTest 13/13 host-only PASS (10 GPU-labeled tests correctly
`***Skipped`); one `gpu_a100` Slurm CTest run **22/23 PASS**, with the sole
failure (`fv3_halo_mpi_transport`) attributable to an ad hoc `srun`
resourcing artifact (6 requested tasks landed on 4 separate single-GPU nodes
instead of the one node the test's design requires) rather than a code
regression — neither new ABI function touches the MPI transport path, and
this exact test already passed at 16,692/0 under correct single-node
resourcing in Prompt 15. **Then, while reading the real production call site
in Repo B to start §5 (the Fortran `ISO_C_BINDING` wrapper), a SECOND, DISTINCT
halo-ownership ambiguity was found and the pass STOPPED and escalated on it**,
per the task's own instruction (§10, "careful FMS-vs-CUDA halo ownership —
STOP and escalate if ambiguous"): `fv_dynamics.F90:616-655` (the
`do n_map=1,k_split` loop) itself **starts** two asynchronous FMS
`start_group_halo_update` calls — `i_pack(1)` for `delp`/`pt`/`q_con`,
`i_pack(8)` for `u`/`v` with `gridtype=DGRID_NE` — immediately *before*
calling `dyn_core`, then passes the same `i_pack(*)` array into `dyn_core` as
`intent(inout)` for `dyn_core` to *complete* internally at specific points.
This is a real, load-bearing FMS halo-pipelining layer **above** `dyn_core`'s
own internal six-tile GPU halo exchange (`PeerHaloTransport`/
`MpiHaloTransport`) — not the same mechanism, and not previously read by any
prompt from 6A through 15, all of which scoped their reading to inside
`dyn_core.F90`'s own savepoint boundaries. It is consistent with, and now
explains, an already-on-record Gate C2 detail (§4 `sp2c`: "the `u`/`v` halos
are in flight (`i_pack(8)` started, not completed)"). Two correctness-relevant
questions follow that neither `dyn_core.F90` nor the CUDA runtime can answer
alone, and guessing wrong on either risks silently corrupting Fortran-side
state in a mixed-backend run (the opposite failure mode of everything Gate C2
validated): (1) if the CUDA branch simply skips issuing these
`start_group_halo_update` calls (since CUDA does not consume the `i_pack`
handles), does anything else in `fv_dynamics.F90` — a later `n_map` iteration,
a diagnostic, FMS-internal slot bookkeeping — assume slots `i_pack(1)`/`(8)`
were started/completed; (2) if the calls are left in place but never
completed, does FMS's `mpp_domains` group-halo-update machinery tolerate a
started-but-never-completed handle being reused next iteration, or does it
deadlock/corrupt state? Resolving this safely requires reading FMS's
halo-update-handle contract beyond `dyn_core`'s call boundary — out of this
pass's 3-fix integration budget and Sonnet-appropriate scope per the task's
own model policy. **No source in either repository was modified to
investigate this** (read-only: `fv_dynamics.F90:600-670`,
`dyn_core.F90:160-274`, both reproduced verbatim in the handoff for reuse).
**Numerical algorithm changed: NO. 1 of 3 integration fixes consumed (the ABI
placement fix, §7.1). Escalating per the task's own instruction rather than
guessing on Sonnet.** Full detail: `docs/agent_handoff/prompt_14_standalone_cuda_integration.md`
§7 (new section, appended — §0–§6 are the original Prompt 14 content,
unchanged).

## 1. Git State

- Repository: `FVdycoreCubed_GridComp`
  - Primary path: `/discover/nobackup/projects/imvi-nse/jli30/FVdycoreCubed_GridComp`
  - Also visible at: `/gpfsm/dnb06/projects/p295/jli30/FVdycoreCubed_GridComp`
- Branch: `cuda/gpu-bringup`
- HEAD: `eb6d68913c932f00c87c86ff8c7b891d66f6214f` ("Optimize CUDA dyn_core
  launch orchestration") — this commit was made by the user between agent
  sessions (not by an agent pass) and consolidates the previously-uncommitted
  **Prompt 12** `a2b4_global_corner_kernel` restructuring and **Prompt 13**
  CUDA-Graph launch repair. The prior HEAD `7dda37b` ("Optimize FV3 CUDA halo
  edge launches", also user-authored) consolidated Prompt 11's
  `--fuse-halo-edges` work.
- Ahead of `origin/cuda/gpu-bringup` (`191157f`) by 3 commits (`b456897`,
  `7dda37b`, `eb6d689`) — **not pushed**.
- Tracked working tree: **modified, uncommitted — all of it from Prompt 15**
  (the Prompt 12/13 changes previously listed here are now committed in
  `eb6d689`). Modified: `cuda/dyn_core/CMakeLists.txt` (+75),
  `cuda/dyn_core/benchmarks/run_small_perf_bench.cpp` (+208, the `--mpi-halo`
  mode), `cuda/dyn_core/include/fv3_gpu_communication.h` (+45, the public
  transport-selection API), `cuda/dyn_core/include/fv3_gpu_context.hpp` (+23),
  `cuda/dyn_core/include/fv3_gpu_halo_exchange.hpp` (+15),
  `cuda/dyn_core/include/fv3_gpu_internal.hpp` (+20),
  `cuda/dyn_core/runtime/fv3_gpu_halo_exchange.cu` (+11),
  `cuda/dyn_core/runtime/fv3_gpu_runtime.cpp` (+295, the `HaloTransport`
  refactor of `perform_peer_exchange`/`resolve_halo`/attach/detach). Also
  modified: this file. New and untracked, all Prompt 15 source:
  `cuda/dyn_core/include/fv3_gpu_halo_transport.hpp` (the interface),
  `cuda/dyn_core/include/fv3_gpu_halo_mpi.hpp` (MPI-type-free factories),
  `cuda/dyn_core/runtime/fv3_gpu_halo_mpi.cpp` (the host-staged MPI backend),
  `cuda/dyn_core/tests/integration/test_halo_mpi_transport.cu` (the six-rank
  test). All validated — see the Prompt 15 lead paragraph and
  `prompt_15_mpi_halo_transport.md`: R8 ctest 22/22 PASS with MPI OFF and
  23/23 with MPI ON, six-rank transport test PASS (16,692 checks / 0
  failures), and the six-MPI-process `dyn_core_exit` **78/78 bitwise
  identical** to the in-process path at 1 and 5 calls. The MPI path is
  opt-in twice over (CMake `FV3_DYN_CORE_ENABLE_MPI_HALO=OFF` by default, and
  `--mpi-halo` off by default), so the default path is provably unchanged.
  Not committed, not pushed — do so only if explicitly requested.
- **New in Prompt 18, uncommitted: exactly two modified files.**
  `cuda/dyn_core/tests/integration/test_halo_mpi_transport.cu` — stream-ordered
  poison memsets and field-array H2D (the fix), the `POISON-REMAINS`/`ZERO`/
  `WRONG-VALUE` `classify()`, removal of the `goto next_field` first-failure
  suppression, per-block `BLOCK FAIL … slot census` diagnostics, the owned-box
  check rewritten from early-exit to counting (`OWNED-BOX MOVED`), and the
  `FV3_HALO_TEST_DEBUG` / `FV3_HALO_TEST_LEGACY_SETUP` env switches; and
  `cuda/dyn_core/runtime/fv3_gpu_runtime.cpp` (~:1290-1291) — attach-time halo
  buffer clears converted to `cudaMemsetAsync` on `d.transfer_stream`, with the
  reasoning appended to the existing milestone-3A.10 §20 comment. New handoff
  doc `docs/agent_handoff/prompt_18_mpi_transport_determinism.md` (commit with
  the source). **Do not commit** the job scripts `p18_phaseA.sbatch`,
  `p18_accept.sbatch`, `p18_intelmpi.sbatch`, `p18_transport.sbatch` or any
  `p18_*.out` log. Nothing was committed or pushed.
- **New in Prompt 17, all uncommitted, in TWO repositories.** In this repo
  (`FVdycoreCubed_GridComp`): new Fortran bridge layer
  `cuda/interface/fv3_gpu_dyncore_bridge.F90`, `fv3_gpu_abi_mod.F90`,
  `fv3_gpu_field_registry.F90` plus `cuda/interface/CMakeLists.txt`; new
  `cuda/dyn_core/runtime/fv3_gpu_registry_shim.cpp`; modified
  `cuda/dyn_core/include/fv3_gpu_api.h`,
  `cuda/dyn_core/runtime/fv3_gpu_runtime.cpp`,
  `runtime/fv3_gpu_production_dispatch.cu`,
  `cuda/dyn_core/operators/wp_d_contract.cpp`, `cuda/dyn_core/CMakeLists.txt`;
  build/run helpers `cuda_intelmpi_build.sh`, `p17_rebuild_exe.sh`,
  `p17_rebuild_all.sh`, `p17_regress.sbatch`, `p17_halo_repeat.sbatch`,
  `p17_halo_ompi.sbatch`. In the **fixture** repo's nested `@fvdycore`
  (`GEOSfvdycore_fortran_perf/src/Components/@FVdycoreCubed_GridComp/@fvdycore`):
  modified `model/fv_dynamics.F90` (the Design A backend branch + the `dce`/`dcx`
  savepoint calls), `model/dyn_core.F90` (savepoints only, as before),
  `CMakeLists.txt` (link the CUDA archives), and new
  `model/fv3_savepoint_mod.F90`. **Do not commit** `build_cuda_r8_intelmpi/`,
  the `experiments/` trees, `*.log`, or any `p17_*.out` job log. Nothing was
  committed or pushed.
- **New in Prompt 16, also uncommitted:** `cuda/dyn_core/include/fv3_gpu_api.h`
  (+2 declarations: `fv3_gpu_set_numerical_config`, `fv3_gpu_install_production_dispatch`,
  +the `fv3_gpu_numerical_config` struct), `cuda/dyn_core/runtime/fv3_gpu_runtime.cpp`
  (+the `fv3_gpu_set_numerical_config` implementation), and
  `cuda/dyn_core/runtime/fv3_gpu_production_dispatch.cu` (+the
  `fv3_gpu_install_production_dispatch` implementation — placed here, not in
  `fv3_gpu_runtime.cpp`, to avoid a static-library dependency cycle; see the
  Prompt 16 lead paragraph). Full clean rebuild verified; R8 ctest 22/23 GPU +
  13/13 host (one unrelated resourcing artifact, not a regression — see Prompt
  16 lead paragraph). Not committed, not pushed.
- Untracked at repo root: many one-off scratch shell scripts/logs from prior
  passes (`task*_6c_cuda.sh`, `sp*rerun*.sh`, `*_3a11*.py`, `cln*.txt`,
  `cmp_nord*.txt`, `det_diff_*.txt`, `pre*.txt`, `rprobe_6b`, `scratch_6b/`,
  build helper scripts like `omega_build*.sh`/`r8ctest_final.sh`), plus
  `tools/` (savepoint validation tooling, e.g. `verify_omega_host.cpp`,
  `verify_adv_pe_geometry.cpp`, `compare_cuda_fortran.py` — reusable, not
  scratch) and `perf_rerun_logs/` (sbatch scripts + job logs for Prompts
  7/8/9/11/12, including new `prompt12_a2b4_tuning.sbatch`,
  `prompt12_a2b4_kernel.patch`, their `*_58238626.out` log, and
  `ncu_p12_{before,after}.csv` — the latter two contain only an
  `ERR_NVGPUCTRPERM` permission-denial message, not real profiling data; see
  `prompt_12_a2b4_kernel_tuning.md` §2). New this pass:
  `perf_rerun_logs/prompt13_graph_repair.sbatch`, its
  `prompt13_graph_repair_58238910.out` log, `nsys_p13_{before,after}.nsys-rep`
  and their generated `.sqlite` (**profiler artifacts — do not commit**), and
  `docs/agent_handoff/prompt_13_architectural_optimization.md`. Also untracked
  from prior passes: `docs/agent_handoff/prompt_12_a2b4_kernel_tuning.md`,
  `prompt_10_copy_corners_batching.md`,
  `docs/agent_handoff/prompt_11_next_launch_target.md`,
  `docs/agent_handoff/prompt_14_standalone_cuda_integration.md`. **New this
  pass (Prompt 15), untracked:**
  `docs/agent_handoff/prompt_15_mpi_halo_transport.md` (the handoff — commit
  with the source), plus **artifacts that must NOT be committed**:
  `build_cuda_r8_mpi/` and `build_cuda_r8_mpibench/` (build dirs),
  `scratch_p15/` (the two dyn_core_exit dump trees compared bitwise in §14),
  `perf_rerun_logs/prompt15_mpi_halo.sbatch` +
  `prompt15_mpi_halo_58240280.out`, and
  `perf_rerun_logs/prompt15_mpi_dyncore.sbatch` +
  `prompt15_mpi_dyncore_58240534.out`. Do not commit or
  delete any of these without checking they are not cited as reproduction
  scripts by a milestone doc; do not commit or push unless explicitly
  requested.

## 2. Authoritative Fortran Source

- `GFDL_atmos_cubed_sphere`, GEOS-ESM fork, tag `geos/v3.0.0`,
  SHA `ca30e26305fda43831095334c02a3f5e0b8c0c45`.
- `@fvdycore` is **NOT populated** in this CUDA repo working tree. Do not
  populate it automatically per `docs/agent_instructions/discover_platform.md`.
- Authoritative standalone Fortran fixture (separate tree):
  `/discover/nobackup/projects/imvi-nse/jli30/GEOSfvdycore_fortran_perf`
  - Fixture repo HEAD `8ba7bdd4f7f647cb6e6d70060c131e20938f1f0c`
  - `@FVdycoreCubed_GridComp` layer HEAD `ac43844bae40dcc9558deda85c7aeebb4d06cf41`
  - `@fvdycore` SHA matches the pinned `ca30e26...` above.
  - Executable: `install/bin/StandAlone_FV3_Dycore.x`, Intel `ifort` build.
  - Its nested `@fvdycore/model/dyn_core.F90` carries **uncommitted
    instrumentation only** (savepoints; no production numerical change): the
    Prompt 6A SP0/SP1/SP2/SP3 blocks plus the Prompt 6B Task-J `sp2a`
    (`:986-995`), `sp2b` (`:1097-1103`), `sp2c` (`:1283-1290`) blocks and the
    `crx/xfx/cry/yfx` additions to SP2 (`:884-887`).

## 3. Target Architecture

```
GEOS/FV3 Fortran -> ISO_C_BINDING / stable C ABI -> WP-E GPU context
    -> persistent device-resident CUDA dyn_core (WP-A/B/C/D)
    -> six-tile GPU-side cubed-sphere halo exchange
           -> HaloTransport: { PeerHaloTransport | MpiHaloTransport }
```
CUDA backend optional; Fortran backend preserved. See
`docs/agent_instructions/project_architecture.md` for full detail (unchanged).

**Prompt 15 addition:** the halo exchange is now split at a
`HaloTransport` interface (`include/fv3_gpu_halo_transport.hpp`) with two
interchangeable backends — `PeerHaloTransport` (in-process, six host threads,
the historical "Model B" `HaloPeerGroup`; still the default) and
`MpiHaloTransport` (host-staged real MPI, six separate processes, one tile per
rank — the production decomposition). Topology, pack/unpack kernels, rotation
and sign logic are shared; only `transfer()`/`release_send()` differ. This is
the concrete realization of `project_architecture.md`'s "Stage 1" host-staged
MPI; Stage 2 (CUDA-aware MPI) and Stage 3 (comm/compute overlap) become new
backends behind the same interface, not a redesign.

## 4. Implemented CUDA dyn_core Coverage

- Full 54-stage production first-run inventory (WP-E 10, WP-A 7, WP-B 18,
  WP-C 9, WP-D 10): **IMPLEMENTED**, **COMPILED**, **GPU EXECUTED**.
- n_split acoustic loop: IMPLEMENTED, GPU EXECUTED (n_split=3 authoritative).
- Six-tile GPU-side halo exchange: IMPLEMENTED, GPU EXECUTED, liveness
  confirmed by `fv3_dispatch_smoke` Phase E causality test (R8 and R4).
- **Cross-process (real MPI) six-tile halo exchange (Prompt 15): IMPLEMENTED,
  COMPILED, GPU EXECUTED across six separate MPI processes, and NUMERICALLY
  VALIDATED against the in-process path — `dyn_core_exit` 78/78 (13 fields × 6
  tiles) BITWISE IDENTICAL at 1 and at 5 dyn_core calls.** Opt-in twice
  (CMake `FV3_DYN_CORE_ENABLE_MPI_HALO`, default OFF; bench `--mpi-halo`,
  default off). Detail: the Prompt 15 lead paragraph and
  `prompt_15_mpi_halo_transport.md`.
- Persistent device residency: IMPLEMENTED, GPU EXECUTED.
- Per-level damping translation layer (`fv3_damping_translation.{hpp,cpp}`):
  IMPLEMENTED, COMPILED, host-unit-tested.

### NUMERICALLY VALIDATED — all three acoustic substeps, plus 12/13 exit fields

Configuration for every result below: c8 / npz=10 / n_split=3 / k_split=1 / R8 /
6 MPI ranks = 6 tiles, started from authoritative SP0 state with authoritative
grid metrics, `da_min`/`da_min_c`, `bdt=3600`, `ptop`/`phis`/`zs`/`dp_ref`,
`rdgas`, `n_sponge=-1`, authoritative `edge_*`, and `akap_exner`. Verdict boxes
are the owned/defined boxes; the four cubed-sphere **diagonal corner halo** cells
are reported separately and excluded (they have no donor tile). Worst value
across all 6 tiles is quoted. Grading: BITWISE / ROUND-OFF SCALE (≤1e-13) /
SMALL NUMERICAL DIFFERENCE (≤1e-8) / SIGNIFICANT (>1e-8).

**SP1 `it=1`** (post-C-grid predictor, `dyn_core.F90:636-643`) — **PASS**.
Validates `c_sw`, `update_dz_c`, `Riem_Solver_c`, `p_grad_c`.

| field | max_abs | max_rel | verdict |
|-------|---------|---------|---------|
| `delpc` | 1.82e-12 | 1.82e-16 | ROUND-OFF |
| `ptc` | 3.55e-15 | 1.44e-16 | ROUND-OFF |
| `gz` | 2.91e-10 | 6.34e-16 | ROUND-OFF |
| `pkc` | 2.91e-11 | 2.91e-16 | ROUND-OFF |
| `uc` | 9.28e-14 | 6.44e-15 | ROUND-OFF |
| `vc` | 9.92e-14 | 6.89e-15 | ROUND-OFF |
| `ws3` | 9.47e-17 | 4.97e-13 | SMALL |

Caveat retained: interior `gz` levels validate `Riem_Solver_c`'s output, not
`update_dz_c`'s, because `Riem_Solver_c` overwrites `gz` in full. `ws3`'s max_abs
is below its own cancellation floor but is recorded as SMALL, not claimed as
round-off. SP0 identity (zero numerical kernels executed) is BITWISE, 60/60
fields, 6 tiles.

**SP2 `it=1`** (post-`d_sw`, `:869-887`) — **PASS, 14/14 fields.** All ROUND-OFF
SCALE on all 6 tiles: `divgd`, `cx`, `cy`, `mfx`, `mfy`, `delp`, `pt`, `u`, `v`,
plus the four fields added this pass — `crx`, `cry`, `xfx`, `yfx`. `w ≡ 0`
identically on both sides here because `d_sw` does not update `w`, so its
agreement is **vacuous** and honest `d_sw` coverage is 9/10 — but `w` is no
longer uncovered overall, see `sp2a`. `cx`/`cy` are declared over the halo range
in their non-staggered direction and their verdict is a claim about the owned box
only. Adding `crx/cry/xfx/yfx` **closes `update_dz_d`'s input contract**.

**`sp2a` `it=1`** (post-`Riem_Solver3`, `:986-995`) — **PASS.** First
authoritative coverage of `Riem_Solver3`/`SIM1_solver` at any `it`, and of
`update_dz_d`'s output (`ws` is `intent(IN)` to `Riem_Solver3`).

| field | max_abs | max_rel | verdict |
|-------|---------|---------|---------|
| `zh` | 2.91e-11 | 6.27e-16 | ROUND-OFF |
| `delz` | 3.64e-11 | 1.22e-15 | ROUND-OFF |
| `pk3` | 2.13e-14 | 7.94e-16 | ROUND-OFF |
| `delp` | 3.64e-12 | 3.64e-16 | ROUND-OFF |
| `pt` | 7.11e-15 | 2.88e-16 | ROUND-OFF |
| `pkc` | 4.03e-14 | 4.52e-12 | SMALL |
| `w` | 4.74e-14 | 8.36e-12 | SMALL |
| `ws` | 3.55e-17 | 1.94e-12 | SMALL |

**`sp2b` `it=1`** (post-`nh_p_grad`, `:1097-1103`) — **PASS.** Validates
`nh_p_grad` at `it=1`. `u` 2.26e-13 / 5.81e-13 SMALL; `v` 2.03e-13 / 5.22e-13
SMALL. The `max_rel` ranking is driven by pe4, whose `u` genuinely maxes near
0.39, so the dynamic-range floor is small there.

**`sp2c` `it=1`** (end of acoustic substep, the state carried into `it=2`,
`:1283-1290`) — **PASS.** `delp` 3.64e-12, `pt` 7.11e-15, `delz` 3.64e-11 all
ROUND-OFF; `u` 2.26e-13, `v` 2.03e-13, `w` 4.74e-14 all SMALL. The CUDA
comparand is the `nh_pressure` stage dump; that is valid on source evidence, not
assumption — at `it < n_split` nothing after `nh_p_grad` writes
`u,v,w,delz,delp,pt` (E11 is `it==n_split`-gated at `:1178`, the omega block is
`last_step`-gated, `heat_source` is applied only after the loop).

**The full in-loop grid — all 15 boundaries PASS.** Every
`{SP1, SP2, sp2a, sp2b, sp2c}` × `{it=1, it=2, it=3}` combination grades at or
better than SMALL NUMERICAL DIFFERENCE (SP2 `it=1` is ROUND-OFF SCALE), worst
value across all 6 tiles, same exclusions as above. Representative `it=2` SP1
values: `delpc` 3.64e-12, `ptc` 1.07e-14, `gz` 4.66e-10, `pkc` 2.91e-11, `uc`
1.63e-13, `vc` 1.94e-13 ROUND-OFF; `ws3` 9.47e-17 / 5.17e-13 SMALL. Re-verified
with the post-E11/post-heat-accumulation build, and again with the post-`nf_ke`
build (15/15 still PASS, no regression). Per-boundary detail:
`prompt_6b_cuda_fortran_validation.md` §20.1 and §21.6.

**dyn_core exit (`sp3`, `it=3`) — 13 of 13 fields validated.** Worst across all
6 tiles. `w` and `delz` receive genuinely **non-vacuous** validation here (their
SP2 agreement was vacuous because `w ≡ 0` there). No exit field is vacuous.

| field | max_abs | max_rel | verdict |
|-------|---------|---------|---------|
| `u` | 4.086e-13 | 4.110e-13 | SMALL |
| `v` | 3.855e-13 | 4.795e-13 | SMALL |
| `w` | 9.509e-14 | 1.413e-11 | SMALL |
| `delz` | 3.638e-11 | 1.215e-15 | ROUND-OFF |
| `delp` | 1.091e-11 | 1.090e-15 | ROUND-OFF |
| `pe` (owned box) | 4.366e-11 | 4.358e-16 | ROUND-OFF |
| `peln` | 1.776e-15 | 1.543e-16 | ROUND-OFF |
| `pk` | 2.487e-14 | 9.266e-16 | ROUND-OFF |
| `pkz` | 1.421e-13 | 5.374e-15 | ROUND-OFF |
| `ua` | 3.268e-13 | 2.214e-14 | ROUND-OFF |
| `va` | 2.807e-13 | 1.765e-14 | ROUND-OFF |
| `pt` | 1.421e-14 | 5.757e-16 | ROUND-OFF |
| `omga` | 4.875e-14 | 1.506e-12 | SMALL |

`pt` reached this state in the `nf_ke` pass (§4 #12/#13 below), from a previous
6.397e-05 / 2.592e-06 SIGNIFICANT. Per-tile detail, all ROUND-OFF SCALE:
max_abs 1.421085e-14 (pe0/pe1/pe3), 1.065814e-14 (pe2/pe4/pe5); guarded max_rel
5.756846e-16 / 4.28…4.32e-16; rms ~2.5e-15; mismatch 395–445 of 640; worst
location e.g. pe0 `(7, 5, 0)`. `pkz` moved with it — `dyn_core.F90:1348`
rewrites `pkz` at every heated level inside the same NH heating block.

`pe`'s single Fortran halo ring is also **NOT WRITTEN** by CUDA (holds the
harness 1e-15 poison sentinel, ring max|diff| 1.002e+05); the comparator box was
narrowed to the owned box and the ring classified separately.

**Run health.** Zero NaN/Inf anywhere (`nan/inf=0/` on every `[FV3DBG]` line), no
NONFINITE or FAIL lines, harness exit code 0. The previously recorded "`it=3`
NONFINITE on tiles 2/3/4" symptom no longer appears, and `it=3` is now
**compared and passing** at all five in-loop boundaries.

`omga` reached this state in the `omga` finalization pass (see "Missing
authoritative behaviour now IMPLEMENTED" below), from a previous
4.740e-02 / 1.003e+00 NOT WRITTEN. Per-tile max_abs
2.53e-14…4.88e-14, guarded max_rel 7.8e-13…1.51e-12, rms 8.07e-15…1.07e-14,
mismatch 639–640 of 640. Essentially every cell differs in its last bits by
construction: `omga` is the difference of two ~1e5 Pa pressures divided by
`dt = 1200` s, whose cancellation floor is ≈1.2e-14, so this grades SMALL and is
**not** claimed as ROUND-OFF. The comparison box is `is:ie / js:je`, which is the
authoritative write footprint exactly (`dyn_core.F90:1207-1213` plus `adv_pe`'s
own `is..ie / js..je`) — Fortran does not write `omga`'s halo ring in the
finalization either, so `omga`'s "full declared bounds" figure grades SIGNIFICANT
for the same reason `ua`/`va`'s already did.

**Scope of these claims.** All three acoustic substeps end-to-end, plus **all 13
dyn_core-exit fields**. Gate C2 is CLOSED for this configuration (c8 / npz=10 /
n_split=3 / k_split=1 / R8 / six tiles / production first-run numerics). Do not
extend the claim to other grids, precisions, or configurations, and do not treat
the 54-stage GPU-executed inventory as numerically trustworthy beyond the
results above.

### Input-fidelity / layout defects found and fixed, in order of discovery

Thirteen total. The first seven are host-side and **opt-in** (validation-only
flags; the default synthetic/performance path is provably unchanged). The eighth
is opt-in. **The ninth, tenth and eleventh change the default path too** — see
their entries.

1. `bdt` = 10.0 vs authoritative **3600.0** → `--bdt`.
2. `ptop` = 1000.0 vs authoritative **100.0** → `--sp1-boundary`.
3. Surface topography `zs`/`phis` identically 0 vs authoritative real
   topography → `--sp1-boundary`.
4. `dp_ref` from the synthetic log-spaced column (`bot_ratio` 0.613137) vs
   authoritative pure-sigma **0.500000000000** → `--sp1-boundary` (fixed `ws3`
   by ~2.8e11×).
5. `rdgas` = 287.0 vs authoritative FMS/GEOS **8314.47/28.965 =
   287.0523045054** → `kFv3Rdgas` (fixed `gz` 3.3e-04 → 6.3e-16). Traced through
   three levels of build evidence, not guessed: `nh_utils.F90:53` →
   `FMS_DIR` = `ESMA-Baselibs-8.32.0` → `-DCONSTANTS=GEOS` →
   `geos_constants.fh:44`.
6. `n_sponge` = 9 (hardcoded in `geos_production_flags()`) vs authoritative
   **−1** → `--damp-n-sponge` (corrupted 9 of 10 damping levels; fixed SP2 `it=1`
   `u`/`v` 1.86e-02 → 9.42e-06). Verified by
   `tools/savepoints/verify_damping_column.py`.
7. `edge_w/e/s/n` fabricated as `1.0 + 0.01*index` (1.00…1.13, outside the [0,1]
   range an interpolation weight can occupy at all) vs authoritative great-circle
   weights `0.298255…0.701745` → `namespace grid_geometry_6b`, a verbatim
   `fv_grid_utils.F90:1188-1293` `edge_factors` port, gated on `--grid-metrics`.
   Fixed SP2 `it=1` `u`/`v` 9.42e-06 → 4.23e-16.
8. **Two-kappa split.** The scalar `akap` driving the **Exner** variable only
   (`ptk`, `pk3`, `pk3_halo`) is the authoritative single-precision `MAPL_KAPPA`
   = `0.28571426868438721`, while the `cappa(i,j,k)` array driving every
   thermodynamic closure is FMS double dry `2/7`. Implemented as
   `NumericalConfig::akap_exner` + `--akap-exner`; `cfg.akap` stays `2/7`. Two
   earlier levers were **retracted**: a `--moist-cappa` route (rejected at
   runtime) and a *global* `--akap` override (fixed `pk3` but degraded
   `zh`/`delz`/`w`/`pkc` and would have regressed the validated SP1 `it=1`).
9. **`grid_lonlat` / `agrid_lonlat` store order — the SP1 `it=2` root cause.**
   `make_host_state` filled both arrays **interleaved** (`idx*2 + component`)
   while they are registered **component-slowest**
   (`st_ll = {1, ni_h, ni_h*nj_h}`, the correct translation of Fortran's
   `grid(isd:ied+1, jsd:jed+1, 2)`). The consumer
   (`a2b_great_circle` → `View3` component index `lo_k`/`lo_k+1`) therefore read
   lon/lat of the wrong cells, mixed together. **The only consumer of these
   arrays is `a2b_ord4`'s corner extrapolation** — the interior, edge and
   `qx`/`qy`/`qxx`/`qyy` paths use `dxa`/`dya` — which is exactly why the damage
   was confined to the four physical panel corners of `nh_p_grad`'s B-grid
   pressure fields while every interior cell agreed at round-off, and why seven
   prior audits (which checked *values*, never *store order*) missed it. Fixed by
   filling component-slowest. Corner values: `pkc` 3.73e+00 → ≤2.71e-14, `gz`
   6.74e+05 → ≤1.16e-10. **Unlike 1-8, this is NOT opt-in** — `grid_lonlat` is
   built unconditionally, so the default synthetic/performance path changes too
   (from wrong to right; the §6 timings predate it).
10. **`cv_air` was `cp_air`.** `wp_d_contract.cpp` and `wp_d_contract_cuda.cpp`
    computed `cv_air = cfg.rdgas / cfg.akap`, which *is* `cp_air` (≈1004.7),
    where `dyn_core.F90:323` requires `cv_air = cp_air - rdgas` (≈717.6). Every
    post-loop heating `dtmp` (`:1349`) was a factor `1/(1-akap) = 1.4` **too
    small**. Fixed in both twins. The hydrostatic branch's
    `cp_air = cv_air + rdgas` was written against the correct definition and
    becomes right again automatically. **Affects the default path** (post-loop
    heating runs there).
11. **`cfg.n_con` hardcoded to 1; authoritative value is `npz` = 10.** `n_con` is
    **derived**, `dyn_core.F90:402-412`:
    `if (convert_ke .or. vtdm4 > 1.E-4) n_con = npz; else if (d2_bg_k1 < 1.E-3)
    n_con = 0; else if (d2_bg_k2 < 1.E-3) n_con = 1; else n_con = 2`. The
    in-source comment justifying the `1` actually justified **`nf_ke == 1`** (the
    `corner_edge_service` stub) and then set `n_con` — two independent controls
    conflated. `n_con` bounds how many *levels* of the `pt` update run
    (`:1338 do k=1,n_con`); `nf_ke` bounds how many *`del2_cubed` passes*
    `heat_source` receives first (`:1310-1311`), and only `nf_ke` touches
    `corner_edge_service`. Authoritative `vtdm4 = 0.01` confirmed twice
    independently (fixture `input.nml:23` / `fv3.j:386`, and
    `geos_production_flags()`'s `f.vtdm4 = 0.01` at
    `fv3_damping_translation.cpp:277` ← `FV_StateMod.F90:744`) ⇒ `n_con = npz`.
    This fix alone closed the exit `pkz` divergence: per-level max|diff|
    `5.33e-15, 6.68, 7.99, 8.43, 8.31, 7.74, 6.76, 5.36, 3.54, 1.26` (CUDA had
    written only `k=1`) → **1.528e-13 / 5.777e-15 ROUND-OFF**. **Affects the
    default path** — `geos_production_flags()` also sets `vtdm4 = 0.01`, so
    `n_con` derives to `npz` there too.
12. **`corner_edge_service` was a fail-stub, and `del2_cubed`'s dir=2
    `copy_corners` had no call site at all.** Not a value defect but a *missing
    authoritative operation*: `del2_cubed` rotates the corners twice per pass,
    dir=1 at `dyn_core.F90:2419` and dir=2 at `:2437`, and CUDA issued only the
    first — into a `corner_service_not_implemented` stub that returned a hard
    `FV3_GPU_UNSUPPORTED_CONFIGURATION`. Fixed by a thin
    `corner_edge_service_cuda` adapter onto WP-B's existing
    `launch_copy_corners_cuda` (a direct port of `tp_core.F90:257-318`, already
    exercised at both directions by validated transport paths) plus the missing
    `Axis::y` request in both twins. Key source fact: `copy_corners`'s `ng` is the
    compile-time PARAMETER (`tp_core.F90:52`) = 3, so the rotated block is 3×3 on
    every pass and the request's `halo_width` is the *write band*, not a block
    size. **Affects the default path.**
13. **`cfg.nf_ke` hardcoded to 1; authoritative value is `min(3, nord+1)` = 3.**
    `dyn_core.F90:1310`. This was the entire remaining exit `pt` residual —
    it **under-smoothed** `heat_source` before the `pt` update rather than
    skipping levels of it (that was #11). Evidence it was a smoothing deficit and
    not an amplitude error, gathered before the fix: non-constant per-cell ratio
    (median 1.68, std 44.6), per-level std ratio 3.7–4.9×, correlation 0.72–0.82.
    Now derived after every `cfg.nord` override so `nord=0` still yields 1; not
    hardcoded to 3. Fixing it required #12 first, and the pin's own in-source
    justification cited the stub it was perpetuating — a self-masking pair.
    Result: exit `pt` 6.397e-05 → **1.421085e-14 ROUND-OFF**, and `pkz`
    1.528e-13 → 1.421e-13. **Affects the default path.**

Localization method for #9, worth carrying forward: the corner *inputs* were
proven identical at round-off, then the three extrapolation ratios
`r = d1/(d2-d1)` were **solved for by least squares** from
`3*qout = Σ_t [(1+r_t)q1_t − r_t q2_t]` over 22 equations. Fortran returned
`[0.4212796036, 0.4212796036, 0.4212796036]` on every tile (geometrically
congruent, as a symmetric gnomonic panel corner requires); CUDA returned
`[0.093, −68.5, 1213.3]` on pe4 and `[0.050, −2.68, 1.72]` on pe0, both fitting
at round-off — proving the arithmetic form was correct and the geometry *data*
corrupt. A standalone probe against the harness's own `cube::CubeGrid` then
returned exactly `0.4212796036` on all six tiles, proving the geometry *source*
was authoritative and the corruption lay between harness and kernel.

**Numerical algorithm changed: NO** for all thirteen.

### Missing authoritative behaviour now IMPLEMENTED

- **The `omga` finalization path — entirely absent, now implemented.** Live
  branch is `use_old_omega = .true.` (`fv_arrays.F90:465`, not overridden in the
  fixture namelist), so the authoritative sequence is: **`pem` snapshot**
  (`dyn_core.F90:503-517`, `ptop` + cumulative `delp`, gated
  `it==n_split .and. end_step` — it is **inside** the acoustic loop, not
  pre-loop); then, strictly **after** the E11 block and still inside the loop,
  `:1205 if (last_step)` / `:1206 if (use_old_omega)` /
  `:1211 omga = (pe(i,k+1,j) − pem(i,k+1,j)) * rdt`; then
  `:1218 call adv_pe` (`:1535-1636`). The `.not. use_old_omega` forms at
  `:797-801`, `:838-842`, the `om2d` block at `:1219-1237` and the `id_ws`
  hydrostatic diagnostic at `:1238` are all **dead** in this configuration, and
  `NumericalConfig` carries no `use_old_omega` knob, so only the `.true.` path is
  implementable. Before this pass CUDA's last writer of `omga` was
  `Riem_Solver_c` using it as the `w3` argument — faithful as an *intermediate*
  (Fortran aliases the same array as `c_sw`'s `wc` at `:518`) but not the exit
  value, which is why the exit dump was bit-identical to the post-`Riem_Solver_c`
  dump on all six tiles.
  - Implemented as three host twins plus their device counterparts —
    `omega_pem_snapshot`, `omega_from_pe`, `adv_pe` — and two additive dispatch
    slots. `a2b_ord2` (`a2b_edge.F90:697-800`, live branch
    `grid_type<3 .and. .not. nested`) was **REUSED**, not rewritten: the existing
    `a2b_ord2_global` host/CUDA pair was verified verbatim against it.
  - **`ec1/ec2/en1/en2` are DERIVED, not invented.** They exist in Fortran
    solely for this computation (`fv_grid_utils.F90:759-772`, preceded by the
    comment `! For omega computation:`; `get_center_vect` at `:1792-1841`), are
    **unit** vectors (so `radius` cancels out of `adv_pe` except via the
    authoritative `dx`/`dy`), and are reconstructible from each cell's own four
    B-grid corners. The derivation's one premise — that the harness
    `grid_lonlat` corners *are* the Fortran `gridstruct%grid` corners at the same
    indices and orientation — was **proven radius-free** by
    `tools/savepoints/verify_adv_pe_geometry.cpp`: the ratio
    `dx_authoritative(i,j) / great_circle_dist(corner(i,j), corner(i+1,j), 1)` is
    one constant over `adv_pe`'s exact footprint on all six tiles (worst relative
    spread **6.661e-15**, implied radius exactly 6371000 m), which a store-order,
    index-base, orientation or tile-identity error would destroy. `:308-321`
    independently confirms `grid(:,:,1) = lon`, `grid(:,:,2) = lat`, i.e. the
    component-slowest layout that §4 #9 established.
  - Workspace: `OperatorArena` grown from 10 to 12 arrays (slot 10 = `pem`,
    slot 11 = `pb`), both device-resident. **No host/device transfer added**;
    persistent residency preserved.
  - Arithmetic and derived geometry were validated **independently of the GPU
    and of the runtime wiring** by `tools/savepoints/verify_omega_host.cpp`,
    which drives the real host twins entirely from authoritative Fortran
    savepoints: worst max_abs 1.158221e-14 / guarded max_rel 2.445659e-13. That
    the agreement is **non-vacuous** was measured, not assumed — the `adv_pe`
    increment is 1.619e-03…2.334e-03 against field ranges 3.238e-02…4.736e-02,
    i.e. ~5% of the field and ~1e11× the residual.
  - Recorded caveat, deliberately not chased: Fortran computes these grid
    vectors in quad precision (`f_p = selected_real_kind(20)`; no
    `NO_QUAD_PRECISION` in the fixture's `flags.make`), so the derived unit
    vectors may differ by ≤1 double ulp ⇒ ~1e-18 absolute on `omga`, three
    orders below the `pe − pem` cancellation floor.
  - **Numerical algorithm changed: NO.** `v3·grad` is invariant under global
    rotation, so global panel orientation is irrelevant; only intra-tile index
    orientation matters, and a transpose there would show as O(1) error.
- **E11 — `it==n_split` staggered seam reconciliation.** Authoritative call
  `dyn_core.F90:1178`
  (`mpp_get_boundary(u, v, domain, ebuffery=ebuffer, nbufferx=nbuffer,
  gridtype=DGRID_NE)`), write-back `:1181-1188`, guard
  `it==n_split .and. grid_type<4 .and. .not. nested`. **The `:1142`/`:1145-1152`
  line numbers previously recorded here were STALE.** It is a ring-**0**
  reconciliation of the one shared panel seam line that two tiles compute
  redundantly — not a halo exchange. Implemented as a new group
  `FV3_GPU_H_DWIND_SEAM` plus three minimum general extensions to the existing
  halo infrastructure: `HaloGroupSpec::ring_begin` (0 here, with
  `halo_width = 0`), `HaloFieldSpec::dest_edges` (north for `u`, east for `v`),
  and `halo_ring_count`/`halo_ring_ordinal` so the nine ring-1 groups are
  unaffected. The table entry uses a **cross-component source** (`u` ← `v`,
  `v` ← `u`), which is what `gridtype=DGRID_NE` means. Verified **bitwise**, not
  by tolerance — a tolerance test cannot see it, since `sp2c` `it=3` grades SMALL
  with or without it: `tools/savepoints/probe_e11_map.py` gives max|resid|
  **0.000e+00** for all 12 `(tile, field)` pairs against a genuine max|delta| of
  3.9e-14…3.9e-13. Footprint measured: at `it=3` only `u(is:ie,je+1,:)` and
  `v(ie+1,js:je,:)` change (0.0 off-line); at `it=2` **0 of 80** cells change.
  First observable at `sp2c` `it = n_split = 3`.
- **`heat_source`/`diss_est` per-substep accumulation.** Fortran accumulates
  inside the loop (`:858`) and applies after it (`del2_cubed` `:1311`, `pt`
  update `:1318-1350`); CUDA previously zeroed it every substep, so only the last
  substep survived. Implemented with a **persistent device accumulator**
  (`fv3_gpu_context.hpp: heat_accum`, new
  `runtime/fv3_gpu_heat_accumulator.{hpp,cu}`, pre-loop setup in
  `fv3_gpu_runtime.cpp`). No host/device transfer added; persistent residency
  preserved. Post-loop gate is `n_con /= 0 .and. d_con > 1.e-5` (`:1309`);
  `d_con = 1.0` here, so it is active. Consuming fields are `pt` **and** `pkz`
  (the NH branch rewrites `pkz` at every heated level, `:1348`, before
  incrementing `pt`, `:1349`).
  Lesson worth keeping: the accumulate kernel iterates **explicit inclusive index
  ranges in the view's own index space** with a hard bounds guard returning
  `FV3_GPU_INVALID_ARGUMENT`. The first draft derived ranges from allocation
  extents and silently ran one cell past the owned box on the staggered fields.
- `apply_diffusive_heating_nh` (`wp_d_operators.cpp:549-586`) was audited
  line-by-line against `:1338-1353` and **matches exactly** (the `delt` ladder,
  the `pkz` recomputation, the `dtmp` form, the
  `sign(min(delt,|dtmp|),dtmp)/pkz` increment). `n_con` and `cv_air` were the
  only levers.
- `delt_max` **PROVEN inert**, not assumed: measured max |dtmp| across all six
  tiles = **1.0305e-03** against the smallest authoritative bound
  `delt(k=1) = 0.1*|bdt*delt_max| = 0.720` even at the non-authoritative
  `delt_max = 0.002` — 698× headroom, so the limiter never engages.

### Verdicts overturned in earlier passes (do not rediscover as defects)

- `gz`/`pkc`'s "SIGNIFICANT ~0.96 / ~0.08" at SP1 `it=1` was **entirely the four
  cubed-sphere diagonal corner halo cells**, which have no donor tile and which
  both implementations compute from meaningless inputs.
- `uc`/`vc`'s "max_rel 1.7–2.0" was **cancellation noise** (max_abs 6.1e-15 /
  4.5e-14) where `rel = |F−C|/max(|F|,|C|)` fabricates ≈1. `p_grad_c` never
  needed a fix. Comparator corrected: denominator floored at the field's own
  dynamic range; corner cells split onto their own line. The corrected comparator
  was verified to **still** rank the pre-fix dumps SIGNIFICANT.
- The `nord`/ring-width and `clamp_nord_cap` hypothesis for the SP2 `u`/`v` frame
  is **DISPROVEN** at production `nord=2`; the cause was `edge_*` (#7).

## 5. Current Correctness / Stability Status

As of commit `dd7b45e` (3A.11T/U/V consolidated) and `6fc409a`, plus the
uncommitted Prompt 6B work:

- Authoritative production first-run config (`d4_bg=0.12`, `nord=2`, `npz=10`,
  globally cube-consistent IC): 8×8, 16×16, 32×32 all **PASS**, `delp` stays
  positive, decelerating growth, zero NaN/Inf, 3/3 independent 8×8 runs
  bit-identical.
- R8 CTest: **22/22 PASS** with all current Prompt 6B fixes built in (including
  E11, the heat accumulator, the `nf_ke`/corner-service work, and the `omga`
  finalization — Slurm job 58161075, whose build log confirms it recompiled
  `wp_d_operators.cpp.o` / `wp_d_cuda.cu.o` and relinked every test binary), with
  `test_dispatch_smoke.cpp`'s `cfg.nf_ke` at its pinned 1. R4 CTest: 22/22 PASS
  (last run before the 6B fixes). Default synthetic smoke: **PASS** (Slurm job
  58164201 — `RUN exit=0`, all six tiles finite on `u,v,delp,pt,w,delz`, `delp`
  positive over the owned box).
  Note on the E11 regression: adding a tenth halo resolve point made
  `fv3_dispatch_smoke` fail **by design** (`FV3_GPU_UNSUPPORTED_CONFIGURATION`
  naming `H_DWIND_SEAM`) — that harness spells out `kRuntimeResolvedGroups[]`
  precisely so a new resolve point cannot be silently ignored.
  `FV3_GPU_H_DWIND_SEAM` was added to the list as a weak-sense declaration (lets
  the integrated path execute, asserts nothing about seam values); Phase E's real
  six-tile exchange is where the seam semantics are exercised.
- compute-sanitizer memcheck PASS (0 errors), racecheck PASS (0 hazards),
  initcheck PASS except one previously-classified dead-read signature
  (`dgrid::copy_corners_kernel`, value-harmless in both Fortran and CUDA).
- The `(d4_bg=0.16, nord=2)` fv_arrays default pairing remains a **known
  foot-gun** (exceeds the del-2n stability limit) — not a production defect (GEOS
  overrides to 0.12), not a port defect (Fortran has the identical exposure).
  Runtime warns; never clips.
- The historical "two disagreeing gravitational constants" defect was repaired in
  `dd7b45e`. Only a stale comment still asserts otherwise — see §10.

## 6. Performance Baseline

**STATUS: RE-ESTABLISHED post-Gate-C2 (Prompt 7, 2026-09-03).** Full detail,
methodology, and the nsys profiling analysis:
`docs/agent_handoff/prompt_7_post_gate_c2_performance_baseline.md`. Git HEAD
for this baseline: `191157f` (matches `origin/cuda/gpu-bringup`).

Fortran R8 median `DYN_CORE` ms/call (6 MPI ranks, one per cubed-sphere face,
Intel/AVX2, Discover `mil` nodes) — **REUSED**, not remeasured this pass:
fixture git HEAD/config unchanged, and every Gate C2 fix is CUDA-side only.

| Grid  | ms/call |
|-------|---------|
| 8×8   | 12.730  |
| 16×16 | 16.704  |
| 32×32 | 30.261  |

CUDA R8 median ms/call (1 A100, `warpa013`, POST-GATE-C2):

| Grid  | resident | transfer-inclusive |
|-------|----------|---------------------|
| 8×8   | 29.918   | 30.799              |
| 16×16 | 28.799   | 28.767              |
| 32×32 | 26.009   | 28.075              |

Speedup (Fortran ÷ CUDA, resident): 8×8 **0.426×**, 16×16 **0.580×**, 32×32
**1.163×**. Change vs. the old (pre-Gate-C2) baseline, resident: 8×8 **+5.60%**,
16×16 **−3.00%**, 32×32 **−9.78%** (transfer-inclusive: +4.91%/−6.89%/−9.95%).
Despite four fidelity fixes (`grid_lonlat` store order, `cv_air`, `n_con`,
`nf_ke`) plus E11/the heat accumulator/the new corner service/the `omga`
finalization all adding real per-call GPU work to the default path, resident
cost did not regress at 16×16/32×32 and rose only 5.6% at 8×8 — consistent
with (not proof of) the launch-latency-bound regime confirmed by profiling
below. Old and new runs used different A100 nodes (`warpa003` vs `warpa013`),
a confound; no stronger causal claim is made.

**Not a normalized hardware-efficiency comparison** (6 CPU ranks vs 1 GPU); no
CUDA implementation-quality conclusion implied.

**One bounded nsys profiling pass (32×32, resident, job 58226917) confirms the
launch/orchestration-bound hypothesis still holds**: 17,730 kernel
launches/dyn_core call, `cudaLaunchKernel` = 90.6% of CUDA API time, no single
kernel exceeds 6.3% of kernel time, memory-op time ≈2.41 ms/call (91.3% D2D,
i.e. GPU-side halo traffic, not host transfer) against ≈74 ms/call of kernel
busy time. `ncu` was judged NOT JUSTIFIED (no kernel individually dominant).
Top 3 optimization candidates identified (not implemented): kernel
fusion/batching of the many small repetitive kernels; CUDA Graphs to amortize
per-launch issue latency; consolidation of the halo/corner-service kernel
family (~18% of kernel time). **No numerical code was changed.**

**Prompt 11 update (2026-09-04, jobs 58238397/58238413):** with
`--fuse-halo-edges` (pack/unpack halo-edge fusion, opt-in, default off; see
summary above and `prompt_11_next_launch_target.md`), 32×32/R8 resident
median falls from the Prompt-7-baseline-consistent 29.20 ms/call to
**23.28 ms/call (−20.28%)**, and whole-dyn_core-call kernel launches fall
17,730→16,074/call (−9.34%), kernel busy time 74.0→67.44 ms/call (−8.86%).
`cudaLaunchKernel` remains 90.0% of CUDA API time (vs 90.6% baseline) — the
launch-bound classification is unchanged in kind, only reduced in magnitude.
GPU-side memory-op time is unchanged (≈2.47 ms/call, 91.4% D2D). Combined with
Prompt 9's `--fuse-corner-fill`: 23.86 ms/call — very slightly slower than
`--fuse-halo-edges` alone, recorded but not chased further (still bitwise
identical). **No numerical code was changed.**

**Prompt 13 update (2026-09-04, job 58238910, node `warpa013`) — CUDA Graph
mode is now functional and measured.** All figures 32×32 / R8 / resident /
`--fuse-halo-edges`, same job, same node, same binary ±`--cuda-graph`,
`--warmup 20 --iters 100 --repeats 5`, median of 5:

| configuration | ms/call | per-repeat spread |
|---------------|---------|-------------------|
| graph OFF (reference) | 21.660791 | 0.149 |
| graph ON (`--cuda-graph`) | **20.546787** | 0.497 |

Whole-dyn_core-call improvement **−5.14%** (−1.114 ms), above both spreads.
The cross-session Prompt 12 reference (22.255549 ms/call) would read −7.7%,
but the same-node BEFORE above is the honest comparand and is what is
claimed. nsys (two short runs, before/after in the same job, normalized over
25 dyn_core calls × 6 tiles): `cudaLaunchKernel` 16,182.0 →
**11,067.1 per dyn_core call (−31.6%)**; per tile 2,697.0 → 1,844.5 — the
BEFORE figure reproduces Prompt 11's recorded 2,697 exactly, which validates
the normalization. `cudaLaunchKernel` share of CUDA API time 89.0% → 78.2%.
`cudaGraphLaunch` = 450 = 25 calls × 3 substeps × 6 tiles, *including* the 6
capture occurrences — direct proof the repaired capture arm launches.

**Material regime change, and the reason further launch-count work is not
recommended: the launch-latency model is now broken.** `wall ≈
launches_per_tile × median cudaLaunchKernel latency` fit Prompt 7 (2,955 ×
8.14 µs = 24.05 vs 26.01 measured, 92%) and Prompt 11 (2,697 × 7.54 µs =
20.34 vs 23.28, 87%). Here a **−31.6% launch reduction bought only −5.14%
wall clock.** Corroborating evidence from the same traces: host issue time per
tile per call fell 22.07 → 16.45 ms (−25%, nsys-inflated) while
`cudaStreamSynchronize` total time rose **0.44 → 135.5 ms** (Max 0.036 →
12.999 ms) — the host now *waits on the device* rather than the device waiting
on the host. Against a kernel-busy floor of 11.24 ms/tile/call versus 20.55
achieved, the case is now **GPU-EXECUTION-BOUND**. Any future estimate of a
launch-count optimization's value must be discounted accordingly; do not
assume proportionality.

## 7. Current Validation Boundary

- Gate C1 (integrated dyn_core GPU execution): **CLOSED** (milestone 3A.9).
- **Cross-process MPI halo transport (Prompt 15): CLOSED.** Six separate MPI
  processes, one tile per rank, produce a `dyn_core_exit` state **BITWISE
  IDENTICAL** to the Gate-C2-validated in-process path (78/78 files, at 1 and
  at 5 dyn_core calls), plus a six-rank provenance test at 16,692 checks / 0
  failures. Same configuration scope as Gate C2 and no wider.
- Prompt 6A (authoritative Fortran savepoints): **CLOSED** — see §8 history.
- Prompt 6B Task-J in-loop instrumentation: **CLOSED.** `sp2a`/`sp2b`/`sp2c`
  exist, are labelled with the real `it`, and the c8 reference was regenerated
  once: `experiments/c8_L10_savepoint_6c_run{1,2}/savepoints`, 1206 files each,
  `diff -r -q` clean ⇒ **BIT-IDENTICAL**.
- Fortran/CUDA runtime numerical validation (Prompt 6B): **SP0 plus all 15
  in-loop boundaries — `{SP1, SP2, sp2a, sp2b, sp2c}` × `{it=1, it=2, it=3}` —
  PASS.** Per-field tables in §4. Validated operators: `c_sw`, `update_dz_c`,
  `Riem_Solver_c`, `p_grad_c`, `d_sw`, `update_dz_d`,
  `Riem_Solver3`/`SIM1_solver`, `nh_p_grad`, `adv_pe`/`a2b_ord2`, and (bitwise,
  by donor-line map) the E11 seam reconciliation.
- **dyn_core exit boundary (`sp3`, `it=3`): COMPARED, 13 of 13 fields PASS.**
  E11, the `heat_source` accumulation, the authoritative `nf_ke = 3` smoothing
  path and the `omga` finalization are all IMPLEMENTED (§4). The exit `pt`
  VALIDATES at ROUND-OFF SCALE and the exit `omga` VALIDATES at SMALL NUMERICAL
  DIFFERENCE, both on all six tiles.
- **GATE C2: PASS — CLOSED.** All six criteria hold: SP0 remains authoritative
  and BITWISE; all 15 in-loop boundaries pass; all 13 exit fields pass on their
  authoritative defined boxes; `omga` is **non-vacuously** validated (the
  `adv_pe` increment is ~5% of the field, ~1e11× the residual); every remaining
  discrepancy is explained (cancellation floors, quad-precision grid vectors,
  the diagonal corner halo cells with no donor tile); **no scientific algorithm
  was changed to force agreement**; zero NaN/Inf; and six-tile GPU-side halo
  communication is live and real (no stubs).
- Scope of the Gate C2 claim: c8 / npz=10 / n_split=3 / k_split=1 / R8 / 6 MPI
  ranks = 6 tiles / production first-run numerics. **Not** yet claimed for R4,
  other grids, other configurations, or moist (`q_con ≠ 0`) runs.
- Two recorded coverage gaps that Gate C2 does **not** cover, both unchanged by
  this pass: `pe`'s single Fortran halo ring is NOT WRITTEN by CUDA, and
  `MOIST_CAPPA`/`USE_COND` are compiled in the Fortran but not in CUDA (proven
  round-off-neutral for this fixture only). See §10.
- Do not claim numerical equivalence from the performance baseline in §6, nor
  from the Prompt 6A savepoints alone (Fortran-only).
- **Cross-process MPI halo transport (Prompt 15): VALIDATED by equivalence, not
  by a new Fortran comparison.** The six-MPI-process `dyn_core_exit` is BITWISE
  IDENTICAL (78/78 files = 13 fields × 6 tiles) to the in-process
  `HaloPeerGroup` path at both 1 and 5 dyn_core calls, from the same binary and
  configuration on the same node. Since the in-process path is the one Gate C2
  validated against authoritative Fortran, the Gate C2 result carries over to
  the MPI transport **for this configuration and no wider**. This is bitwise
  equivalence to a validated reference — the strongest available form — but it
  is explicitly *not* an independent Fortran comparison of the MPI path, and
  the Gate C2 scope caveats above apply unchanged.

- **Standalone production integration (Prompt 17): the CUDA dyn_core backend
  RUNS INSIDE THE REAL `StandAlone_FV3_Dycore.x`** under the production
  decomposition (`FV_NX=1`/`FV_NY=6`, six MPI ranks, one tile per rank), and
  agrees with the Fortran backend from the same executable: **dyn_core ENTRY
  call 1 BITWISE IDENTICAL 6/6 ranks on all 13 state fields**, dyn_core EXIT
  **all 17 fields ROUND-OFF SCALE or SMALL NUMERICAL DIFFERENCE at calls
  1/2/3/6** (worst `max_rel` 2.008e-11, always `w`). This is a *short-run*
  integration result — 6 dyn_core calls, c8/npz=10/n_split=3/k_split=1/R8 — and
  is **not** an extension of Gate C2's scope to any other configuration.
  ~~It is gated for longer validation by the intermittent transport defect in
  §10.~~ **Prompt 18: that gate is LIFTED** — the transport defect was a
  test-harness CUDA stream-ordering hazard, is fixed, and the six-rank transport
  now passes 24/24 consecutive OpenMPI runs. The standalone short run was
  re-verified afterwards and reproduces every Prompt 17 figure to the digit
  (all 6 calls compared this time: entry call 1 BITWISE on 13 state fields 6/6
  ranks; exit 102 verdicts = 78 ROUND-OFF + 24 SMALL, worst `max_rel`
  2.034576e-11, zero SIGNIFICANT, zero NONFINITE, `delp` positive and identical
  on both arms). **Longer standalone validation is now unblocked** (§9 item 17).

## 8. Current Task

**Prompt 6B — CUDA vs. authoritative Fortran numerical comparison: COMPLETE.
GATE C2 CLOSED.** All three acoustic substeps validate at all five in-loop
boundaries and **all 13 dyn_core-exit fields** validate. There are no remaining
gaps in this configuration.

**Prompt 7 — post-Gate-C2 performance baseline + one bounded nsys profiling
pass: COMPLETE.** See §6 above and
`docs/agent_handoff/prompt_7_post_gate_c2_performance_baseline.md` for full
detail. No optimization was performed (explicitly out of scope for that pass).
No numerical code was changed.

**Prompt 8 — CUDA Graphs prototype: was BLOCKED, now SUPERSEDED by Prompt 13
(see below).** Its two findings are resolved/explained: (1) the numerical
divergence was **not** a device-pointer/parameter-binding problem — that
leading hypothesis was wrong; the capture arm simply never launched the
captured graph, so one `d_sw` substep was dropped (Prompt 13 §2); and (2) the
suspected resourcing confound is **confirmed** — its sbatch ran `mpirun -np 6
--oversubscribe` over a bench that already spawns 6 tile threads, i.e. 36
tile-contexts / 36 threads on one A100 with 12 CPUs, which its own trace
proves (`cudaStreamBeginCapture` = 36 = 6×6; `cudaGraphLaunch` = 2,664 = 74 ×
36). **Do not reuse `perf_rerun_logs/prompt8_graph.sbatch`'s resource
request**; use `perf_rerun_logs/prompt13_graph_repair.sbatch`, which invokes
the benchmark directly with no `mpirun`. Read
`prompt_8_cuda_graphs_prototype.md` only for historical context; its
conclusions about graph viability are retracted.

**Prompt 9 — narrow kernel-fusion prototype: COMPLETE.** See summary above and
`docs/agent_handoff/prompt_9_kernel_fusion_validation.md` for full detail.
Fused `c_sw`'s per-level `fill2_4corners`+`fill_4corners` corner-fill pair
(both directions) into one launch, opt-in via `--fuse-corner-fill` (default
off, default path unchanged). Correctness: bitwise-identical `dyn_core_exit`
dump on all 13 fields × 6 tiles, ctest 22/22 PASS. Performance: small
(−2.0% kernel launches/call, −1.4% kernel busy time/call) — this fusion only
touched 2 of 17,730 launch-generating call sites, so a small effect is
expected, not a disappointment. A raw wall-clock read suggested ~10% but was
**not validated** (comparable to job noise) — do not quote it. This does not
close the "kernel fusion" optimization candidate from Prompt 7 §6; it
validates the approach on one small pair. Higher-leverage unfused targets
identified: `copy_corners_kernel` (1,140 calls/dyn_core-call), the
halo/corner-service family generally (~18% of kernel time, per Prompt 7),
`a2b4_global_corner_kernel` (108 calls/call), `pack_edge_kernel` (1,056
calls/call). **Numerical algorithm changed: NO.**

**Prompt 10 — `copy_corners_kernel` batching characterization: COMPLETE, NO
IMPLEMENTATION.** Full detail:
`docs/agent_handoff/prompt_10_copy_corners_batching.md`. Reused Prompt 9's
existing nsys trace (no new profiling); read every one of the 11 production
`launch_copy_corners_cuda` call sites (10 in `wp_b_cuda.cu`, 1 in
`fv3_gpu_production_dispatch.cu`) and their callers. Reconstructed a
per-tile-per-dyn_core-call inventory (≈192, ×6 tiles ≈1,152, matching the
empirical 1,140 within ~1%). All four batching dimensions the task specified
were ruled out on direct evidence, not assumption:
- **(A) tile-batching** — the six tiles run as independent concurrent host
  threads (`run_small_perf_bench.cpp:3277-3303`), each with its own GPU
  context/stream; no shared launch surface exists. Redesign-level, not
  attempted.
- **(B) same-field/direction merge** and **(D) adjacent fusion** — every
  loop-based call site (`del6_vt_flux_cuda`, `deln_flux_pass`) has a real
  intervening consumer kernel between its `dir=1` and `dir=2` corner calls
  that reads the `dir=1` rotation before `dir=2` overwrites the same cells.
  Proven non-reorderable, not assumed.
- **(C) cross-field merge** — the four `scalar_transport` calls in `d_sw`
  (`delp`/`w`/`pt`/`vort`) looked independent but are a proven strict
  sequential pipeline (S7→S8→S10→S11→S12→S14, and separately S32→S33→S34);
  the three direct `del6_vt_flux_cuda` calls (`z2`/`wk`/`w`) live in three
  separate subsystems with no proven-safe reorder relationship.
One narrower, genuinely-safe lever was found and quantified, then rejected on
cost/benefit: merging the *outer* per-`nord` corner calls in
`del6_vt_flux_cuda`/`deln_flux_pass` across different `nord` values (they
touch disjoint vertical-level subsets of the same buffer, so the rotation
itself is nord-independent) — ceiling ≈180 of 17,730 total launches (≈1.0%),
below Prompt 9's own measured wall-clock noise floor, at the cost of a new
`LevelFilter` mode and restructuring two Gate-C2-validated damping operators
at 5 call sites. Per the task's own instructions ("do not force [a] target if
dependencies prove otherwise"; HARD STOP on "negligible launch reduction"),
**no code was written, no build was touched, and no Slurm job was run.**
**Numerical algorithm changed: N/A (no change made).** Higher-leverage unfused
targets carried forward unchanged from Prompt 9: `a2b4_global_corner_kernel`
(108 calls/call, 6.2% of kernel time), `pack_edge_kernel` (1,056 calls/call,
3.6%, higher call count than `copy_corners` and not yet characterized), and
the halo/corner-service family generally (~18% of kernel time).

**Prompt 11 — pack/unpack halo-edge fusion: COMPLETE, IMPLEMENTED, and
MEASURED with a real improvement.** Full detail:
`docs/agent_handoff/prompt_11_next_launch_target.md`. Characterized exactly 2
candidates as required: `pack_edge_kernel`/`unpack_edge_kernel` (~2,076
combined launches/call, 6.8% of kernel time — carried forward from Prompt
9/10) and `a2b4_global_corner_kernel` (108 launches/call, 6.2–6.6% of kernel
time). Selected the pack/unpack family — it is the only one of the two whose
launch-count ceiling (~1,548/call, ≈8.7% of the whole-call baseline) clears
the task's ≥3% GO/NO-GO threshold; `a2b4_global_corner_kernel`'s ceiling
(≤108/call, ≤0.61%) cannot, and its real fix is kernel-time/occupancy tuning,
explicitly deferred by project policy. Implemented `--fuse-halo-edges`
(opt-in, default off): new `pack_edge4_kernel`/`unpack_edge4_kernel` merge
the 4-edge loop (west/east/south/north) into one launch per `(halo group,
field)`, since each edge's send/recv slot is disjoint with **no
producer/consumer dependency between edges** — unlike Prompt 10's
`copy_corners` dir=1/dir=2 case, which had a real intervening consumer.
**Numerical algorithm changed: NO** — same index arithmetic, same values,
same buffer addresses. Validated `dyn_core_exit` **78/78 fields × tiles
BITWISE IDENTICAL** (both alone and combined with Prompt 9's
`--fuse-corner-fill`), R8 ctest PASS, both smokes PASS. Measured: whole-call
kernel launches 17,730→16,074/call (**−9.34%**), kernel busy time
74.0→67.44 ms/call (**−8.86%**), 32×32/R8 resident wall-clock 29.20→23.28
ms/call (**−20.28%**, reproducible, spread ≤0.33 ms across 5 repeats — well
above the noise floor that made Prompt 9's own wall-clock claim unusable).
One nsys profile was run (gated correctly: only after validation PASSED and
improvement cleared ≥3%). Outlook recorded as **DIMINISHING** for further
local launch-count fusion in the halo/corner-service family — the two
largest-ceiling remaining targets (`copy_corners_kernel`'s axes, this
family's edge axis) are both now characterized to completion (one ruled out,
one implemented); what remains in that family is individually below the ≥3%
threshold, so the next step should be kernel-time/occupancy tuning or
revisiting the blocked CUDA Graphs prototype (Prompt 8), both explicitly
deferred architectural work, not another local fusion pass.

**Prompt 12 — `a2b4_global_corner_kernel` compute tuning: COMPLETE, MODERATE
value (−2.15%).** See the leading summary, §9 item 7, and
`prompt_12_a2b4_kernel_tuning.md`. Bitwise identical (156/156). `ncu` was
platform-blocked; see §10.

**Prompt 13 — architecture-level optimization: COMPLETE, CUDA GRAPH REPAIR
SELECTED AND IMPLEMENTED (−5.14%).** Full detail:
`docs/agent_handoff/prompt_13_architectural_optimization.md`. Compared two
architecture-level directions from source before implementing anything: **(A)
repair CUDA Graph replay safety for a bounded capture region** versus **(B)
reduce host launch/orchestration overhead by explicit stage/tile batching
without graphs**. Selected A because it classified **LOCAL** — a missing
`cudaGraphLaunch`, not a graph-semantics or communication problem — whereas
B's only ≥5%-ceiling candidates were MODERATE-to-RUNTIME-REDESIGN.
- **§1 host-orchestration characterization** (compact diagram in the handoff,
  no kernel enumeration): 6 host threads in **one** process
  (`run_small_perf_bench.cpp:3297-3310`, `kTiles=6`), one GPU context per tile
  (`:2822`), all six sharing the device's CUDA **primary** context (only
  `cudaSetDevice` is called; `cudaStreamCreateWithFlags` Num Calls = 12 = 2
  streams × 6 tiles); host-side `n_split` acoustic loop
  (`fv3_gpu_runtime.cpp:1826-2071`) over a 13-slot `NumericalDispatch` table;
  10 halo resolve sites in-loop (7 fire per substep) plus `H_HEAT` after it; 6
  events, all `cudaEventDisableTiming`; **zero device syncs inside the loop on
  the default path** (only two host `pg->barrier()` calls ordering enqueue);
  one lazy at-most-once `heat_accum` `cudaMalloc` before the loop.
- **§2 root cause of the Prompt 8 failure — verified, not hypothesized.**
  Stream-captured work is **recorded, not executed**. The capture arm did
  BeginCapture → run → EndCapture → Instantiate with **no `cudaGraphLaunch`**,
  and the `} else` bound the fallback `run(dispatch.dgrid_dynamics)` to
  non-graph mode only, so one whole `d_sw` substep (transport,
  divergence/vorticity damping, KE/`heat_source` generation) was dropped on
  the first substep of the first `fv3_gpu_dyn_core_step` per context per
  repeat — matching Prompt 8's O(1) *uncorrelated* wind divergence exactly.
  Every alternative mechanism was affirmatively excluded (8-row table in the
  handoff): **graph-node parameter updates required = ZERO** (no per-substep
  scalar reaches `dgrid_dynamics`); device pointers are allocated once and
  arena bases re-derived byte-identically; halo data is LIVE inside the region
  (base frozen, data ordered in by `cudaStreamWaitEvent(compute_stream,
  event_halo_xfer)`); no CUDA API call inside the region except
  `cudaPeekAtLastError()`. The one by-value host scalar that *could* differ
  between calls is `DSwOptions::dt` (`= bdt/n_split`), handled by a
  capture-time invariance guard rather than node rewriting.
- **§6/§7 prototype and validation.** ~15 lines across 2 files
  (`runtime/fv3_gpu_runtime.cpp`, `include/fv3_gpu_context.hpp`); **capture
  region unchanged — `dgrid_dynamics` only; halo exchange is NOT captured.**
  Ladder: 1 call (the capture occurrence — precisely what Prompt 8 got wrong)
  **78/78 bitwise identical**; 5 calls (1 capture + 14 replays/tile)
  **78/78 bitwise identical**; 600-call / 1,800-substep / 1,770-replay stress
  clean — **no illegal memory access, no NONFINITE, no FAIL lines**. Prompt
  8's replay crash did not reproduce; most plausibly a consequence of the same
  defect (subsequent substeps on corrupted state) and/or the 36-context
  oversubscription — stated as inference, since it can no longer be reproduced
  to test. 60 `[FV3_GPU_GRAPH] capture+instantiate_ms=` lines (11.9–16.2 ms) =
  6 tiles × 10 graph-mode invocations, confirming exactly one capture per
  context per repeat and nothing re-capturing mid-run.
- **§8 performance:** 21.660791 → 20.546787 ms/call (**−5.14%**), launches
  16,182 → 11,067/call (**−31.6%**) — see §6 above for the full tables and for
  the regime-change finding that retires Direction B.
- **§9/§10 cost control:** one architecture choice, one prototype, 32×32/R8
  only, **zero of the three permitted fix cycles consumed** (the prototype
  worked on the first attempt), no Fortran rerun, no R4, no new savepoints, no
  individual-kernel tuning, `ncu` not run. Ended on hard-stop condition "the
  prototype succeeds and is measured".
**Numerical algorithm changed: NO.** Graph mode remains opt-in
(`--cuda-graph`, default off), so the default path is provably unchanged.

**Prompt 14 — standalone CUDA production integration: BLOCKED/ESCALATED, then
UNBLOCKED by Prompt 15.** Its §2 production `dyn_core` call-boundary interface
table (`fv_dynamics.F90:645-655` call site, `dyn_core.F90:166-263` signature,
field by field) is complete and reusable as-is. Its blocker — no cross-process
halo transport — is closed; see Prompt 15. Its secondary finding still stands:
`NumericalConfig` installation and real `TopologyView`-based topology
installation are still only reachable through internal C++
(`install_config`/`set_topology`), so a public ABI extension for those is still
needed. Detail: `prompt_14_standalone_cuda_integration.md`.

**Prompt 15 — cross-process MPI halo transport: COMPLETE. CLOSES the Prompt 14
blocker.** Full detail: `docs/agent_handoff/prompt_15_mpi_halo_transport.md`;
summary in the lead paragraph at the top of this file. In brief: a
`HaloTransport` interface with two backends (`PeerHaloTransport` in-process,
default; `MpiHaloTransport` host-staged real MPI, six processes, one tile per
rank), sharing all topology/rotation/sign/pack/unpack logic so the MPI backend
duplicates **no** edge-rotation formula; a public C ABI for transport selection
with **no C++/MPI type** (communicator crosses as the Fortran INTEGER handle);
and `HaloPeerGroup` functionality fully preserved. Evidence: six-rank
provenance test **16,692 checks / 0 failures** covering scalar, both wind
staggerings (DGRID_NE *and* the transposed CGRID_NE), the 4-field width-3
multi-ring group, and E11; six-MPI-process `dyn_core_exit` **78/78 BITWISE
IDENTICAL** to the in-process path at 1 and 5 calls; R8 ctest **22/22 MPI-OFF /
23/23 MPI-ON**. **Numerical algorithm changed: NO.** Zero of 3 permitted
transport fixes consumed. Method notes worth carrying forward:
- **Put the transport seam where the ordering already is.** Reading the proven
  exchange ordering first (`pack → pack_done → [transport] → unpack →
  xfer_done → consumer → [release]`) showed that exactly two steps are
  transport-dependent, so the interface has exactly two methods and no logic
  could be duplicated even by accident. Designing the interface from the
  ordering rather than from the MPI call list is what kept the rotation
  formulas in one place.
- **Verify the mapping instead of trusting it.** The rank↔tile identity was
  confirmed from source *and* then checked at runtime by allgathering the tile
  map and cross-checking every neighbour's `(rank, tile)` pair. A mis-wired
  launcher now fails at attach rather than exchanging plausible-looking wrong
  data.
- **Test the transposed pair, not just one wind group.** `H_DWIND_*` and
  `H_CWIND` carry the same two staggerings in *opposite* argument slots; a
  staggering-keyed sign would pass one and fail the other. Both were exercised
  deliberately for that reason.
- **Prove equivalence against the already-validated path, not against a
  loosened tolerance.** Running both transports from the same binary, same
  flags, same node, same job — differing only in `--mpi-halo` — made BITWISE
  the achievable bar and made any difference attributable to the transport
  alone. No new Fortran run was needed.
- **Collective-safety hazards are design problems, not debugging problems.** A
  lone SKIP-77, a mid-run failure, and a premature detach would each hang or
  corrupt the other five ranks; all three were handled up front
  (`MPI_Allreduce(MPI_MIN)` on device usability, `MPI_Abort` instead of
  `return false`, teardown before the timing reduction) rather than diagnosed
  from a hung job.

**Earlier pass — `omga` exit finalization.** Reused the existing 6C
savepoints (no new fixture run, no new Fortran instrumentation). Scope was the
exit `omga` only. **One fix used**, of the three permitted: the missing omega
finalization path, implemented as `omega_pem_snapshot` + `omega_from_pe` +
`adv_pe` on both host and device, with `a2b_ord2` REUSED and
`ec1/ec2/en1/en2` DERIVED-and-VERIFIED from existing authoritative geometry
(detail in §4 under "Missing authoritative behaviour now IMPLEMENTED").
**Numerical algorithm changed: NO.** R8 ctest 22/22 PASS; default synthetic smoke
PASS; all 15 in-loop boundaries and the other 12 exit fields re-verified from the
same dump with no regression (the exit `pt` is bit-unchanged). Result: exit
`omga` from max_abs ≈4.7e-02 / max_rel ≈1.0 to **4.875354e-14 / 1.505773e-12
SMALL NUMERICAL DIFFERENCE** on all six tiles, a ~1e12× reduction. The pass ended
on **HARD STOP condition (7), "`omga` passes and Gate C2 can close"**. Detail:
`prompt_6b_cuda_fortran_validation.md` §22.

Method notes worth carrying forward from this pass:
- **Validate arithmetic before, and separately from, the GPU.**
  `tools/savepoints/verify_omega_host.cpp` drove the *real* host twins from
  authoritative savepoint data on the login node, with no GPU and no queue wait,
  and reached 1.158e-14 — so when the GPU dump later reached 4.9e-14 the
  remaining question was only about the device twin and the wiring, not about the
  formulas. This was built *because* `gpu_a100` was ~6 h deep in queue; it turned
  the wait into the decisive result.
- **Prove a derived quantity's premise with a quantity you already trust.**
  `en/ec` could not be exported without new Fortran instrumentation (which would
  have fired HARD STOP (2)), so instead the premise — harness corners ≡ Fortran
  `gridstruct%grid` corners — was tested against the *authoritative `dx`/`dy`
  export*, radius-free, as a constant-ratio test over `adv_pe`'s exact footprint.
- **Measure the increment you are claiming to validate.** The `adv_pe`
  contribution was explicitly differenced out (`omga_pre`) so that agreement on
  `omga` could be shown to be evidence about `adv_pe` rather than about the
  dominant `(pe−pem)*rdt` term alone.
- **Explain a small residual instead of just reporting it.** `omga`'s floor is
  the `pe − pem` cancellation floor (~1.2e-14 at `pe ≈ 1e5` Pa, `dt = 1200` s),
  which is why it is graded SMALL and deliberately not claimed as ROUND-OFF, and
  why `mismatch = 640/640` is expected rather than alarming.

**Previous pass — `pt` / `nf_ke` exit reconciliation.** Reused the existing 6C
savepoints (no new fixture run, no new Fortran instrumentation). Scope was the
exit `pt` only; `omga` was explicitly excluded. **Three fixes used**, all inside
the assigned categories: (1) a real **`corner_edge_service_cuda`**, a thin
adapter onto WP-B's existing, already-exercised `launch_copy_corners_cuda` rather
than a new topology subsystem, replacing `corner_service_not_implemented`;
(2) the **missing dir=2 `copy_corners` rotation** at `dyn_core.F90:2437`, absent
from both the CUDA and host twins, which required splitting the host
`del2_pass` into three phases so the interleave is expressible; (3) **`nf_ke`
derived** as `min(3, nord+1)` (`:1310`) instead of pinned to 1. **Numerical
algorithm changed: NO** for all three. R8 ctest 22/22 PASS; default synthetic
smoke PASS; all 15 in-loop boundaries re-verified with no regression.

The pass ended on **HARD STOP condition (7), "exit `pt` passes"**: `pt` went from
6.397e-05 / 2.592e-06 SIGNIFICANT to 1.421085e-14 / 5.756846e-16 ROUND-OFF SCALE
on all six tiles, a ~4.5e9× reduction that confirms the previous pass's
under-smoothing diagnosis by construction. `pkz` moved to ROUND-OFF with it.
Detail: `prompt_6b_cuda_fortran_validation.md` §21.

The whole defect had been **self-masking**: with `nf_ke` pinned to 1 there is one
pass, `nt = 0`, and *both* corner rotations are skipped — so neither the
fail-stub nor the missing `:2437` call site could ever be reached, and the pin's
own in-source justification cited the stub it was perpetuating.

Localization notes worth carrying forward:
- The exit `pt` residual was **not** an amplitude error. The CUDA/Fortran
  per-cell ratio is not constant (median 1.68, std 44.6), while per level CUDA's
  increment std is 3.7–4.9× Fortran's with correlation only 0.72–0.82 — that is
  an *under-smoothing* signature, which pointed straight at the `del2_cubed`
  pass count rather than at any coefficient.
- `pkz` is computed from the **pre-increment** `pt`/`delp`/`delz`, so its
  round-off agreement at every level proves the state *entering* the post-loop
  tail agrees, isolating the residual to `heat_source` itself.
- `omga`'s provenance was **proven, not inferred**: the exit dump is
  **bit-identical** to the post-`Riem_Solver_c` dump on all six tiles
  (max|d| = 0.000e+00).

Three design choices in the instrumentation are worth carrying forward:
- `sp2a` captures `ws`, which is `intent(IN)` to `Riem_Solver3`, so one savepoint
  discriminates `update_dz_d` from `Riem_Solver3` without a fourth capture.
- `sp2b`'s `pkc`/`pk3`/`gz` are the **B-grid (corner) transforms**, because
  `nh_p_grad` calls `a2b_ord4(..., replace=.true.)` on all three. That is what
  made the decisive comparison possible: the exact arrays the pressure gradient
  differenced are directly observable on both sides.
- `sp2c`'s `u`/`v` halos are in flight (`i_pack(8)` started, not completed), so
  only the owned box is a defined comparison target there.

Settled in the previous pass and still true: **`heat_source`/`diss_est` is INERT
for every in-loop savepoint** — Fortran accumulates inside the loop
(`dyn_core.F90:858`) but does not read the result until *after* it
(`del2_cubed` at `:1311`, `pt` update `:1318-1350`). It could never have been the
SP1 `it=2` cause, on source evidence alone. The accumulation is now IMPLEMENTED
(§4), and its effect is confined to the exit `pt`/`pkz` exactly as predicted.

Also settled: the CUDA `pk3` dump differs from Fortran at `k=1` only, and that is
a **dump artifact, not a defect**. `set_top_kernel` (`wp_d_cuda.cu:455-458`)
writes `ptk` into the **arena scratch** the pressure gradient consumes, not back
into the registered `views.pk3`. Confirmed inert empirically: after the layout
fix `u`/`v` PASS while the `pk3` `k=1` dump difference is bit-unchanged.

Config-knob matching and field-layout/registration checks are otherwise clean.
Closed blockers, in order of discovery: fabricated grid metrics; missing SP0
loader; `bdt`; the missing `it==1` pre-`c_sw` scalar halo (the former "E5" item —
a **real CUDA defect**, one `resolve(FV3_GPU_H_DSCALAR)` line, root cause of the
earlier SP1 NONFINITE failure); `ptop`/`phis`/`zs`; `dp_ref`; `rdgas`;
`n_sponge`; `edge_*`; the Exner-kappa split; the `grid_lonlat` store order; E11;
the `heat_source` accumulation; `cv_air`; and `n_con`.

Full detail: `docs/agent_handoff/prompt_6b_cuda_fortran_validation.md` §19
(previous pass) and **§20** (this pass).

<details>
<summary>Prompt 6A history (closed)</summary>

Prompt 6A closed cleanly: authoritative Fortran `dyn_core` savepoints (SP0
entry, SP1 post-C-grid-predictor, SP2 post-D-grid/d_sw, SP3 exit; c8/npz=10/
n_split=3/R8/6 MPI ranks; production first-run numerics) were generated from
two independent runs of the real `GEOSfvdycore_fortran_perf` fixture driver,
classified **BIT-IDENTICAL** across all 462 captured fields, and passed all
finite/positivity sanity checks. Full detail, one documented driver-level
deviation (`test_case` 5→14; see that doc §3.1), and one instrumentation-only
bug found/fixed (damping-table write-format mismatch) are in
`docs/agent_handoff/prompt_6a_fortran_savepoints.md`. Reusable tooling:
`tools/savepoints/{manifest_builder.py,savepoint_reader.py}`; instrumentation
diff saved at `tools/savepoints/fortran_savepoint_instrumentation.patch`
(uncommitted in the fixture's nested `@fvdycore` working tree — not this
repo). Raw savepoint data lives only on the shared filesystem (not
committed): `GEOSfvdycore_fortran_perf/experiments/c8_L10_savepoint_6{a,c}_run{1,2}/savepoints/`.

</details>

## 9. Immediate Next Steps

1. ~~SP1 `it=1` divergences~~, ~~SP2 `it=1` `u`/`v` frame~~, ~~SP3 `it=1`
   observability~~, ~~SP1 `it=2` divergence~~, ~~the `it=2` and `it=3` chains~~,
   ~~E11~~, ~~the `heat_source` accumulation~~, ~~the exit `pkz`/`pe`~~,
   ~~the exit `pt` / `nf_ke`~~, ~~the exit `omga`~~, ~~Gate C2~~ — all **DONE**.
   See §4 and §7.
2. ~~Re-establish the CUDA performance baseline after Gate C2, then profile
   the small-grid launch-bound behavior with nsys before choosing optimization
   targets~~ — **DONE (Prompt 7)**. See §6 and
   `prompt_7_post_gate_c2_performance_baseline.md`.
3. ~~Prototype CUDA Graphs capture for one acoustic substep (32×32 case)~~ —
   **DONE, BLOCKED (Prompt 8)**. See summary above and
   `prompt_8_cuda_graphs_prototype.md`. Do not retry without addressing its
   two findings (numerical divergence on a single graph-mode call; replay
   instability under sustained repeats).
4. ~~Prototype kernel fusion for one small repetitive-kernel pair, verify
   bitwise, then measure~~ — **DONE (Prompt 9)**. See summary above and
   `prompt_9_kernel_fusion_validation.md`. Fusing `c_sw`'s corner-fill pair
   validated the approach (bitwise-identical, ctest clean) but only moved the
   launch count 17,730→17,370/call (−2.0%) — too small a target to call the
   fusion/batching optimization candidate closed.
5. ~~Characterize `copy_corners_kernel` (1,140 calls/dyn_core-call) as a
   batching target~~ — **DONE, NO IMPLEMENTATION WARRANTED (Prompt 10)**. See
   summary above and `prompt_10_copy_corners_batching.md`. All four batching
   dimensions were ruled out by direct source reading (tile-batching needs a
   runtime redesign; every same-buffer `dir=1`/`dir=2` pair has a proven
   intervening consumer; the apparent cross-field candidates are a proven
   sequential pipeline). The one safe partial lever found tops out at ≈1.0%
   of total launches — below Prompt 9's measured noise floor.
6. ~~Characterize `pack_edge_kernel`/`unpack_edge_kernel` and
   `a2b4_global_corner_kernel` as launch-fusion targets, rank them, implement
   the winner if it clears a 3% whole-call threshold~~ — **DONE, IMPLEMENTED
   AND MEASURED (Prompt 11)**. See summary above and
   `prompt_11_next_launch_target.md`. Pack/unpack halo-edge fusion
   (`--fuse-halo-edges`) selected over `a2b4_global_corner_kernel` (whose cost
   is kernel-time-bound at only 108 calls/call, below threshold). Measured
   17,730→16,074 launches/call (−9.34%), 74.0→67.44 ms/call kernel time
   (−8.86%), and a reproducible 32×32/R8 resident wall-clock improvement of
   −20.28% (29.20→23.28 ms/call median of 5 repeats). Bitwise identical to
   default on all 13 exit fields × 6 tiles, alone and combined with Prompt 9's
   `--fuse-corner-fill`. This closes out option (b) from the prior version of
   this item.
7. ~~Local launch-count fusion of the halo/corner-service family had
   diminishing headroom; characterize `a2b4_global_corner_kernel` as a
   compute/kernel-time problem instead of a launch-count problem~~ — **DONE,
   MODERATE VALUE (Prompt 12)**. See summary above and
   `prompt_12_a2b4_kernel_tuning.md`. Restructured the launch from 1 block/128
   threads (10 active, 1 SM used, up to 4 corners serialized per thread) to
   `dim3(4, levels, 1)`/1 thread — same formula, same summation order,
   bitwise-identical (156/156). Measured **−2.15%** whole-dyn_core-call
   (22.743409→22.255549 ms/call, 32×32/R8/resident/`--fuse-halo-edges`) —
   MODERATE, below the 3% HIGH-VALUE bar. **ncu profiling is platform-blocked
   on this Discover allocation** (`ERR_NVGPUCTRPERM` — GPU performance-counter
   access denied to this account), so the bottleneck classification
   (SERIAL/LOW-PARALLELISM primary, OCCUPANCY-BOUND secondary) is
   source-reading inference, not ncu-measured. See §10 for this as a new
   recorded platform constraint.
8. ~~Two paths forward: (a) request `ncu` counter access for further
   kernel-time tuning; or (b) investigate the CUDA Graphs numerical/replay
   defect in `dgrid_dynamics`~~ — **(b) DONE (Prompt 13): the graph defect was
   LOCAL, is repaired, and graph mode now measures −5.14%.** See the leading
   summary, §6, §8 and `prompt_13_architectural_optimization.md`. Its leading
   hypothesis (device-pointer/parameter bindings across the halo boundary) was
   **wrong**; the capture arm never launched the captured graph. Path (a)
   remains open and unattempted (`ncu` is still platform-blocked, §10).
9. ~~The recommended next milestone is STANDALONE CUDA INTEGRATION, not
   further local performance work~~ — **ATTEMPTED (Prompt 14), BLOCKED,
   ESCALATED to Opus; then UNBLOCKED by Prompt 15 (item 14) and now resumable
   at item 15 — RESUME THERE, not here.**
   Rationale, all measured
   in Prompt 13 (§6): the workload is now **GPU-execution-bound**, not
   host-launch-bound — a −31.6% launch reduction bought only −5.14% wall
   clock, and `cudaStreamSynchronize` total time rose 0.44 → 135.5 ms. The
   launch-latency model that made Prompts 7/9/11/12 straightforward to
   estimate no longer predicts benefit, so the remaining launch-side lever
   (Direction B, explicit stage/tile batching; the largest candidate is
   collapsing `c_sw`'s per-level launches, −5,670/call = −35.3%, numerical
   risk NONE by proven level independence) is now worth only ~5% for a
   MODERATE-scope change across ~31 signatures — a materially worse
   value/risk ratio than it looked pre-Prompt-13. Genuine further speedup
   would need kernel-time/occupancy work on the individually expensive
   kernels (`riem_solver3_kernel`/`riem_solver_c_kernel`, ~144–148 μs avg
   each), which is **gated on `ncu` counter access** (§10) and is explicitly
   deferred by `project_architecture.md` ("Performance Work Comes Later").
   Confirm scope with the user before starting any further optimization pass.
   If resuming performance work, the current same-node reference is Prompt
   13's: **21.660791 ms/call** (32×32/R8/resident/`--fuse-halo-edges`, graph
   off) and **20.546787 ms/call** (same + `--cuda-graph`); reuse
   `perf_rerun_logs/prompt13_graph_repair.sbatch`'s resourcing, never Prompt
   8's.
10. Before extending the Gate C2 claim beyond this configuration, close the
   recorded gaps in §10 in this order: R4 validation (currently only compiled and
   ctest-clean, never savepoint-compared); `MOIST_CAPPA`/`USE_COND` for any moist
   configuration; the four `a_imp <= 0.999` inert-but-non-authoritative knobs
   (`scale_m`, `m_split`, `delt_max`); `pe`'s unwritten halo ring; the
   un-localized `fv3_dispatch_smoke` Phase C failure at `nf_ke > 1`.
11. ~~Consider consolidating the ten-plus passes of uncommitted Prompt 6B/7/8/9
    work into a commit~~ — **DONE**, by the user, in `b456897` (not by an agent
    pass; see §1). Not pushed. Push/merge only if explicitly requested.
12. ~~Consolidate the uncommitted Prompt 11 pack/unpack halo-edge fusion work
    into a commit~~ — **DONE**, by the user, in `7dda37b` (not by an agent
    pass; see §1). Not pushed.
13. ~~Consolidate the uncommitted Prompt 12/13 work~~ — **DONE**, by the user,
    in `eb6d689` ("Optimize CUDA dyn_core launch orchestration"; not by an
    agent pass — see §1). Not pushed. What is now uncommitted is the **Prompt
    15 MPI halo transport** (8 modified files + 4 new source files + this file
    + `prompt_15_mpi_halo_transport.md`), plus the still-untracked Prompt
    10/11/12/13/14 handoff docs and `perf_rerun_logs/*.sbatch` scripts —
    pending explicit user request; not committed or pushed (see §1). **Do not
    commit** the profiler artifacts (`nsys_p13_*.nsys-rep`, `*.sqlite`), the
    build directories (`build_cuda_r8_mpi/`, `build_cuda_r8_mpibench/`),
    `scratch_p15/`, or any `*.out` job log.
14. ~~Prompt 14 blocked on the absence of a cross-process halo transport~~ —
    **UNBLOCKED (Prompt 15): the transport now exists and is validated.** A
    `HaloTransport` interface with a host-staged real-MPI backend (six
    processes, one tile per rank) produces a `dyn_core_exit` state BITWISE
    IDENTICAL to the Gate-C2-validated in-process path, with the in-process
    `HaloPeerGroup` fully preserved and the MPI path opt-in. See the Prompt 15
    lead paragraph, §3, §4, §7, §8, and
    `docs/agent_handoff/prompt_15_mpi_halo_transport.md`. Direction chosen:
    **host-staged MPI** — `project_architecture.md`'s "Stage 1". Stage 2
    (CUDA-aware MPI) and Stage 3 (comm/compute overlap) are new backends behind
    the same interface, not a redesign, and remain deferred.
15. ~~RESUME HERE — standalone CUDA production integration, second
    attempt~~ — **STARTED (Prompt 16), PARTIAL: item (1) below is DONE; items
    (2)-(4) are BLOCKED — see item 16.** Prompt 14's blocker is gone, so the
    work it deferred was the next step, in this order:
    1. **Extend the public C ABI for per-run setup.** `NumericalConfig`
       installation and real `TopologyView`-based topology installation are
       still only reachable through internal, non-ABI C++
       (`install_config`/`set_topology` in the benchmark and
       `fv3_gpu_internal.hpp`). Transport attachment is *already* public
       (`fv3_gpu_attach_mpi_halo_transport` /
       `fv3_gpu_detach_halo_transport` / `fv3_gpu_halo_transport_kind_of`,
       `include/fv3_gpu_communication.h`) — model the new entry points on
       those, and keep the same rule: no C++ or MPI type in the ABI.
    2. **Write the Fortran `ISO_C_BINDING` wrapper and branch the real call
       site** at `fv_dynamics.F90:645-655` (inside `do n_map=1,k_split`),
       keeping the Fortran backend the default. The field-by-field interface
       table in `prompt_14_standalone_cuda_integration.md` §2 is complete and
       reusable as-is — do not re-derive it.
    3. **Pass the GEOS/FV3 communicator through as the Fortran INTEGER
       handle** (`MPI_Comm_c2f` on the caller side); this is already what the
       transport expects, and is why no MPI type appears in the ABI.
    4. Run the real `StandAlone_FV3_Dycore.x` under `FV_NX=1`/`FV_NY=6`, six
       ranks, and compare against the Fortran backend.
    Note on what is *not* yet proven: the standalone Fortran backend was not
    re-verified in Prompt 14 or 15 (no build of it was run in either pass), and
    ABI/compiler-family compatibility with an Intel production GEOS build
    remains an open question flagged in `discover_platform.md` — the CUDA side
    is GCC-based.
    - Sub-item (1), the public C ABI, is now **DONE (Prompt 16)** — see the
      Prompt 16 lead paragraph and
      `prompt_14_standalone_cuda_integration.md` §7.1.
16. ~~Resolve the `i_pack`/FMS group-halo-update ownership question before
    writing any more of the Fortran wrapper~~ — **DONE (Prompt 17): ownership
    RESOLVED from FMS source, Design A adopted, and the standalone CUDA
    backend now runs and compares.** `mpp_start_group_update` packs/posts and
    increments `num_nonblock_group_update` immediately and only
    `mpp_complete_group_update` unwinds it, so abandoning a started handle
    leaks a slot and MPI requests per `k_split` — the CUDA arm therefore does
    not *issue* `i_pack(1)`/`i_pack(8)` (both DUPLICATE of CUDA's own
    `H_DSCALAR`/`H_DWIND_IN` before the same consumers) while `i_pack(10)`
    (tracer, DISTINCT REQUIRED) is still issued. See the Prompt 17 lead
    paragraph and `prompt_14_standalone_cuda_integration.md` §8.
    **Original text retained below for the source-evidence trail.**
16b. ~~Fix the intermittent `fv3_halo_mpi_transport` failure before any longer
    standalone validation run~~ — **DONE (Prompt 18): root-caused, fixed, and
    the transport is DETERMINISTIC at 24/24 consecutive OpenMPI runs (16692
    checks / 0 failures each), Intel MPI 5/5 PASS, R8 CTest 23/23.** The defect
    was **not** in the transport: the test harness issued its own `0xFF` poison
    `cudaMemset` and its field-array H2D on the **legacy** stream, unordered
    against the `cudaStreamNonBlocking` exchange stream. Not an MPI contract
    violation, not an OpenMPI bug. Three prior premises were measured wrong —
    the Intel-MPI-vs-OpenMPI comparison was inverted, the failing value was
    poison rather than 0.0, and the blast radius is a whole 105-element recv
    slot rather than one cell. One production hardening applied
    (`fv3_gpu_runtime.cpp:1290-1291` attach-time clears → `cudaMemsetAsync` on
    `d.transfer_stream`). The standalone short run was re-verified unchanged.
    See the Prompt 18 lead paragraph and
    `prompt_18_mpi_transport_determinism.md`.
17. ~~RESUME HERE — continue standalone validation: a longer multi-`k_split` /
    multi-day run~~ — **DONE (Prompt 19): PASS.** Stage A (72 dyn_core calls,
    k_split=1) and Stage B (12 dyn_core calls, k_split=2, the smallest
    unambiguous multi-`k_split` case) both PASS with zero SIGNIFICANT
    differences, no jump at the k_split/remap boundary, and `delp` positive on
    both backends. See the Prompt 19 lead paragraph and
    `docs/agent_handoff/prompt_19_longer_standalone_validation.md`. The
    reproduction recipe
    (`p17_both.sbatch`, `p17_rebuild_exe.sh`, `p17_rebuild_all.sh`,
    `p17_regress.sbatch`, `compare_dcx_backends.py --stage {dce,dcx}`) is in
    `prompt_14_standalone_cuda_integration.md` §8.9; the Prompt 19
    `p19_stageA_*`/`p19_stageB_*` experiment directories under
    `GEOSfvdycore_fortran_perf/experiments/` are a ready-made template for
    varying grid/precision similarly. Note the still-open
    items: ABI/compiler-family compatibility with an Intel *production* GEOS
    build is untested (this fixture links GCC-built CUDA archives into an
    Intel-Fortran executable successfully, which is evidence but not proof for
    full GEOS), and the four `a_imp <= 0.999` knobs plus
    `MOIST_CAPPA`/`USE_COND` remain as recorded in §10.
18. ~~R4 savepoint comparison against authoritative Fortran~~ — **DONE
    (Prompt 20): PASS**, at the same short-run envelope as Prompt 17
    (c8/npz=10/n_split=3/k_split=1/6 calls). See the Prompt 20 lead paragraph
    and `docs/agent_handoff/prompt_20_r4_numerical_validation.md`.
19. ~~Extend R4 to Prompt 19's longer/multi-`k_split` ladder (Stage A 72
    calls / k_split=1, Stage B 12 calls / k_split=2), watching the `w` and
    `omga` fields' growth trend~~ — **DONE (Prompt 21): PASS, both stages.**
    See the Prompt 21 lead paragraph and
    `docs/agent_handoff/prompt_21_longer_r4_validation.md`. Stage A: 72
    calls, zero NaN/Inf, `delp` positive throughout, `delp`/`pt`/`delz`/`pe`
    comfortably ROUND-OFF/SMALL for the full run; `w` bounded/oscillatory
    (not growing); `u`/`v` small episodic crossings peaking at call 48; the
    one genuine finding is `omga`'s *bulk*-metric SUPERLINEAR growth (0.06%
    → 4.06% of its own RMS over 72 calls, crossing the R4-SIGNIFICANT
    max_rel line around call 19) — explained as the field's own magnitude
    decaying while the absolute CUDA/Fortran discrepancy plateaus after call
    ≈43, not as unbounded divergence, but flagged to re-check if R4 is ever
    run longer than 72 calls. Stage B: 12 calls, same fields all bounded; the
    decisive remap-boundary check (`dcx` exit at call 2 vs `dce` entry at
    call 3, straddling the first k_split=2 remap) shows no discontinuous
    jump for either `w` or `omga` — post-remap entry is the same order of
    magnitude as (for `omga`, smaller than) pre-remap exit. **RESUME HERE —
    next validation envelope is a wider grid (for both R4 and R8), per the
    already-queued step in item 10/§10 below.** Reuse
    `p21_stageA_r4_both.sbatch`/`p21_stageB_r4_both.sbatch`'s pattern; if R4
    is extended past 72 calls at any point, re-check `omga`'s bulk-ratio
    trend for acceleration or a plateau.
16-original. **RESUME HERE — resolve the `i_pack`/FMS group-halo-update ownership
    question before writing any more of the Fortran wrapper.** Found this
    pass (Prompt 16); full detail in the Prompt 16 lead paragraph above and
    `prompt_14_standalone_cuda_integration.md` §7.2. Concretely:
    `fv_dynamics.F90:620/626/627/630` starts FMS group-halo updates
    (`i_pack(1)` for `delp`/`pt`/`q_con`, `i_pack(8)` for `u`/`v`
    `gridtype=DGRID_NE`) and passes `i_pack(*)` into `dyn_core` at `:645` for
    completion inside it — a layer above and separate from `dyn_core`'s own
    CUDA six-tile halo exchange. Read FMS's `mpp_domains`
    `start_group_halo_update`/handle-completion contract (not yet read by any
    prompt) to determine whether the CUDA branch may safely skip issuing
    these calls, or must complete them some other way, before writing §5 (the
    Fortran `ISO_C_BINDING` wrapper) or branching the real call site at §6.
    Escalate to Opus if the FMS contract itself is ambiguous or under-
    documented, per the task's own model policy.

Carry forward the ordering that found 13 of 13 input-fidelity root causes:
**audit input fidelity — values, store order, registration, and *derived* config
quantities — before touching arithmetic.** #11 (`n_con`) and #13 (`nf_ke`) are
the newest reminders that a hardcoded config value with a plausible-looking
in-source justification is still an input-fidelity defect: in both cases the
comment justified a *different* knob, and in #13 the justification had become
self-fulfilling — `nf_ke` was pinned to 1 to avoid a stub, which then guaranteed
the stub was never reached and so never got implemented.

Two reading rules for `tools/savepoints/compare_cuda_fortran.py`, both of which
briefly produced false FAILs this pass:
- The accepted verdict is the **defined box**. The parenthetical "full declared
  bounds, incl. both sides' undefined outer halo rings" figure is excluded by the
  established methodology and grades SIGNIFICANT at SP1/`sp2a` purely from the
  four diagonal corner halo cells, which have no donor tile.
- A large `max_abs` is not by itself a failure. `mfx`/`mfy` at SP2 carry
  `max_abs` ≈ 1.5 against their own dynamic range of ~4.2e+14 — `max_rel`
  ≈ 3.6e-15, ROUND-OFF.

## 10. Known Open Issues / Deferred Items

- **RESOLVED (Prompt 18, 2026-09-04) — the intermittent
  `fv3_halo_mpi_transport` failure. Recorded here as a standing CODING
  CONSTRAINT, because the hazard class is easy to reintroduce.** **A plain
  `cudaMemset` on device memory, and a synchronous `cudaMemcpy` H2D from
  pageable memory, are BOTH legacy-stream operations and are NOT ordered
  against a stream created with `cudaStreamNonBlocking`** — which is how
  `d.transfer_stream` and the transport test's exchange stream are both
  created, precisely to opt out of implicit legacy synchronization. Any clear,
  poison-fill, or seed of a buffer that a non-blocking stream will later read
  must be enqueued **on that stream** (`cudaMemsetAsync`/`cudaMemcpyAsync`).
  Two live instances were found and fixed: the transport test's `0xFF` poison
  memsets plus its `TileArray::create()` field-array H2D (the measured cause of
  the intermittent six-rank failure — whole-slot `POISON-REMAINS`, 6 of 12
  repeats), and, as hardening in the same class,
  `fv3_gpu_runtime.cpp:1290-1291`'s attach-time halo-buffer clears. Two useful
  artifacts to reuse rather than rebuild: **`FV3_HALO_TEST_LEGACY_SETUP=1`**
  reproduces the defect on demand from the fixed binary (3 of 6 repeats fail),
  which is how the fix was proven to be the differentiator rather than a
  timing coincidence; and **`FV3_HALO_TEST_DEBUG=1`** emits the per-block
  diagnostics (`BLOCK FAIL n/m cells | group … | tag … | slot offset … | slot
  census: poison/zero/other`) plus a `classify()` that distinguishes
  `POISON-REMAINS` from `ZERO` from `WRONG-VALUE`. Keep that distinction: a
  `0xFF`-poisoned R8 is NaN, `std::llround(NaN)` → `LLONG_MIN`, and the
  provenance oracle decodes that to the *same* text as a genuine 0.0
  (`tile 0 (-8,-8,0)`) — which is exactly why two prior passes read the symptom
  as "unwritten" and looked inside the transport. Detail:
  `prompt_18_mpi_transport_determinism.md`.
- **RESOLVED (Prompt 15, 2026-09-04) — was the BLOCKING item for standalone
  integration.** A real cross-process halo transport now exists
  (`runtime/fv3_gpu_halo_mpi.cpp`, host-staged MPI, six processes / one tile
  per rank) behind a `HaloTransport` interface, with `HaloPeerGroup` preserved
  as the default in-process backend, and it is validated BITWISE against that
  path at the dyn_core exit. Direction taken: `project_architecture.md`'s
  "Stage 1" host-staged MPI. See the Prompt 15 lead paragraph, §3/§4/§7/§8,
  §9 items 14–15, and `prompt_15_mpi_halo_transport.md`. Two limits recorded
  so they are not assumed away: the transport requires a communicator of
  **exactly 6 ranks** with a one-tile-per-rank map (it refuses anything else
  at attach), and the staging is **host-staged** — CUDA-aware MPI/GPU-direct
  (Stage 2) and comm/compute overlap (Stage 3) are still unimplemented, and
  are new backends behind the same interface rather than changes to it. The
  original finding is retained below for context.
  **(Historical, Prompt 14, 2026-09-04): the CUDA
  halo exchange transport (`HaloPeerGroup`, `fv3_gpu_halo_peer.hpp`) is
  single-process/six-host-thread only ("Model B") and has no cross-process
  (MPI or otherwise) counterpart anywhere in the codebase.** Real production
  is six separate MPI ranks/processes (`FV_NX=1`/`FV_NY=6`); `HaloPeerGroup`
  rendezvouses with a process-local `std::mutex`/`std::condition_variable`
  and copies between raw device pointers only valid in one address space —
  neither works across OS processes. Confirmed by grep that no real MPI API
  call exists in `cuda/dyn_core/runtime`/`cuda/dyn_core/include`, and that
  `fv3_gpu_create_desc::communicator_token` is unwired everywhere outside its
  own definition. This is a known, previously self-documented scope
  limitation from Milestone 3A.10 (`milestone_3a10_gpu_halo_exchange.md:471,955`),
  not a newly introduced defect. Closing it requires implementing a real
  cross-process halo transport (candidates: host-staged real MPI, matching
  `project_architecture.md`'s own deferred "Stage 1"; or CUDA-aware MPI/
  GPU-direct, its "Stage 2/3") — a runtime redesign, escalated to Opus rather
  than attempted here. See §9 item 14 and
  `docs/agent_handoff/prompt_14_standalone_cuda_integration.md`.
- **New platform constraint (Prompt 12, 2026-09-04): `ncu` (Nsight Compute)
  profiling is BLOCKED by GPU performance-counter permissions on this
  Discover `gpu_a100` allocation.** Both scoped `ncu` invocations in job
  58238626 (`--kernel-name a2b4_global_corner_kernel`) returned
  `ERR_NVGPUCTRPERM — The user does not have permission to access NVIDIA GPU
  Performance Counters on the target device`. The wrapped application ran
  normally in both cases; only counter collection was denied. This is a
  driver/administrative restriction (`NVreg_RestrictProfilingToAdminUsers`-
  style), not a flag or kernel-name error, and not fixable from an
  unprivileged job submission. Any future task that needs `ncu` (as opposed
  to `nsys`, which has worked throughout Prompts 7/9/11/13) should either
  request NCCS grant counter access for this account, or plan to rely on
  source-reading/`nsys`-only evidence instead. Detail:
  `prompt_12_a2b4_kernel_tuning.md` §2. **This constraint is now the binding
  one for further performance work** — see the next item.
- **New performance finding (Prompt 13, 2026-09-04): the workload is
  GPU-EXECUTION-BOUND, so launch-count optimization is no longer the right
  lever.** Measured, not inferred: a −31.6% reduction in `cudaLaunchKernel`
  per dyn_core call (16,182 → 11,067) bought only **−5.14%** wall clock, and
  `cudaStreamSynchronize` total time rose **0.44 → 135.5 ms** (Max 0.036 →
  12.999 ms) — the host now waits on the device. The `wall ≈ launches_per_tile
  × launch latency` model that fit Prompt 7 (92%) and Prompt 11 (87%) is
  therefore **retired**; do not use it to estimate a future optimization's
  value. Consequence: **Direction B (explicit stage/tile batching without
  graphs) is de-prioritized** even though its largest candidate is
  numerically risk-free (collapsing `c_sw`'s per-level launches, −5,670/call =
  −35.3%; level independence proven three ways — driver comment at
  `cgrid/fv3_cgrid_predictor.cu:710-716`, a per-level-disjoint arena, and zero
  `k±1` accesses), because it would now be worth only ~5% for a MODERATE-scope
  change across ~31 kernel signatures. Real further speedup requires
  kernel-time/occupancy work on `riem_solver3_kernel`/`riem_solver_c_kernel`,
  which is gated on the `ncu` permission above. Detail:
  `prompt_13_architectural_optimization.md` §8.
- **CUDA Graph mode (`--cuda-graph`): FUNCTIONAL, opt-in, validated
  bitwise — but its scope is narrow.** Prompt 13 repaired the missing
  `cudaGraphLaunch` in the capture arm (Prompt 8's "replay diverges" defect).
  Recorded limits, so they are not rediscovered or accidentally widened: the
  captured region is **`dgrid_dynamics` only** and deliberately excludes the
  halo exchange (bounded by `resolve(H_CWIND)` before and
  `resolve(H_DSCALAR)` after, capture mode `ThreadLocal`); capture freezes
  `DSwOptions::dt = bdt/n_split` by value into every node, so replay **refuses
  with `FV3_GPU_UNSUPPORTED_CONFIGURATION`** if `bdt`/`n_split` differ from
  the capture-time pair (a fixed timestep for the context lifetime is
  required — a variable-timestep configuration would need
  `cudaGraphExecKernelNodeSetParams` or re-capture); one capture per context
  per benchmark repeat, ~11.9–16.2 ms each, which must be amortized. Default
  remains off, so the default path is provably unchanged. Prompt 8's replay
  crash did not reproduce across 1,770 replays, but that is **absence of
  reproduction, not a proven root cause** — if it ever resurfaces, treat it as
  open again.
- **`nf_ke` / corner service: IMPLEMENTED** (was Gate-C2-blocking). The exit
  `pt` now validates at ROUND-OFF SCALE; see §4 #12/#13 and
  `prompt_6b_cuda_fortran_validation.md` §21. Three things came out of it that
  are worth not re-deriving:
  - **`copy_corners`'s `ng` is the compile-time PARAMETER** imported at
    `tp_core.F90:52` (`use fv_mp_mod, only: ng`) = 3, **not** `bd%ng` and **not**
    the pass's `nt`. The rotated block is always 3×3 on every pass, so a corner
    request's `halo_width` is the requester's *write band*, never a block size.
  - **`del2_cubed` rotates the corners twice per pass**, once per flux
    direction — dir=1 at `dyn_core.F90:2419` before the `fx` loop and dir=2 at
    `:2437` between the `fx` and `fy` loops. Only the first had a call site in
    CUDA.
  - The new `corner_edge_service_cuda` supports exactly
    `operation=copy, orientation=scalar, stagger=A, axis=x|y` and refuses
    anything else with a specific message. Everything outside that surface
    (`fill`, `fill_pair`, `vector_transform`, `a_to_b_edge`, B/C/D staggers) is
    still unimplemented — extend it there when a future path needs one, rather
    than loosening the guard.
- **Deferred, NOT localized: `fv3_dispatch_smoke` Phase C fails if the test's
  `cfg.nf_ke` is raised from its pinned 1 to the authoritative 3.** The pin keeps
  `nt = 0` on the single del2 pass, so the smoke test never requests the corner
  service; raising it as optional extra coverage produced `Phase C result: FAIL`
  / `Dispatch smoke test: FAIL` (21/22) and was **reverted** — three `gpu_a100`
  attempts to capture the diagnostic all stayed `PD (Priority)`, so the cause is
  unknown, and the 22/22 recorded in §5 is the state with the pin at 1. Already
  ruled out by inspection: `heat_source` is registered on the full
  `is-3..ie+3 / js-3..je+3` box with `lower[2] == 0`, satisfying every guard the
  service applies, and Phase C does wire the production service. **Not evidence
  against the service** — it is exercised on real A100 hardware by the
  authoritative-SP0 bench on all six tiles, where the exit `pt`/`pkz` validate at
  round-off. Localize before relying on `nf_ke > 1` in that test.
- **`omga` finalization: IMPLEMENTED and VALIDATED** (was the last
  Gate-C2 blocker). Detail in §4 and `prompt_6b_cuda_fortran_validation.md` §22.
  Two documentation corrections came out of it:
  - `pem` is allocated and filled **inside** the acoustic loop at
    `dyn_core.F90:503-517` under `if (it==n_split .and. end_step)`. The
    "pre-loop at `:505-513`" previously recorded in §9 was **stale**.
  - The omega finalization runs **after** the E11 seam block (`:1176-1188`),
    still inside the loop — not after the loop.
  - Scope of the new `adv_pe`/`omega_*` operators: they support the
    `use_old_omega = .true.`, non-hydrostatic, `grid_type < 3`, non-nested path
    only. `NumericalConfig` has no `use_old_omega` knob, so the `.false.` forms
    (`:797-801`, `:838-842`, `:1219-1237`) and the `:1238` hydrostatic `id_ws`
    diagnostic remain unimplemented. Extend deliberately if a future
    configuration needs them; do not loosen a guard to reach them.
- **Gate C2 is CLOSED, but the claim is configuration-scoped and is an R8
  claim.** **RESOLVED for a narrower R4 scope (Prompt 20, 2026-09-08): R4 is
  now savepoint-compared against Fortran and PASSES** at the same short-run
  envelope as Prompt 17 (c8/npz=10/n_split=3/k_split=1/6 dyn_core calls,
  R4-specific thresholds — see the Prompt 20 lead paragraph and
  `prompt_20_r4_numerical_validation.md`). **EXTENDED (Prompt 21,
  2026-09-08): R4 now also PASSES at the same longer/multi-`k_split` depth
  R8 reached at Prompt 19** — Stage A (k_split=1, 72 calls) and Stage B
  (k_split=2, 12 calls), same configuration otherwise. See the Prompt 21
  lead paragraph and `prompt_21_longer_r4_validation.md`. One finding to
  carry forward, not a defect: `omga`'s *bulk*-metric discrepancy shows real
  SUPERLINEAR growth over Stage A's 72 calls (0.06%→4.06% of its own RMS),
  driven by the field's own magnitude decaying rather than by the absolute
  CUDA/Fortran error growing (which plateaus after call ≈43) — re-check this
  trend if R4 is ever run past 72 calls. R4 validation still does **not**
  extend to any wider grid than c8 — continue to report R4 separately from
  R8 and do not infer either from the other outside the specific envelope
  each has actually been tested at.
- **E11: IMPLEMENTED** (was Gate-C2-blocking; see §4 for the implementation and
  the bitwise verification). Two documentation corrections that came out of it,
  recorded so they are not re-broken:
  - The authoritative line numbers are `dyn_core.F90:1178` (call) and
    `:1181-1188` (write-back). The `:1142`/`:1145-1152` previously recorded here
    were **stale**.
  - `fv3_gpu_halo_exchange.cu:48-51` and `fv3_gpu_halo_topology.hpp:293-296`
    stated `mpp_get_boundary` returns into `ebuffer`/`nbuffer` and "never into
    the field array itself" — **wrong**; `:1181-1188` copies both buffers
    straight back into `u`/`v`. Corrected in `fv3_gpu_halo_topology.hpp`.
  - The inherited claim that ring-0 is "structurally unreachable" in the halo API
    is **retired**: `HaloGroupSpec::ring_begin = 0` reaches it, and E11 now uses
    that path in the integrated runtime.
- **`heat_source`/`diss_est` per-substep accumulation: IMPLEMENTED** (was
  Gate-C2-blocking; see §4). Its classification as INERT for all in-loop
  savepoints was correct and is confirmed empirically — all 15 in-loop boundaries
  are unchanged by it. Only the exit `pt`/`pkz` are affected.
- **Proven non-authoritative but proven INERT under `a_imp = 1.0`** (recorded so
  they are not rediscovered as "new" defects). `nh_core.F90:160-176` dispatches
  `Riem_Solver3` on `a_imp`; `a_imp > 0.999` selects `SIM1_solver`, which takes
  neither `ms` nor `scale_m` — confirmed on the CUDA side too, where
  `sim1_column` takes neither:
  - `cfg.scale_m = 1.0` vs authoritative `scale_z = 0.0` (`fv_arrays.F90:337`);
  - `cfg.m_split = 2` vs authoritative `m_split = 0` (`fv_arrays.F90:632`);
  - `cfg.delt_max = 0.002` vs authoritative `1.0` (`fv_arrays.F90:759`) — the
    post-loop heat-source limiter. Now **PROVEN inert at the dyn_core exit**, not
    merely assumed out of window: measured max |dtmp| = 1.0305e-03 against the
    smallest bound `delt(k=1) = 0.1*|bdt*delt_max| = 0.720` even at 0.002 —
    698× headroom, so the limiter never engages and cannot contribute to the
    exit `pt`.
  Deliberately not changed while inert (each would perturb the default synthetic
  path the 3A.11T/U/V baseline was measured on). **Must be corrected before any
  `a_imp <= 0.999` configuration and before GEOS integration.**
- **Recorded, deliberately not fixed: `test_dispatch_smoke.cpp:535-538` and
  `:756-757` carry the identical interleaved-vs-registered lon/lat layout
  mismatch** that was root-caused in the harness (§4 #9). Self-contained tests
  with their own tolerances, not compared against Fortran, so leaving them was a
  scope decision, not a correctness judgement. Fix them the next time that file
  is touched.
- **Deferred (quantified, deliberately not changed): `MOIST_CAPPA`/`USE_COND`
  are compiled in the authoritative Fortran but not in CUDA.** The fixture's
  `@fvdycore/CMakeLists.txt:88-89` sets both, confirmed in the generated
  `flags.make`. Proven round-off-neutral **for this fixture**, not assumed:
  `q_con ≡ 0.0` exactly on all 6 tiles (0/1960 nonzero) ⇒ `peg ≡ pem`. Refined
  this cycle: the `MOIST_CAPPA` build is also what **separates the Exner scalar
  from the closure kappa**, which is the authoritative structure the
  `akap_exner` split now reproduces (§4 #8) — so the residual gap is narrower
  than previously recorded, but the `(pem+dm)−pem` vs `dm` non-associativity
  (~1e-16 relative) remains. Must be closed before any moist (`q_con ≠ 0`)
  configuration or GEOS integration.
- **Stale in-source documentation at `run_small_perf_bench.cpp:808-835`**
  (documentation defect, no numerical defect). That comment asserts CUDA runs
  two disagreeing gravities, that `fv3_vertical_kernels.cu` "never references
  grav at ALL", and that the authoritative `grav` is "undetermined". All three
  are false: the plumbing was repaired in milestone 3A.11V
  (`riem_solver3_column` takes a threaded `T grav`), the launcher passes
  `fv_real(cfg.grav)`, and `geos_constants.fh:38` establishes `GRAV = 9.80665`
  as authoritative. Correct the comment on the next pass that touches that file.
- **Retracted claim, recorded so it is not rediscovered:** an intermediate
  revision of the `edge_*` fix also changed `make_host_state`'s
  `cube::CubeGrid` tile argument to `tile + 1`, on the theory that a 0-based
  MPI rank was being passed to a 1-based parameter. **Wrong** — the harness
  `tile` is already 1-based (`run_small_perf_bench.cpp:2279` derives its on-disk
  directory as `"tile" + (tile-1)`). `CubeGrid` tile 7 falls through
  `tile_rotate`'s `default` while `corner_owner` reads it as odd parity, and the
  inconsistent halo mapping produced NaN in `u`/`v`/`delp` at `it=1`. Reverted.
  Lesson: read an index-base convention off an actual caller, never infer it —
  and note the `edge_*` block's dx/dy authenticity self-check **cannot** catch
  this class of error, because distances are rotation-invariant and the six
  panels are pure rotations of one another, so it proves panel shape only, not
  tile identity.
- ~~**`it=3` not compared.**~~ **RESOLVED** — `it=3` is now compared and passing
  at all five in-loop boundaries (§4). The formerly recorded "`it=3` NONFINITE on
  tiles 2/3/4" symptom does not reproduce. E11 is live exactly at `it=3` for
  n_split=3, and its footprint there was measured directly (only
  `u(is:ie,je+1,:)` / `v(ie+1,js:je,:)` change; 0.0 off-line).
- ~~**`pe`'s one Fortran halo ring is NOT WRITTEN by CUDA**, a coverage gap
  with no known downstream consumer~~ — **RESOLVED (Prompt 17), and the
  "no consumer" half of this item was WRONG.** The ring *is* consumed:
  `Lagrangian_to_Eulerian`'s D-grid wind/flux vertical remap reads it at
  `fv_mapz.F90:686/693` (driving `u`/`mfy`/`cy` at `j=js` and `j=je+1`) and
  `:729/740` (driving `v`/`mfx`/`cx` at `i=is` and `i=ie+1`). Leaving it
  unwritten produced an O(1) standalone divergence from the **second** dyn_core
  call onward (`u` 1.845e+00 confined to exactly `j=js`/`j=je+1`, `v` to
  exactly `i=is`/`i=ie+1` — the footprint is what identified the consumer),
  plus a *uniform-in-k* `pt` offset of 1.032e-04 via the `consv_te = 1.0`
  energy fixer. Fixed by reproducing `pe_halo` (`dyn_core.F90:1504-1532`, called
  at `:1011` under `remap_step`) on the host after the D2H as
  `fv3_gpu_interface_pe_halo`; bit-for-bit equivalent because no `delp` write
  exists between `:1011` and `:1375`, so the ring is exactly
  `cumsum(ptop, exit delp ring)` and `delp`'s own ring is already host-resident.
  `pe_halo` is `private` (`:142`), hence reproduced rather than called. The
  standalone-comparison narrowing note still applies to
  `compare_cuda_fortran.py`'s `SP3_DEFINED_BOX["pe"]`, which is unchanged.
  **General lesson worth keeping: "unwritten but apparently inert" is a claim
  about consumers, and it was never checked past `dyn_core`'s own exit.** Two
  sibling boundary items were closed in the same fix — the four flux
  capacitors (`mfx`/`mfy`/`cx`/`cy`) and `ws` were also never copied out, and
  `z_rat`/`ut`/`vt`/`peln` descriptors had wrong bounds.
- **`clamp_nord_cap` (D1)** caps the WP-B group loop at `ng-1==2`, so
  `nord_k==3` silently gets zero divergence damping while still returning
  success. **Not a defect at production `nord=2`** — `clamp_nord_cap(2,3) = 2`,
  the launcher loop is inclusive, and every `del6_vt_flux` extent, flux-loop
  bound and `copy_corners` placement matches Fortran for `nord=2`/`ng=3`. What
  remains is the dead `nord_k == 3` branch (would silently under-damp if a
  future configuration requested it) and the docs' incorrect `nord=1-3` support
  claim.
- **`del6_vt_flux` n==1 uninitialized `d2` ring**: a dead-but-real uninitialized
  read, reachable only at `ndif==1` (not exercised by production `nord=2`, which
  yields `ndif ∈ {0,2}`). Value-harmless in both Fortran and CUDA. Deferred: add
  an `ndif==1` case to `test_wp_b_cuda_smoke.cpp` to reproduce under initcheck,
  then decide on widening `del6_init_kernel`.
- **k=0 damping row**: root-caused to `n_sponge = 9` and **fixed for validation
  mode**. Still open for the default/performance path, which retains
  `geos_production_flags()`'s GEOS values by design; harness-only, not implicated
  in any measured instability.
- **`docs/agent_instructions/cost_efficiency.md` does not exist yet** —
  referenced by §12 below but not yet created; create separately, do not inline a
  large policy here.
- Smaller recorded-not-fixed items (see `milestone_3A11T_...md` §7 and
  `milestone_3A11U_...md` "Genuine port defects" for detail): `divg2`
  read-without-writer guard; write-write race in the smag arena outer ring
  (appears inert); halo documentation defect at
  `fv3_gpu_halo_exchange.cu:44-52` / `fv3_gpu_halo_topology.hpp:289-297`
  (3 inaccuracies, contradicted with specifics in the 3A.11U handoff and above).

## 11. Hard Constraints

- Authoritative GFDL FV3 Fortran (`geos/v3.0.0`, `ca30e26...`) is the
  numerical source of truth.
- Do not modify `geos-gtfv3`.
- Numerical validation may now be claimed **only** within the closed Gate C2
  scope (§7): c8 / npz=10 / n_split=3 / k_split=1 / R8 / six tiles / production
  first-run numerics. Any other grid, precision or configuration is still
  unvalidated and must be reported as such.
- Preserve persistent GPU residency; no full-state H2D/D2H between work
  packages.
- Use real six-tile halo semantics; do not stub communication for integrated
  validation.
- Do not populate `@fvdycore` in this repo without a demonstrated dependency,
  documented first.
- Generated build directories, Slurm logs, sanitizer logs, scratch scripts and
  CUDA dump directories are not committed.
- Do not optimize numerical kernels before correctness localization, unless a
  task explicitly requests performance work (current phase does not).
- Do not push or merge unless explicitly requested.

## 12. Cost-Efficient Agent Rules

See `docs/agent_instructions/cost_efficiency.md` (not yet created as of this
writing — create as its own file if a dedicated policy is needed; do not
inline one here). Until then, apply directly:

- Read this file (`CURRENT_STATE.md`) first; avoid re-reading superseded
  milestone handoffs unless an open item above requires the detail.
- Use targeted greps/searches over broad repository scans.
- Use bounded, coarse-to-fine debugging when localizing numerical divergences
  (savepoint-by-savepoint, not whole-run diffs). Every Prompt 6B root
  causes (all thirteen, as it turned out) were found by auditing **input
  fidelity — values, store order, registration and *derived* config quantities —
  before touching kernel arithmetic**; keep that ordering. Where a needed
  authoritative input genuinely does not exist in the port, prefer **deriving it
  from geometry already proven authoritative and then verifying the derivation
  against a trusted export** over adding new Fortran instrumentation.
- Prefer the shortest Slurm job that answers the question (short `srun` probes
  over long `sbatch` runs where possible).
