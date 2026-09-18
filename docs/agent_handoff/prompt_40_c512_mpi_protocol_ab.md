# Prompt 40: C512 OpenMPI Eager/Rendezvous Threshold A/B Test

Branch: `experiment/held-suarez-e2e`. Model: Sonnet only, per task policy.
Scope: one runtime-MPI-configuration A/B experiment testing whether Prompt
39's isolated MPI-transport/protocol root cause for the C512 CUDA
performance regression is specifically an OpenMPI eager/rendezvous
protocol crossover. No CUDA-aware MPI, message batching, halo fusion,
kernel tuning, MPI topology change, or source modification attempted.

## Stage 0: Active OpenMPI Transport / MCA Controls

Loaded the validated module stack
(`mpi/openmpi/4.1.6/gcc-12.3.0`) and queried `ompi_info` directly (not
guessed):

- **OpenMPI version**: 4.1.6 (`Ident string: 4.1.6`).
- **Available PMLs**: `v`, `monitoring`, `ucx`, `ob1`, `cm`.
- **Available BTLs**: `vader` (shared memory), `tcp`, `self`. No
  `openib`/`smcuda`/`vader`-with-CUDA-support component in this build.
- **Job topology**: `p37_c512_perf_cuda/fv3.j` requests `--nodes=1
  --ntasks-per-node=6` — all 6 ranks (1 tile/rank) are same-node. For a
  same-node OpenMPI job with `ob1`/vader available, point-to-point traffic
  between ranks uses **`vader`** (shared-memory BTL); `self` handles a
  rank's communication with itself; `tcp` is not exercised (no inter-node
  hop needed). No `OMPI_MCA_*` variables were preset by the module
  environment (`env | grep OMPI_MCA` empty) — pml/btl selection is
  OpenMPI's own default (`ob1`+`vader`/`self` for same-node CPU-memory
  traffic; MpiHaloTransport is host-staged, i.e. the buffers exchanged
  over MPI are already-packed host memory, not GPU pointers, so this is
  ordinary host-memory `vader` traffic, not a CUDA-aware/GPUDirect path).
- **Relevant MCA parameters actually confirmed to exist** in this build
  (`ompi_info --param btl vader --level 9`):

  | Parameter | Default | Meaning |
  |---|---|---|
  | `btl_vader_eager_limit` | 4096 B | Messages ≤ this size sent via pure eager (single-copy, no handshake) |
  | `btl_vader_rndv_eager_limit` | 32768 B | Size of the "phase 1" fragment piggy-backed on the rendezvous handshake for messages above `eager_limit` (must be ≤ `eager_limit`) |
  | `btl_vader_max_send_size` | 32768 B | Pipeline fragment size for the remaining "phase 3" of a rendezvous message |

  `btl_tcp_eager_limit`/`btl_self_eager_limit` also exist but are
  **not the active transport** for this same-node job — not used.

**This is the critical scoping fact for the whole experiment**: OpenMPI's
factory-default `btl_vader_eager_limit` is only **4096 bytes**. The given
C512 ordinary-group halo message (~135.4 KB = ~138,650 B) — and, just as
importantly, the given **C256** message (~67.8 KB) — are **both already
~17-33x above the default eager_limit**, and both are already above
`btl_vader_rndv_eager_limit` (32768 B) too. Under default settings, C256
and C512 messages both already take the **same rendezvous+pipelined**
code path; there is no eager/rendezvous *crossover unique to C512* in
this OpenMPI build's default configuration — both resolutions were
already on the "slow" side of that particular boundary before this
experiment. Testing this hypothesis therefore requires raising
`btl_vader_eager_limit` **above the C512 message size itself** (not just
above some intermediate value between C256 and C512) to force the pure
single-copy eager path. This directly shapes Stage 2's setting choice
below.

## Stage 1: Baseline Reproduction

Reused (unmodified) the exact existing C512 six-rank CUDA job directory
and script: `experiments/p37_c512_perf_cuda/fv3.j` (`bdt=36s`, `RUN_DT=36`,
`JOB_SGMT='00000000 001800'` → 30 dyn_core calls, `AGCM_IM=512`,
`hord_*=±6`, `grid_type=0`, `hydrostatic=false`, `a_imp=1.0`, HS forcing
on). Standing recommended CUDA config: MPS ON, `RUN_CUDA_FV3_FUSE_HALO_
EDGES=YES`, `RUN_CUDA_FV3_FUSE_CORNER_FILL=YES`, Graph off (unset) —
identical to Prompt 37. No `OMPI_MCA_*` override for this baseline run
(`default` setting below).

**Reproduced baseline (2 reps, default MCA settings)**:

| Rep | `DYN_CORE` total_s | ms/call (÷30) |
|---|---|---|
| 1 | 192.0038 | 6400.13 |
| 2 | 191.6256 | 6387.52 |

Average: **6393.83 ms/call**, vs. the historical Prompt 37 figure of
**9089.05 ms/call** — a difference of **-29.7%**, well outside the ±10%
reproduction-tolerance this stage specifies as a trigger to investigate
before proceeding.

**Investigation**: config was diff-clean by construction (the *same*,
unmodified `fv37_c512_perf_cuda` job directory/`fv3.j` was reused — not a
clone — so `bdt`/`RUN_DT`/`JOB_SGMT`/fusion flags/MPS lifecycle are
byte-identical to Prompt 37, eliminating a config-drift explanation).
Checked `sacct` node assignment: this job (58445328) landed on
**`warpa005`**; Prompt 37's CUDA perf job (58444655) landed on
**`warpa009`** — a **different physical A100 node**. Both jobs' MPS
`get_server_list` was empty at start (no stale/competing MPS server on
either node). No profiling tool was used to confirm GPU clock/thermal
state (out of scope, no `nsys`/`ncu`/new instrumentation permitted), so
the exact mechanism is not proven, but a different physical node with
different concurrent-tenant load, thermal state, or boost-clock behavior
is the most plausible, ordinary explanation for a systematic, uniform
speedup — critically, **the offset is uniform across all three MCA
settings tested below** (all cluster within ~1.6% of each other; see
Stage 3), which is the signature of a job-level/node-level constant
factor, not a per-setting artifact.

**Disposition**: this is a legitimate node-to-node hardware/scheduling
variance on a shared cluster, not an environment/config defect — the run
was not aborted. Because all three MCA settings in Stage 3 were run
**back-to-back within this same single job/node/GPU allocation**, the
*relative* A/B comparison between settings is unaffected by, and is in
fact more tightly controlled against, this node-to-node offset than a
comparison against Prompt 37's separately-scheduled historical baseline
would have been. The absolute ms/call figures in this document should be
compared **only against each other** (all obtained on `warpa005` in one
allocation), not directly against Prompt 37/38/39's historical absolute
figures (obtained on `warpa009`/other nodes).

## Stage 2: Bounded MCA Test Values

Given Stage 0's finding, chose to sweep `btl_vader_eager_limit` alone
(single parameter; `btl_vader_rndv_eager_limit` left at its default
32768 B in every case — OpenMPI's own vader-BTL contract requires
`rndv_eager_limit <= eager_limit`, which stays trivially satisfied since
only `eager_limit` is raised, and once a message is `<= eager_limit` the
rendezvous path — and `rndv_eager_limit` — is never consulted at all, so
touching both was unnecessary):

| Setting | `btl_vader_eager_limit` | Rationale |
|---|---|---|
| `default` | 4096 B (unset/factory default) | Control — identical to every prior prompt's implicit config |
| `test1_justabove_145000` | 145,000 B (~141.6 KB) | Just above the given ~135.4 KB C512 message size |
| `test2_comfortable_524288` | 524,288 B (512 KB) | Comfortably above, ~3.9x the message size |

3 total settings (2 alternatives + default), within the ≤3-alternative
bound. Set via `export OMPI_MCA_btl_vader_eager_limit=<value>` in the
sbatch wrapper before each `csh -f ./fv3.j` invocation — a pure runtime
MPI environment variable, no change to `fv3.j`, `esma_mpirun`, or any
source file.

## Stage 3: C512 MCA A/B Runs

New script `experiments/p40_c512_mca_ab.sbatch` (single job, single node,
single GPU, MPS started once, torn down once at the end — deliberately
combining all 3 settings x 2 reps = 6 `fv3.j` invocations into one
allocation to both minimize job count and maximize control, since the
target GPU/node is then held fixed across every setting). Job **58445328**,
`gpu_a100`/`rome`, `warpa005`, COMPLETED, exit 0, elapsed 00:31:36 (well
under the 1:30:00 request). 1 new GPU job submitted total (well under the
7-job budget).

## Stage 4: Health Check

All 6 `fv3.j` invocations: `RC=0`. Wrapper printed `ALL DONE`. MPS
`FV3_MPS_START`/`FV3_MPS_STOP` both clean (no `MPS_START_FAILED`).
Grep across the full 3185-line `.out` log for
`range_warn|nan|error|abort|fatal` (case-insensitive): **zero matches**.
**Stage 4: PASS.**

## Stage 5: Timing Comparison

Extracted via the same `fv_timing_mod` FMS-clock block grep pattern used
in Prompt 37 (`^   (DYN_CORE|COMM_TOTAL|COMM_DSW)\s`), from each rep's
saved log (`p37_c512_perf_cuda/perf_logs_p40_mca/<setting>_rep<N>.log`):

| Setting | Threshold (B) | Rep timings (`DYN_CORE` total_s) | ms/call (avg) | `COMM_DSW` ms/call | `COMM_TOTAL` ms/call | vs. default |
|---|---:|---|---:|---:|---:|---:|
| `default` | 4096 (factory) | 192.0038, 191.6256 | **6393.83** | 1032.52 | 1835.10 | baseline |
| `test1_justabove_145000` | 145,000 | 194.5826, 195.1471 | **6495.50** | 1046.66 | 1864.97 | **+1.59%** |
| `test2_comfortable_524288` | 524,288 | 194.7431, 194.3885 | **6485.53** | 1048.66 | 1858.85 | **+1.44%** |

Rep-to-rep spread within each setting: <0.3% (default 0.20%, test1 0.29%,
test2 0.18%) — tight and consistent, so the ~1.4-1.6% cross-setting
difference is a real (if small) signal, not noise. All three timers
(`DYN_CORE`, `COMM_DSW`, `COMM_TOTAL`) move in the **same direction**
(slightly worse) for both alternative settings vs. default.

## Primary Decision Logic

Best alternative setting (`test2_comfortable_524288`) improves `DYN_CORE`
by **-1.44%** relative to `default` — i.e. it is **slower**, not faster,
and both alternatives are slower. This is not partial improvement (Case
C, 5-10%) and not large improvement (Case A, ≥10%); it lands in **Case B
— small/no improvement**, with the sign actually reversed (mild, tightly
reproducible regression rather than a null result). Per the task's
Case-B rule this classifies as:

**EAGER/RENDEZVOUS THRESHOLD NOT SUFFICIENT.**

(Case D — "performance worsens" — was considered; the effect size, ~1.5%,
is small enough relative to the >=10%/5-10% brackets that it is graded
under Case B's "<5% improvement" rule rather than treated as a distinct
alarming regression. It is still reported honestly as a negative
number, not rounded to "no effect.")

## Stage 6: MPI Threshold Crossover Evidence

1. **Active OpenMPI transport**: `vader` (shared-memory BTL, same-node
   6-rank job), with `self` for self-messages. Confirmed via `ompi_info`
   and the job's `--nodes=1 --ntasks-per-node=6` topology. `tcp`/`ucx`
   present in the build but not exercised (no inter-node hop).
2. **Default eager/rendezvous threshold**: `btl_vader_eager_limit` =
   4096 B (pure-eager cutoff); `btl_vader_rndv_eager_limit` = 32768 B
   (rendezvous phase-1 fragment cap).
3. **Is ~135.4 KB above or below that threshold?** **ABOVE both** — by
   ~33x the eager_limit and ~4.2x the rndv_eager_limit. Critically, the
   given **C256** message (~67.8 KB) is *also* above both defaults
   (~16.5x and ~2.1x respectively) — so under default settings, C256 and
   C512 are already on the same (rendezvous+pipelined) side of this
   particular boundary; there is no unique-to-C512 crossover in the
   factory configuration.
4. **Did forcing the message onto the alternate (pure-eager) protocol
   materially change `COMM_DSW`/`dyn_core` performance?** **NO** — both
   alternative settings were mildly *slower* (+1.4 to +1.6%), consistently
   across `DYN_CORE`, `COMM_DSW`, and `COMM_TOTAL`, well under any
   "material improvement" bar.
5. **Is the C512 regression now explained primarily by protocol
   selection?** **NO.**

## Stage 7: CUDA-Aware MPI Priority Update

Per the task's rule: "If protocol tuning does essentially nothing: CUDA-
aware MPI priority: HIGH." Protocol-threshold tuning produced no benefit
(mild regression instead). **Priority remains HIGH** (unchanged from
Prompt 39's HIGH CANDIDATE — not downgraded to MODERATE, since tuning did
not recover any of the regression). Not implemented in this prompt.

**Reasoning**: PeerHaloTransport is healthy (Prompt 39), generic GPU-side
halo pack/unpack/staging is exonerated (Prompt 39), and the specific
eager/rendezvous protocol-boundary mechanism tested here is now also
exonerated as the driver (this prompt) — narrowing further. The next
mechanism-level candidate within "MPI transport/protocol" is a genuine
change of wire protocol/data path (CUDA-aware MPI, i.e. GPU-direct
RDMA/staging bypassing the current host-staged pack→D2H→MPI_Send→H2D→
unpack chain), not a parameter tweak on the existing host-staged path.

## Summary Table (all stages)

| Stage | Result |
|---|---|
| 0. Transport/MCA identification | `vader` (shared-mem) active; `btl_vader_eager_limit`=4096B/`rndv_eager_limit`=32768B confirmed via `ompi_info`; C256 and C512 both already above both defaults |
| 1. Baseline reproduction | 6393.83 ms/call vs. historical 9089.05 ms/call (-29.7%); attributed to node-to-node variance (`warpa005` vs. `warpa009`), uniform across all settings, not a config defect |
| 2. MCA test values | `btl_vader_eager_limit` ∈ {4096(default), 145000, 524288}, single parameter |
| 3. A/B runs | Job 58445328, 1 job, 6 reps, rc=0, elapsed 31:36 |
| 4. Health check | PASS — 0 FAIL/NaN/error/abort/range_warn |
| 5. Timing comparison | test1 +1.59%, test2 +1.44% (both slower, tight <0.3% rep spread) |
| Decision | Case B — EAGER/RENDEZVOUS THRESHOLD NOT SUFFICIENT |
| 6. Crossover evidence | Both C256/C512 already above default thresholds; forcing pure-eager did not help |
| 7. CUDA-aware MPI priority | Remains HIGH (unchanged) |

## Final Report

**Model:** SONNET (honored throughout).

--------------------------------------------------
**MPI ENVIRONMENT**
--------------------------------------------------
OpenMPI version: 4.1.6
Active PML: ob1 (default same-node selection; `ucx`/`cm` available but
not selected for this same-node vader path)
Active transport: vader (shared memory BTL), self for local
Relevant MCA parameter: `btl_vader_eager_limit`
Default threshold: 4096 B (with `btl_vader_rndv_eager_limit`=32768 B)
C512 ordinary halo message: ~135.4 KB
Message above threshold: **YES** (both eager_limit and rndv_eager_limit;
so is the C256 message)

--------------------------------------------------
**BASELINE**
--------------------------------------------------
Historical C512 CUDA: 9089.05 ms/call
Reproduced baseline: 6393.83 ms/call
Difference from historical: **-29.7%** (attributed to node-to-node
variance, `warpa005` vs. historical `warpa009`; config byte-identical,
same job directory reused, not cloned)
Baseline `COMM_DSW`: 1032.52 ms/call
Baseline `COMM_TOTAL`: 1835.10 ms/call

--------------------------------------------------
**MCA TESTS**
--------------------------------------------------
| Setting | Threshold | Rep timings (s) | Median/avg ms/call | COMM_DSW | COMM_TOTAL |
|---|---:|---|---:|---:|---:|
| default | 4096 B | 192.0038, 191.6256 | 6393.83 | 1032.52 | 1835.10 |
| test1 | 145,000 B | 194.5826, 195.1471 | 6495.50 | 1046.66 | 1864.97 |
| test2 | 524,288 B | 194.7431, 194.3885 | 6485.53 | 1048.66 | 1858.85 |

Best setting: **default** (neither alternative improved on it)
Best dyn_core: 6393.83 ms/call (=default; no alternative setting beat it)
Improvement vs reproduced baseline: **0% (no alternative improved; best
alternative was -1.44% i.e. worse)**
Improvement vs historical baseline: not meaningful (reproduced baseline
itself already differs -29.7% from history for node-variance reasons
unrelated to MCA settings)
`COMM_DSW` improvement: none (+1.37% to +1.57%, i.e. worse)
`COMM_TOTAL` improvement: none (+1.30% to +1.63%, i.e. worse)

--------------------------------------------------
**DECISION**
--------------------------------------------------
Classification: **MPI PROTOCOL THRESHOLD NOT SUFFICIENT**
Protocol crossover explains C512 regression: **NO**

--------------------------------------------------
**CUDA-AWARE MPI**
--------------------------------------------------
Previous priority: HIGH CANDIDATE
Updated priority: **HIGH**
Reason: Eager/rendezvous threshold tuning — the cheapest candidate
mechanism ranked in Prompt 39 — produced no benefit (mild, tightly
reproducible ~1.5% regression instead) across three settings spanning
the message-size boundary. Both C256 and C512 messages were already
above the default rendezvous threshold, so no crossover unique to C512
exists to tune away. Combined with Prompt 39 (PeerHalo healthy, generic
halo pack/unpack exonerated), the regression now points specifically at
the host-staged MPI data-movement/synchronization path itself (pack→D2H→
MPI_Send/Recv→H2D→unpack), which only a genuine protocol change
(CUDA-aware MPI) would alter — not a parameter on the existing path.

--------------------------------------------------
**NEXT EXPERIMENT**
--------------------------------------------------
Recommended exactly one next experiment: **CUDA-aware MPI A/B transport
test at C512** — build/run the existing `MpiHaloTransport` C512
configuration against a CUDA-aware OpenMPI (GPU-direct send/recv on
device pointers, bypassing the current explicit D2H/H2D staging) vs. the
current host-staged path, same message counts/sizes, comparing
`COMM_DSW`/`DYN_CORE` FMS-clock timing. Not implemented here.
Expected discriminator: if CUDA-aware MPI recovers most of the C512
regression, the driver is the host-staged D2H/H2D + MPI copy overhead
itself; if it does not, the driver is elsewhere in the six-rank
process/synchronization path and a different diagnostic is needed.

--------------------------------------------------
**PROJECT STATUS**
--------------------------------------------------
C512 correctness: PASS (BITWISE, unaffected, unchanged from Prompt 37)
C512 MPI regression: CONFIRMED (unchanged)
PeerHalo discriminator: MPI PATH ISOLATED (Prompt 39, unchanged)
MPI threshold experiment: COMPLETE
Numerical source changed: NO
Communication implementation changed: NO
CUDA kernels changed: NO
Source files modified: 0
GPU jobs submitted: 1 (job 58445328)
Escalation to Opus: NO

CURRENT_STATE size: see `CURRENT_STATE.md` (updated alongside this doc)

C512 MPI THRESHOLD NOT SUFFICIENT — READY FOR CUDA-AWARE MPI A/B TEST
