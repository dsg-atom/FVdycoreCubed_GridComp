# FV3 CUDA Modernization — Reusable Work Summary

Distilled from the full project history (git log, `docs/agent_handoff/*`,
`docs/performance/*`, `docs/agent_instructions/*`, `CURRENT_STATE.md`) as of
2026-10-01. This is **not a chronological diary**. It exists so a different
engineer or agent, working on a *different* legacy-Fortran → C++/CUDA →
hybrid-CPU/GPU modernization, can reuse the architecture, workflow, and
hard-won lessons without re-reading ~130 prompt-level handoff documents.

Throughout this document, each claim is tagged:
- **[FACT]** — specific, verified thing that happened in this project.
- **[LESSON]** — a generalizable rule distilled from that fact.
- **[RECOMMENDATION]** — advice for a new project, less strongly evidenced.

---

## 1. Purpose

**[FACT]** The project modernizes the GEOS FV3 atmospheric dynamical core
(`dyn_core`: C-grid predictor, D-grid transport, vertical/nonhydrostatic
Riemann solver, pressure/grid operators) for heterogeneous CPU/GPU execution,
while treating the authoritative GFDL FV3 Fortran implementation as the
permanent numerical source of truth. The target path is:

```
GEOS/FV3 Fortran -> ISO_C_BINDING -> stable C ABI -> C++/CUDA runtime
    -> persistent GPU dyn_core subsystem
```

**[FACT]** An idealized dry-dynamics Held-Suarez (1994, BAMS) configuration
was adopted as the end-to-end validation vehicle, not the final goal.
Held-Suarez forcing touches only `u`, `v`, `pt`, never moisture/tracers, so it
exercises the full dyn_core numerics and the full six-tile MPI/halo machinery
at production resolutions (up to C512) without requiring moist-physics
parity. This let correctness and performance be validated at scale *before*
tackling full-GEOS moist-physics integration, which remains future work.

**[LESSON]** When modernizing one subsystem of a much larger legacy model,
find or build the smallest *scientifically real* end-to-end configuration
that exercises your subsystem's full numerical and communication surface
without requiring the rest of the model's complexity to be ported first.
A dry dynamical core inside a full model harness is a reusable pattern for
any coupled Fortran model (ocean, land, chemistry) where the dynamical
"skeleton" can be isolated from the "flesh" of physics packages.

**[RECOMMENDATION]** Decide this vehicle early and document explicitly what
it does *not* validate (here: moist physics, `a_imp<=0.999` configs). Revisit
only when a dedicated task is scoped for it.

---

## 2. Final Architecture Reached

```
GEOS/FV3 Fortran driver (FV_StateMod.F90 / fv_dynamics.F90)
        |
   ISO_C_BINDING bridge (cuda/interface/*.F90)
        |
   stable C ABI (fv3_gpu_api.h, fv3_gpu_communication.h)
        |
   WP-E C++ runtime: GPU context, FieldRegistry, persistent device state
        |
   CUDA dyn_core kernels: WP-A (C-grid) / WP-B (D-grid) / WP-C (vertical)
                            / WP-D (pressure/grid operators)
        |
   six-tile GPU-side cubed-sphere halo exchange
        (PeerHaloTransport in-process | MpiHaloTransport host-staged/CUDA-aware)
```

**[FACT]** Component status as of 2026-10-01 (`CURRENT_STATE.md` §4-§5):

| Component | Backend | Status | Reason |
|---|---|---|---|
| C-grid predictor (WP-A), D-grid dynamics (WP-B), vertical/NH (WP-C), pressure/grid ops (WP-D) | CUDA default | NUMERICALLY VALIDATED (Gate C2, bitwise/round-off) | Full 54-stage production first-run inventory GPU-executed and validated against Fortran savepoints |
| Six-tile halo exchange | CUDA default (`PeerHaloTransport`) | NUMERICALLY VALIDATED bitwise | In-process default; `MpiHaloTransport` (host-staged/CUDA-aware) is an interchangeable, equally-validated alternative |
| PT/DELZ remap | CUDA default | PRODUCTION DEFAULT | REMAPPING −33.7%, TOTAL −7.3% |
| REMAP_UV | CUDA default | PRODUCTION DEFAULT | Bitwise-exact, ~10.23x local, TOTAL +~1% |
| REMAP_Q | CUDA default | PRODUCTION DEFAULT | 3.23x local, TOTAL −1.51% |
| REMAP_PRESSUREVARS | CUDA default (escape hatch available) | PRODUCTION DEFAULT | Bitwise, sanitizer-clean, 3.50x local, TOTAL +1.805% |
| `State_To_FV` | CUDA conditional (`device_count>=2`), else Fortran | CONDITIONAL DEFAULT | 1-GPU/6-rank density memory-unsafe; validated bitwise at 2-GPU density |
| `FV_To_State` | Fortran (CPU) | INTENTIONALLY KEPT CPU | Already host-resident at both ends — no transfer to amortize |
| TRACER_2D (hord==8 path) | Fortran default | STOPPED — do not reopen | GPU-validated kernel but regression-confirmed (+37.87%); moot in production since production uses `hord_mt/vt/tm=6`, which never takes this path |
| REMAP_TOTALENERGYCON | No port taken | BLOCKED | `mpp_global_sum` (communication) is ~89% of the timer; not a kernel problem |
| `MOIST_CAPPA`/`USE_COND` | Unimplemented | NOT STARTED by design | Needs Riemann-solver `peg`/`gm2`/`cp2` terms, triple-gated rejection, not a defect |
| `a_imp <= 0.999` | Unimplemented | NOT STARTED by design | Only SIM1 (`a_imp=1.0`) implemented |
| Sub-tile (>6-rank) CUDA halo topology | Design + synthetic validation only | EXPERIMENTAL, gate-closed | Topology/transport-peer-identity layer GPU-validated at real 24-rank MPI scale; the production `comm_size==6` gate is intentionally still closed — see §9 |

**[LESSON]** A single compact backend-status table like this, kept in the
living handoff doc, is far more useful than prose — it is the first thing a
new agent or engineer should be able to find.

---

## 3. Modernization Workflow That Worked

**[LESSON]** The repeatable, validated sequence for porting one numerical
subsystem from Fortran to CUDA, observed across PT/DELZ, REMAP_UV, REMAP_Q,
REMAP_PRESSUREVARS, and `State_To_FV`:

1. Map the subsystem's architecture: owning module, call graph, inputs/
   outputs, data residency, existing timers.
2. Choose one bounded target subsystem — not a whole phase, not a whole file.
3. Establish an authoritative Fortran baseline/savepoint for that exact
   target before writing any CUDA.
4. Build a minimal host-level (CPU C++) harness/fixture that exercises the
   target in isolation, independent of the full model.
5. Implement the host C++ reference path first; validate it bit-for-bit (or
   to an explicit, justified tolerance) against the Fortran baseline.
6. Only then implement the CUDA kernel version of the same path.
7. Validate CUDA-vs-host-C++ numerical equivalence in isolation (unit/
   fixture level), before touching the integrated model.
8. Integrate via the existing stable ABI boundary — do not create a new,
   parallel integration mechanism per subsystem.
9. Validate CUDA kernels again once integrated (not just in isolation) —
   integration can introduce divergence a fixture test cannot see.
10. Validate the integrated model at increasing scale/duration (small grid →
    production grid, short run → longer run).
11. Profile the real production workload with the model's own timers —
    not a microbenchmark — to measure the subsystem's actual contribution.
12. Decide KEEP (CUDA becomes default) / STOP (document why, keep Fortran
    default) / DEFER (real blocker identified, not abandoned) using the
    decision framework in §6.
13. Document the evidence (what was measured, how, what the numbers were)
    in a dedicated handoff file — not just a verdict.
14. Select the next target using the current production hotspot profile,
    not an a-priori list.

**[LESSON — STOP conditions baked into this sequence]**
- Stop before step 6 (write CUDA) if step 5's host validation does not pass.
- Stop before step 10 if step 7's isolated kernel validation does not pass.
- Stop before step 12's KEEP decision if production profiling (step 11)
  shows the subsystem is not actually a measured bottleneck, or if the
  measured speedup does not clear the project's acceptance bar (this
  project required ≥5x local speedup to accept TRACER_2D's hord==8 path;
  it measured a regression and was correctly stopped).
- Never skip steps 3-4 to "save time" by writing CUDA directly against the
  integrated model — the TRACER_2D path's extensive debugging in Prompts
  73-83 is exactly what happens when integration-level numerics must be
  reverse-engineered after the fact rather than validated incrementally.

**[FACT]** Correctness came before performance throughout: Gate C2 (full
numerical validation of all 54 dyn_core stages) was closed before any
hotspot-driven CUDA porting (PT/DELZ onward) began.

---

## 4. Validation Hierarchy

**[FACT]** The project enforces four strictly ordered, never-conflated
levels, defined in `docs/agent_instructions/project_architecture.md`:

1. **IMPLEMENTED** — source exists, stage is wired into the call graph.
2. **COMPILED** — host/CUDA compiler accepts and links it.
3. **GPU EXECUTED** — it actually launches and runs on a real GPU.
4. **NUMERICALLY VALIDATED** — compared against authoritative Fortran
   reference data at an accepted tolerance (bitwise, round-off, or an
   explicitly justified ULP/relative threshold).

**[LESSON — the central cautionary tale]** `docs/agent_handoff/
implementation_reality_audit.md` (2026-08-21) found that an earlier
handoff had declared "Production first-run numerical path: COMPLETE, all 54
stages implemented, no PARTIAL/UNIMPLEMENTED stages remain" — while that
same document's own per-stage table said `CUDA=NOT RUN, GPU=NOT RUN,
Savepoint=NOT RUN` on every row, and the actual production entry point
(`fv3_gpu_dyn_core_step`) was a one-line stub that unconditionally returned
`FV3_GPU_UNSUPPORTED_CONFIGURATION` without calling any kernel. ~76 real,
numerically-faithful `__global__` CUDA kernels existed across WP-A/C/D but
were entirely orphaned — no caller reached them from the one function
GEOS/Fortran would actually invoke. WP-B had zero CUDA kernels at all (host
C++ only), contrary to the "COMPLETE" claim.

Concretely, **"host C++ reference implementation exists and passes a CPU
unit test" is not the same claim as "CUDA kernel is reachable from the
production entry point and runs on a GPU."** The conflation happened in
summary prose, not in the underlying tables — the tables were honest; the
narrative built on top of them was not.

**[LESSON]** "GPU EXECUTED" does not imply "numerically validated" — the
riem_solver3 kernel in this audit was a genuine, correct `__global__` kernel,
GPU-executable in principle, with zero callers; being executable is not the
same as being exercised by the real workload. Likewise "successful run" does
not imply "correct halo exchange" — a model can run to completion with wrong
boundary data if the halo exchange silently drops or mis-routes data; halo
correctness needs its own explicit bitwise check against a known-good
reference path (this project's cross-check: `MpiHaloTransport` vs.
`PeerHaloTransport`, both producing bitwise-identical results).

**[LESSON]** For structurally-zero production variables, you cannot validate
with the production configuration alone — a documented structural gap in
this project (`docs/agent_instructions/project_architecture.md` /
`CURRENT_STATE.md` §7) is that with `COLDSTART=1` and empty `ExtData.rc`,
tracers (`Atm%q`) stay exactly zero for the entire run, so no production run
can ever validate tracer-transport numerics end-to-end. A synthetic/
non-production test harness that seeds nonzero values is required to
validate such a path at all — relying on "the production run didn't crash"
proves nothing about a field that is identically zero by construction.

**[LESSON]** A timing result is never a correctness result, and vice versa.
The two are measured and reported independently throughout this project's
handoffs (see §5 for how timing was done; §11 for how correctness was done).

**[RECOMMENDATION]** Require every milestone handoff to state its validation
level explicitly, per component, using exactly these four words. Treat any
summary sentence that upgrades a level without new evidence as a defect in
the handoff, not a status update — this is precisely the failure the
2026-08-21 audit caught and fixed.

---

## 5. Performance Methodology

**[LESSON]** Production timer hierarchy, not microbenchmarks, drove every
porting decision. The model's own `fv_timing_mod` `DYN_CORE`/`TOTAL` timers
(and later MAPL's own timers for non-dycore residuals like `DYN_EPILOGUE`)
were the ground truth — never an isolated kernel microbenchmark used to
extrapolate end-to-end impact.

**[LESSON — hotspot-first, not list-first]** After Gate C2 closed, the next
porting target was chosen by profiling the *current* production binary with
`nsys` and picking the largest remaining timed hotspot, then re-profiling
after each port to find the next one — not from a fixed roadmap written in
advance. This produced the actual sequence: PT/DELZ → REMAP_UV → TRACER_2D
(stopped) → REMAP_Q → REMAP_PRESSUREVARS → REMAP_TOTALENERGYCON (blocked) →
`FV_To_State` (kept CPU) → `State_To_FV`.

**[LESSON — direct kernel timing vs TOTAL]** A local kernel speedup (e.g.
REMAP_UV's bitwise-exact ~10.23x local speedup) does not necessarily move the
end-to-end `TOTAL` timer by a comparable fraction — REMAP_UV's integrated
effect was only +~1% TOTAL. Report both numbers; never substitute one for
the other. A large local speedup on a small-contribution stage is a correct,
useful optimization that simply won't show up much in the headline number —
that is not a failure, but it must not be reported as if it were a large
end-to-end win either.

**[LESSON — paired, same-allocation A/B is mandatory at scale]** The single
most expensive mistake avoided in this project's history was treating a
cross-node comparison as a real performance result. The original C512
"CUDA 1.270x slower" finding (Prompt 37) triggered a multi-week root-cause
chain (Prompts 38-47) before Prompt 48 showed the real cause: Fortran and
CUDA had been measured on two different physical A100 nodes with a >2x
node-to-node variance, and a same-node, same-allocation, alternating-order
paired rerun (3 Fortran reps and 3 CUDA reps interleaved: F1 C1 F2 C2 F3 C3,
job 58468244) gave a ratio of medians of 0.99998 — truly comparable
performance, not a regression. **Always pair A and B in the identical Slurm
allocation**, alternate run order, and use the median of several repeats,
not a single run.

**[LESSON — memory headroom is a first-class performance constraint, not an
afterthought]** `State_To_FV`'s CUDA port was correct and GPU-executed at
small scale but was blocked at full C512/6-ranks-per-GPU production density
by a ~440 MiB per-rank shortfall — discovered only via a dedicated memory
inventory (Prompt 101), not via a timing run (which would have simply OOM'd
or silently fallen back). The eventual fix was architectural (spread ranks
across more GPUs, §7), not a code shrink. Treat GPU memory headroom as a
measured, tracked quantity throughout a porting effort, the same as a timer.

**[RECOMMENDATION — illustrative, not prescriptive]** The project's specific
ports (PT/DELZ, REMAP_UV, REMAP_Q, REMAP_PRESSUREVARS, `State_To_FV`,
TRACER_2D, REMAP_TOTALENERGYCON) are FV3-specific examples of this
methodology, not a template of "which subsystems to port" for a different
model — the methodology (hotspot-first, paired A/B, memory-aware, local-vs-
TOTAL distinction) is the reusable part.

---

## 6. Decision Framework: PORT / KEEP CPU / DEFER / STOP

| Decision | Criteria (evidence-based) | Project example |
|---|---|---|
| **PORT (make CUDA default)** | Local kernel validated bitwise/round-off vs Fortran; GPU-executed at production scale; measurable non-negative TOTAL contribution; memory headroom confirmed at target rank density | REMAP_UV (bitwise, +~1% TOTAL), REMAP_PRESSUREVARS (bitwise, +1.805% TOTAL) |
| **PORT conditionally** | Same bar as PORT, but only safe/beneficial above a specific threshold (device count, rank density, etc.) — gate the default on that threshold, don't force it everywhere | `State_To_FV`: CUDA default only at `device_count>=2`, Fortran default otherwise |
| **KEEP CPU (do not port)** | Both backend's data already lives on the host at the call boundary — a port would add a transfer, not remove one | `FV_To_State` |
| **DEFER (real blocker, not abandoned)** | A specific, named blocker exists (communication-bound, missing numerical term, platform permission) that a kernel rewrite cannot fix | REMAP_TOTALENERGYCON (`mpp_global_sum` is ~89% of the timer — a communication problem, not a compute problem); `MOIST_CAPPA`/`a_imp<=0.999` (missing numerical terms, not yet attempted) |
| **STOP (reject, do not reopen without new evidence)** | Measured regression against the required acceptance bar, confirmed by at least one independent reproduction | TRACER_2D hord==8 CUDA path: GPU-validated correct, but +37.87% regression against a required ≥5x speedup gate; additionally moot because production's `hord_mt/vt/tm=6` config never reaches that code path at all |

**[LESSON]** A STOP decision should record both *why it failed the bar* and
*whether it is even reachable in the real production configuration* — the
TRACER_2D case had both: it failed on performance, and it was additionally
irrelevant to the only configuration that matters (`hord==6`, not `hord==8`).
Either fact alone would justify stopping; recording both prevents a future
agent from "fixing the performance" and still shipping something unused.

---

## 7. GPU Memory Lessons

**[LESSON — per-rank vs per-GPU]** The dominant finding (Prompt 102): GPU
memory consumption at fixed rank count scales overwhelmingly **per-rank/
per-CUDA-context**, not fixed-per-GPU. This means the primary lever for
recovering memory headroom is **spreading ranks across more GPUs**, not
shrinking any single port's footprint. Rank-density changes (1/2/3 GPUs at
fixed total rank count) were separately shown to have **no effect on
timing** (Prompt 47) — memory and timing are decoupled levers; don't assume
a memory fix will also move performance, or vice versa.

**[FACT]** Concretely: C512 at 6 ranks/1 GPU left `State_To_FV`'s CUDA port
~440 MiB short per rank; the same workload at 2 GPUs/3-ranks-per-GPU
validated at 19.68 GiB/GPU with memory to spare, bitwise-identical results,
and a modest TOTAL improvement (1.120x) as a side effect of being
unblocked — not because 2-GPU is "faster" per se.

**[LESSON — persistent allocations compound]** This project's
persistent-GPU-residency architecture (§8 of `project_architecture.md`:
state stays device-resident across WP-A→B→C→D without round-tripping to
host) is a performance win but means every persistent buffer is a standing
memory cost for the whole run, not an amortized one-shot cost — inventory
them explicitly before assuming a new port "will probably fit."

**[LESSON — MPS/ranks-per-GPU interacts with both memory and contention]**
CUDA MPS (Multi-Process Service) was required to let multiple MPI ranks
share one GPU context efficiently; `fv3_mps_lifecycle.sh` wraps
start/stop around a job. MPS requires `compute_mode=Default` (confirmed
true on this platform) — check this before assuming MPS is available on a
different cluster.

**[LESSON — reconciling a memory inventory]** Memory accounting was done by
explicit static inventory (sum of named persistent allocations) cross-
checked against a runtime measurement, not by only reading `nvidia-smi` once
after the fact.

**[LESSON — a post-hoc GPU memory read is misleading]** "A post-hoc
`nvidia-smi` checkpoint after a job exits reads near-idle, not the in-run
footprint" (the CUDA context is already destroyed by the time you look) —
use a concurrent background poller sampling throughout the run, never a
single checkpoint taken after the job completes.

**[LESSON]** "A single unsynchronized `cudaMemGetInfo`/`nvidia-smi` reading
across MPS-shared MPI ranks cannot cleanly attribute memory to a lifecycle
stage" — without an `MPI_Barrier` per checkpoint, you cannot tell which
rank's allocation you are actually seeing at a given instant; only a
converged, steady-state end ceiling is trustworthy without barriers.

**[LESSON — aliasing/shrink optimizations are not always worth it]** Once
the architectural fix (more GPUs) resolved the memory blocker, no further
effort went into shrinking individual buffers via aliasing — the project
treated "spread ranks across more GPUs" as cheaper (in engineering time) and
more robust (headroom scales with hardware, not with each new port's
footprint) than chasing per-buffer savings. Prefer the structural fix when
one is available before micro-optimizing allocations.

---

## 8. MPI / Multi-GPU Lessons

**[LESSON — host-staged MPI was sufficient; CUDA-aware MPI was not useful
here]** The staged communication plan (Stage 1 host-staged MPI → Stage 2
CUDA-aware MPI → Stage 3 overlap) was followed in order, but Stage 2's
payoff did not materialize: CUDA-aware `MpiHaloTransport` (Intel MPI 2021.13
+ `I_MPI_OFFLOAD=1`) was validated correct and bitwise-identical to
host-staged transport at a memory-safe density, but gave only a ~0.03%
`DYN_CORE` improvement. **Host-staged remains the default.** The lesson is
not "CUDA-aware MPI never helps" — it is "measure before assuming the more
sophisticated transport is worth its added complexity," and this project's
numerical APIs were deliberately designed so that moving between stages
required no redesign (`project_architecture.md` §"Communication Evolution"),
which is what made it cheap to actually test Stage 2 and then decline it.

**[LESSON — global-rank-modulo device selection is an anti-pattern across
nodes]** Early multi-GPU device selection used `dev = mod(comm_rank,
device_count)` — correct only within a single node. Node-local GPU binding
via `MPI_Comm_split_type` (splitting the communicator by shared memory/node)
was required and validated for genuine multi-node correctness (Prompt 107),
confirmed bit-identical single-node and correct across a genuine 2-node
test. **Rule: GPU device selection must use the node-local rank, never the
global MPI rank, the moment more than one node is in play.**

**[LESSON — `CUDA_VISIBLE_DEVICES` and job-wide vs per-node resource caps]**
This platform's QOS (`alla100`) caps a job's **summed, job-wide** GPU/memory
budget (`MaxTRES`), not a per-node budget — a multi-node job's total request
must be budgeted against the whole-job cap, not computed per-node and
multiplied. This is platform-specific, but the category of mistake
("resource caps may be job-wide, not per-node — check before assuming") is
reusable.

**[LESSON — MPS lifecycle across nodes needs an explicit broadcast]** The
MPS startup/shutdown script was head-node-only by default; genuine
multi-node MPS requires an explicit `srun --overlap --nodes=N
--ntasks-per-node=1` broadcast of the lifecycle script to every node, not
just the node that launched the job.

**[LESSON]** Same-node vs. cross-node timing comparisons must never be
treated as equivalent — see §5's node-to-node A100 variance (>2x at C512)
finding; this is the single most repeated root cause of apparent
"regressions" that were really measurement artifacts in this project.

---

## 9. Halo / Domain-Decomposition Lessons

**[FACT]** The production CUDA bridge is hardcoded to `comm_size==6` (one
MPI rank per cubed-sphere face/tile). Scaling resolution further (C1024+)
needs more than 6 ranks, i.e. more than one rank per face — a sub-tile
decomposition. This was investigated (Prompts 108-116) but **deliberately
not enabled in production** — all four `comm_size==6`-style gates remain
closed by design; only the groundwork was built and validated at a synthetic
scale.

**[LESSON — verify the underlying system already supports your target
decomposition before reimplementing communication]** The single most
important finding in this chain: FMS (the Fortran communication/domain
library underneath GEOS/FV3) **already supports** `layout_x × layout_y`
sub-tile decomposition (`Atm%layout(1:2)`) — this did not need to be
invented. The CUDA bridge had simply never been told about it (it assumed
one-rank-per-face from the start). The correct move was to read FMS's
existing layout metadata and thread it through the ABI (`layout_x`/
`layout_y`, Prompt 113-114), not to build a parallel decomposition
mechanism inside the CUDA runtime. **Before building new communication
infrastructure for a legacy Fortran model, check whether the underlying
Fortran/MPI library you're bridging to already has the decomposition you
need — you may only need to plumb existing metadata through, not invent
anything.**

**[LESSON — separate physical topology identity from transport identity]**
A real blocking bug (Prompt 111→112): `launch_halo_transfer` indexed its
peer-send table directly by `.tile` (a physical cubed-sphere face number,
fixed at 6 possible values), conflating "which face am I" with "which MPI
rank do I talk to." Under a sub-tile layout, multiple ranks share one face,
so face identity and transport-peer identity are no longer interchangeable.
The fix was to add a distinct `.rank` field for transport identity, keeping
`.tile` purely as physical-face identity. **Generalizing a six-tile-assumed
communication layer to N-ranks-per-tile requires explicitly separating
"which physical neighbor is this" (topology) from "which rank do I send to"
(transport identity) — these were silently the same value in the
one-rank-per-tile case, which is exactly what hid the bug for so long.**

**[LESSON — verify runtime rank ordering empirically, never assume it]**
The sub-tile rank-to-`(sub_i, sub_j)` mapping was *assumed* (x-fastest/
row-major) in Prompts 113-114 but explicitly flagged UNVERIFIED, and was
not used in any production path until Prompt 115 verified it empirically: a
Fortran-only, zero-CUDA, 24-rank `layout=(2,2)` diagnostic printed each
rank's actual FMS-assigned compute-domain window and derived `(sub_i,
sub_j)` independently from each axis (avoiding x/y ambiguity by
construction), confirming `local_rank_within_tile = sub_j*layout_x + sub_i`
exactly, zero exceptions, across 6 tiles and 2 rank counts. **Prove rank
ordering with a runtime diagnostic against the real decomposition library
before building derivation logic on top of an assumption — "it's probably
row-major" is not evidence.**

**[LESSON — synthetic routing validation is necessary but is a separate
claim from real-traffic validation]** The topology/donor-identification
layer and the generalized transport-peer-identity fix were GPU-validated at
a real 24-rank MPI scale with 0/22224 failures and 0 sanitizer errors —
genuinely strong evidence that the *topology graph and routing logic* are
correct. This is explicitly a narrower claim than "the production bridge
supports >6 ranks" — the production `comm_size==6` gates were never relaxed,
so this validates the mechanism in isolation, not an enabled production
capability. Keep these two claims distinct in any report.

**[LESSON — stage bridge metadata plumbing separately from behavior
changes]** `layout_x`/`layout_y` were added to the ABI as purely additive,
trailing fields with safe defaults, stored and cross-checked (via an
independent `MPI_Allgather`), but **used nowhere** in production topology
selection — this was deliberately staged as "wire the data through, prove it
arrives correctly, change nothing about what runs" before any topology-
builder switch was even considered. This kept each step's blast radius small
and independently revertible.

**[LESSON]** Corner transport (the 3-rank diagonal junctions of a
cubed-sphere) was explicitly deferred multiple times and remains
metadata-only — no message mechanism — separately gated from the edge-halo
work above. Don't assume "halo exchange" implicitly includes corners; call
out corner handling as its own tracked item.

---

## 10. ABI / Hybrid-Language Integration Lessons

**[FACT]** The Fortran↔C++/CUDA boundary is `ISO_C_BINDING` → a stable C ABI
(`fv3_gpu_api.h`, `fv3_gpu_communication.h`) → the WP-E C++ runtime. No C++
or MPI type crosses the ABI directly — the MPI communicator crosses as a
plain Fortran `INTEGER`, converted via `MPI_Comm_f2c` on the C++ side.

**[LESSON — explicit-width types at the boundary]** All ABI struct fields
use fixed-width `iso_c_binding` types (e.g. `c_int32_t`), not default
Fortran/C integer/real kinds, to avoid any platform- or compiler-dependent
width mismatch across the Fortran/C++ boundary.

**[LESSON — additive-only ABI evolution]** Every ABI change observed in this
project's history (adding `layout_x`/`layout_y`, adding a `device_count`
getter, etc.) was a **trailing, additive field or trailing, additive
argument**, never a reordering or removal. Existing callers (including a
zero-initialized struct literal in an existing test) continue to work
unchanged, with new fields normalized to a safe default (e.g. `<=0` defaults
to `1`) when absent. A documented "purely additive, default-valued field"
convention meant some fields could even be added to an internal struct
without a version bump.

**[LESSON — runtime descriptors guard ABI drift]** The ABI carries its own
`abi_version`/`struct_bytes` fields, checked at runtime, so a
compiler-family or build mismatch between the Fortran caller and the C++
library fails fast at initialization rather than silently corrupting memory
layout later. This was explicitly used as the argument for why Intel-Fortran
production ABI compatibility was "sound by construction" without needing a
live Intel-compiler GPU test.

**[LESSON — fail-fast initialization, not silent fallback]** Multiple
independent `MPI_Allgather`-based cross-checks were added at bridge-init
time (e.g., every rank reporting the same `layout_x/layout_y`) that FATAL
immediately on mismatch, rather than letting an inconsistent configuration
run and produce silently wrong results later.

**[LESSON — shadow-mode validation before switching production behavior]**
The recommended (and in earlier similar steps, actually-used) pattern for
introducing a new derivation or topology mechanism: compute it alongside
the existing, still-authoritative mechanism, compare the two, FATAL on
mismatch in a debug/test-gated mode — and only switch which one actually
drives production behavior after that shadow comparison has run clean. This
lets you validate a new code path's correctness under real traffic without
betting production correctness on it yet.

**[LESSON — preserve legacy behavior as the unconditional default]** Every
new ABI/runtime capability in this project was introduced behind an
opt-in, default-OFF flag (env var or build flag), with the explicit
requirement that default (flag-unset) behavior be provably unchanged "by
construction," not just "by observation" — i.e., the new code path should be
literally unreachable when the flag is unset, not merely untested.

---

## 11. Numerical-Fidelity Lessons

**[FACT]** The project distinguishes, and reports separately: bitwise
equality, round-off-level agreement (differences only at the level of
floating-point reassociation), ULP-level transcendental differences, and
genuine algorithmic divergence.

**[LESSON — decomposition-dependent floating-point ordering is expected and
benign, not a bug]** R8 Gate C2 validation showed worst-case `w` max_rel
~5e-11 — not exactly zero — attributable to floating-point operation
reordering between the Fortran and CUDA implementations (different
summation/loop order), not an algorithmic difference. This is the expected
signature of "same algorithm, different evaluation order," and was
correctly accepted as round-off-only.

**[LESSON — distinguish an explainable artifact from a real divergence
before accepting a large-looking relative error]** R4 validation showed a
worst-case `w` max_rel of ~2.4e-1 — a seemingly large relative error — but
investigation showed it was a small-denominator artifact (the relative-error
formula blows up when the reference value is near zero) with absolute
agreement ≤5.9e-5. The project's rule: a large relative error must be traced
to its cause before being accepted or rejected; do not gate purely on a
relative-error threshold without checking the absolute magnitude when the
reference is small.

**[LESSON — exact-sum / global-reduction operations are a distinct
numerical category]** `mpp_global_sum`-based conservation diagnostics
(REMAP_TOTALENERGYCON) are dominated by communication cost and often rely on
reproducible-summation-order guarantees that a naive CUDA reduction would
not automatically preserve — this is part of why that port was deferred
rather than attempted casually.

**[LESSON — known benign artifacts should be named and tracked, not
re-discovered]** Three specific benign, pre-existing-in-both-backends
artifacts are explicitly documented so they are never mistaken for new CUDA
defects: a subnormal-magnitude halo-padding uninitialized-memory difference
in `delpc`/`pkc` (savepoint-level only, never visible at output level); a
benign FMS plausibility-bound `range_warn` at C128 10-day (identical count
both backends); a near-surface NaN in the lowest few of many vertical levels
from standard MAPL vertical-interpolation masking (identical both backends).
**Maintain a short "known benign, do not re-investigate" list in the living
status doc — rediscovering the same non-bug costs real time otherwise.**

**[LESSON]** "Round-off-clean" is acceptable evidence of correctness only
when the comparison methodology is bitwise/field-level at meaningful
intermediate boundaries, not only at final aggregate output — this project's
validation explicitly compared at dyn_core entry/exit and multiple
in-loop savepoints, not just end-of-run NetCDF fields, specifically so a
divergence could be localized to the first mismatching stage rather than
only detected in the aggregate.

---

## 12. Debugging / Operational Lessons

**[LESSON — preflight checklist before trusting any job's output]**

- [ ] Grep the job's own log for its "`***** USING ****` &lt;path&gt;" (or
  equivalent) binary-identity line — a stale `install/bin/` copy silently
  running instead of your freshly built binary is a repeated real failure
  mode here (Prompt 53). Confirm via `strings <bin> | grep <new marker>`
  that your new flag/symbol is actually present in the binary that ran.
- [ ] Grep any copied/reused sbatch/job script for `setenv EXPDIR` (or your
  project's equivalent run-directory variable) before submitting — a
  stale copy-pasted path silently breaks log capture, and under
  concurrency with another job pointed at the same stale directory this
  caused a real `MPI_Abort`/crash, not just lost logs (Prompt 107).
- [ ] Check the run's actual execution subdirectory (e.g. a `scratch_*/`
  working dir), not just the top-level run directory, for history/output
  files — this project's MAPL History writer never writes at the top
  level; one agent concluded "no output was produced" by checking only
  the wrong directory, on a run that had actually succeeded.
- [ ] Confirm which module/compiler/MPI stack a job actually loaded before
  generalizing any performance or correctness finding from it — this
  project runs two materially different validated stacks (an OpenMPI dev
  stack for CUDA-only bring-up, and an Intel-MPI stack for every
  production-scale job); conflating them invalidates comparisons.
- [ ] Before trusting any env-var-gated debug/counter mechanism's output,
  grep the source for the marker string it's supposed to print — one
  prompt computed a meaningless 0/0 from a dispatch counter that had never
  actually been implemented.
- [ ] Treat a post-hoc GPU memory/process snapshot as unreliable (see §7);
  require a concurrent/in-run measurement for any real memory claim.
- [ ] Before any destructive git operation, re-run `git status` and
  `git log` fresh — do not act on a stale mental model of the tree.

**[LESSON — environment/module ordering matters and is non-obvious]** This
platform requires `module purge` *then* a `module use -a <path>` to extend
the module search path, in that order — `module purge` clears path
extensions, so reversing the order silently loses the custom path. Any
platform with a nonstandard module tree should have its exact
purge/use/load order written down verbatim, not reconstructed from memory
each session.

**[LESSON — missing build infrastructure can masquerade as a code bug]**
`build_cuda_r8_intelmpi/`'s set of prebuilt CUDA static archives did not
exist in one session's checkout; reconstructing it required a specific
non-default CMake flag, an exact `nvcc` path (the bare module-resolved
`nvcc` was a different, incomplete install missing two required static
libs), and the full production MPI module stack loaded so a compiler-wrapper
substitution token would resolve. None of this was a source-code defect —
treat "the build directory doesn't exist" and "the code doesn't compile" as
distinct failure categories before debugging the wrong one.

**[LESSON]** Resource/QOS constraints (job-wide vs per-node caps, §8) are a
platform fact to look up, not infer from a failed job's error message alone.

---

## 13. What Did NOT Work / What Was Stopped

**[FACT — stopped, with reason, do not reopen without new evidence]**

- **TRACER_2D CUDA path (hord==8)**: GPU-validated correct, but measured a
  +37.87% regression against the required ≥5x local-speedup acceptance bar;
  additionally moot in production because `hord_mt/vt/tm=6` (the only
  supported production config) never takes the hord==8 code path at all.
  Stopped on both performance and relevance grounds.
- **CUDA-aware MPI as a C512 performance lever**: correct, bitwise-validated,
  environment-viable — but delivered only ~0.03% `DYN_CORE` improvement.
  Not stopped because it was wrong, but because it did not clear a
  usefulness bar; retained as an optional non-default transport rather than
  deleted, since it cost little to keep and might matter at a different
  scale/platform later.
- **Premature C1024/sub-tile assumptions**: an initial unverified assumption
  about FMS's within-tile rank ordering was *not* promoted into production
  topology-builder logic — it was explicitly held back pending empirical
  verification (Prompt 115), and even after verification, the production
  >6-rank gates were still not relaxed (§9). The project repeatedly chose
  "design/verify, then stop" over "assume and wire it in."
- **Assuming physical-face identity and transport-rank identity were
  interchangeable**: this *did* get built into the halo transport layer
  initially (indexing by `.tile`), and was a real bug once a sub-tile layout
  was attempted — not caught until Prompt 111/112. Recorded here as a
  concrete example of an assumption that *did* make it into code before
  being caught, contrasted with the ones above that were caught at the
  design stage.
- **Treating a cross-node timing comparison as a real result**: the original
  C512 "CUDA 1.270x slower" conclusion (§5, §8) was wrong and consumed
  significant investigation (Prompts 38-47) before being correctly
  superseded by a same-node paired rerun.
- **Over-trusting an early "COMPLETE" claim without re-deriving from source**
  (§4): a documented, explicit audit was required to discover that an
  "IMPLEMENTED COMPLETE" claim rested on stub/orphaned code with zero
  production reachability.

**[LESSON]** None of these were hidden after discovery — each has a
dedicated handoff document stating what was tried, what was measured, and
why it was stopped. A reusable project norm: a stopped/rejected approach gets
exactly as much documentation rigor as a successful one.

---

## 14. Cost-Efficient Agent Workflow

**[FACT]** `docs/agent_handoff/cost_efficiency.md` (and `CLAUDE.md`) define
an explicit cost policy, condensed here:

- Read `CLAUDE.md` + `CURRENT_STATE.md` + task-named files at the start of a
  task; do not reread historical handoffs unless `CURRENT_STATE.md` points
  to one, the current defect originated there, or exact historical evidence
  is required.
- Prefer targeted inspection (exact files/symbols, `git grep`, narrow
  `grep`/`find`) over broad repository scans; expand only when targeted
  inspection is insufficient.
- Reuse existing harnesses/test frameworks/savepoint formats/build scripts
  before building new ones.
- Every debugging task needs an explicit hypothesis, a minimum experiment
  set, and a stop condition; stop once the hypothesis is resolved — don't
  keep gathering evidence that won't change the decision.
- For numerical divergence, localize coarse-to-fine: compare major stage
  boundaries first, find the first mismatching stage, then narrow — never
  dump every field at every stage up front.
- Prefer compact machine-readable diagnostics (CSV/JSON/hashes/norm
  summaries) over large free-form logs fed back into context.
- Keep routine progress terse — report material findings, blockers, plan
  changes, and final results; do not narrate successful routine builds or
  already-resolved hypotheses.
- Match model/effort to task difficulty: cheaper/default models for
  mechanical work (build scripts, routine implementation, documentation);
  reserve the strongest reasoning for numerical reconciliation, halo/
  topology semantics, concurrency reasoning, and unexplained divergence.
  (This specific project's instructions name Sonnet as the default and
  escalate only for unresolved architecture/numerics — a reusable pattern
  is "cheap model for mechanical, strong model for a short analysis/plan,
  then back to cheap for execution," not "strong model for everything.")
- Finish the current acceptance gate before starting adjacent work; record
  unrelated findings as deferred tasks instead of chasing them immediately.
- Never infer GPU execution from compilation, or numerical validation from
  GPU execution (§4) — this is a cost-efficiency rule as much as a
  correctness rule, since re-deriving a conflated claim later (as in the
  2026-08-21 audit) costs far more than stating the correct level the first
  time.
- Do not rerun expensive GPU jobs after a context/session disconnect
  without first checking whether the needed evidence already exists in a
  completed job's logs (see §agent_recovery skill).
- No commit/push without being explicitly asked, every time — not just once.

---

## 15. Recommended Workflow for a New Legacy HPC Project

**[RECOMMENDATION]** A 13-phase template generalized from what actually
worked here. Each phase lists inputs, outputs, a PASS gate, and STOP
conditions.

**Phase 0 — Environment / Source-of-Truth Establishment**
- Inputs: target platform docs, legacy repo(s), module/toolchain
  requirements.
- Outputs: a written, verbatim module-load sequence; a confirmed GPU/queue
  access pattern; an identified authoritative numerical source repo/tag.
- PASS: toolchain versions verified by direct inspection (not assumed from
  module names); a trivial GPU smoke test runs through the job scheduler.
- STOP if: GPU access cannot be established through the scheduler at all —
  resolve this before any CUDA work.

**Phase 1 — Architecture Map**
- Inputs: legacy source tree, build system, existing docs.
- Outputs: call-graph/module-ownership map; data-residency map; identified
  MPI/communication boundaries; dependency classification (owned vs
  external vs vendored).
- PASS: can name, for any numerical routine, which module owns it and what
  calls it.
- STOP if: the call graph cannot be established without executing the code —
  escalate to a dynamic-tracing approach rather than guessing from names.

**Phase 2 — Baseline Validation Harness**
- Inputs: architecture map, a representative small test case.
- Outputs: an authoritative Fortran (or legacy-language) baseline/savepoint
  mechanism for the target subsystem.
- PASS: the baseline is reproducible (same inputs → same outputs) and
  independent of any new code.
- STOP if: the legacy code cannot produce a stable baseline (nondeterminism,
  uninitialized memory) — fix or characterize that first; it will otherwise
  poison every later comparison.

**Phase 3 — Bounded Target Selection**
- Inputs: architecture map, project goals.
- Outputs: one named, bounded numerical subsystem to port, with an explicit
  acceptance bar (tolerance, required speedup if performance is in scope).
- PASS: the target is small enough to validate in isolation.
- STOP if: the target cannot be isolated from the rest of the model without
  excessive scaffolding — shrink the target further.

**Phase 4 — Host-Language Translation (C/C++)**
- Inputs: target subsystem, baseline harness.
- Outputs: a host-language (CPU) reference implementation of the target.
- PASS: bit-for-bit or explicitly-tolerance-justified agreement with the
  legacy baseline, in isolation.
- STOP if: agreement cannot be reached — do not proceed to GPU work on top
  of an unvalidated host translation.

**Phase 5 — ABI Integration**
- Inputs: validated host translation.
- Outputs: a stable, explicit-width, additive-only C ABI boundary; a
  thin binding-language (e.g. ISO_C_BINDING) wrapper calling it.
- PASS: existing legacy callers are unaffected when the new path is
  disabled; ABI version/size is checked at init.
- STOP if: integration requires modifying legacy call sites beyond adding
  one guarded call/branch — reconsider the boundary's placement.

**Phase 6 — CUDA Kernel Development + Isolated Validation**
- Inputs: validated host translation, ABI boundary.
- Outputs: CUDA kernel(s) implementing the same algorithm; isolated
  unit/fixture validation against the host translation.
- PASS: GPU-executed and numerically validated in isolation, at the
  project's accepted tolerance.
- STOP if: isolated validation fails — do not integrate an unvalidated
  kernel "to see if it works at scale."

**Phase 7 — Production Integration**
- Inputs: isolated-validated kernel, ABI boundary.
- Outputs: the kernel reachable from the real production entry point,
  behind a default-off flag.
- PASS: re-validated numerically *after* integration (not just reusing the
  isolated-fixture result) — integration can introduce new divergence.
- STOP if: the production entry point does not actually invoke the new
  code path end-to-end — verify this explicitly (see §4's audit lesson),
  don't infer it from the presence of the kernel file.

**Phase 8 — Hotspot Profiling**
- Inputs: integrated, validated subsystem; production-scale workload.
- Outputs: a profile using the model's own timers identifying the next
  bottleneck.
- PASS: a specific next target is identified with a measured contribution,
  not a guess.
- STOP if: no further profiling can be done without first fixing a known
  measurement artifact (e.g. cross-node noise) — fix the measurement before
  trusting the profile.

**Phase 9 — GPU Residency / Memory Management**
- Inputs: integrated subsystem(s), target production scale.
- Outputs: a memory inventory (static allocations + runtime telemetry) at
  target rank/GPU density.
- PASS: headroom confirmed at the real production density, not just a
  smaller test grid.
- STOP if: memory is insufficient — resolve via the decision framework in
  §16's `gpu_memory_diagnostics` skill (spread across more GPUs, reduce
  persistent footprint, or reduce rank density) before declaring a port
  production-ready.

**Phase 10 — Multi-GPU / MPI Scaling**
- Inputs: validated single-GPU production path.
- Outputs: node-local GPU binding; verified behavior across node counts.
- PASS: bit-identical results at every tested GPU/rank density; node-local
  (not global) rank used for device selection.
- STOP if: multi-node behavior cannot be validated identically to
  single-node — do not ship multi-node as "probably fine."

**Phase 11 — Domain-Decomposition Scaling**
- Inputs: validated multi-GPU path; a legacy decomposition library (if one
  exists, e.g. FMS) that may already support the target decomposition.
- Outputs: verified (not assumed) rank-ordering/topology metadata;
  topology/transport-identity separation if moving beyond a 1:1
  rank:physical-unit mapping.
- PASS: topology validated against real runtime metadata, at real scale,
  with a synthetic routing check before any production gate is relaxed.
- STOP if: the underlying decomposition's rank ordering is unverified —
  verify empirically (Phase 11a: a minimal, zero-GPU diagnostic in the
  legacy language) before writing any derivation logic on top of it.

**Phase 12 — Production Hardening**
- Inputs: all validated components.
- Outputs: default-policy decisions folded into standing job
  scripts/configs; operational cleanup (flags, lifecycle scripts) made the
  default rather than requiring manual opt-in each run.
- PASS: a fresh job, run with no special flags, gets the validated,
  production-recommended configuration.
- STOP if: a validated optimization still requires manual flags to enable —
  that is a tracked follow-up, not "done."

---

## 16. Current FV3 Status (concise)

**[FACT]** As of 2026-10-01 (full detail: `CURRENT_STATE.md`):

- Gate C2 (full 54-stage dyn_core numerical path) CLOSED — R8/R4 both
  NUMERICALLY VALIDATED at c8/c16/c32.
- Held-Suarez dry-core validated bitwise/round-off from C8 through C512,
  including a 100-day C128 climatology; C512 Fortran/CUDA performance
  COMPARABLE (ratio of medians 0.99998, same-node paired).
- Production CUDA defaults: PT/DELZ, REMAP_UV, REMAP_Q, REMAP_PRESSUREVARS;
  `State_To_FV` conditional at `device_count>=2`.
- Multi-GPU validated at C512/2-GPU/3-ranks-per-GPU (bitwise, 19.68 GiB/GPU,
  TOTAL +10.70%); node-local `MPI_Comm_split_type` GPU binding validated
  single- and multi-node.
- Generalized sub-tile edge-halo transport (physical-face vs transport-rank
  identity separated) GPU-validated at a real 24-rank MPI scale — but all
  four production `comm_size==6` gates remain intentionally closed; no real
  >6-rank production path exists yet.
- FMS layout metadata (`layout_x`/`layout_y`) flows additively through the
  ABI/bridge/runtime (wires only, no behavior change); FMS's actual
  within-tile rank ordering is now empirically VERIFIED
  (`local_rank_within_tile = sub_j*layout_x + sub_i`); the resulting
  `SubTileId` derivation formula is DESIGNED (not yet implemented) —
  C++-runtime location, no new ABI field required, no `pelist(1)`
  dependency, exact 1×1 reduction to current behavior proven.
- Next recommended step (`CURRENT_STATE.md` §9): implement the bounded
  `subtile_id_from_rank` helper + a 24-rank test-only oracle unit test + an
  env-gated 1×1 shadow-topology equivalence check — explicitly no gate
  relaxation, no real >6-rank run yet.
- Open gaps: `MOIST_CAPPA`/`USE_COND` and `a_imp<=0.999` unimplemented by
  design; warm restart (`COLDSTART:0`) confirmed broken (Fortran-backend gap,
  not CUDA); corner transport metadata-only; R4/non-HS production validation
  covers only c8/c16/c32; no multi-day runs yet at C256/C512.

---

## 17. Key Artifacts / Where to Look

| Purpose | File/path |
|---|---|
| Current compact status (start here every session) | `docs/agent_handoff/CURRENT_STATE.md` |
| Full historical narrative/metrics (read only when needed) | `docs/agent_handoff/milestone_summary_2026-09-19.md` |
| The "overclaimed completeness" cautionary audit | `docs/agent_handoff/implementation_reality_audit.md` |
| Cost/token policy | `docs/agent_handoff/cost_efficiency.md`, `CLAUDE.md` |
| Platform/Slurm/module instructions | `docs/agent_instructions/discover_platform.md` |
| Architecture/work-package ownership/validation terminology | `docs/agent_instructions/project_architecture.md` |
| Chronological hotspot-porting roadmap | `docs/performance/cuda_fv3_optimization_roadmap_2026-09-21.md` |
| Per-port detailed evidence (PT/DELZ, REMAP_UV, REMAP_Q, REMAP_PRESSUREVARS, State_To_FV, etc.) | `docs/performance/p49_*.md` through `p107_*.md` |
| Sub-tile halo architecture chain | `docs/performance/p108_*.md` through `p116_*.md` |
| Public C ABI | `cuda/dyn_core/include/fv3_gpu_api.h`, `cuda/dyn_core/include/fv3_gpu_communication.h` |
| Fortran bridge (production call site) | `cuda/interface/fv3_gpu_dyncore_bridge.F90`, `fv3_gpu_abi_mod.F90`, `fv3_gpu_field_registry.F90` |
| Topology/sub-tile headers | `cuda/dyn_core/include/fv3_gpu_halo_subtile_topology.hpp`, `fv3_gpu_halo_subtile_view.hpp` |
| Halo transport implementations | `cuda/dyn_core/runtime/fv3_gpu_halo_mpi.{hpp,cpp}`, `fv3_gpu_halo_exchange.cu` |
| Held-Suarez forcing | `Held_Suarez_ForcingMod.F90`, call site in `FV_StateMod.F90::FV_Run` |
| Discover platform specifics (modules, Slurm, Nsight tools) | `docs/agent_instructions/discover_platform.md` |

---

## 18. Reusable Principles

1. Validate numerics before optimizing performance — always, no exceptions.
2. Measure the production timer, not intuition and not a microbenchmark.
3. A local kernel speedup and an end-to-end TOTAL improvement are different
   numbers; report both, never substitute one for the other.
4. Pair A/B performance comparisons in the same allocation, same node,
   alternating order, multiple repeats — cross-node/cross-job comparisons at
   scale are not trustworthy on multi-tenant HPC hardware.
5. Treat GPU memory headroom as a first-class, continuously tracked
   constraint, with the same rigor as a timer — not something you discover
   via an OOM.
6. Keep host/GPU authority clear: know, for every field, which backend owns
   the authoritative copy at every point in the pipeline.
7. Separate physical topology identity from transport/communication
   identity the moment a 1:1 rank-to-physical-unit assumption might break.
8. Before building new decomposition/communication infrastructure, check
   whether the underlying legacy library already supports what you need —
   you may only need to plumb metadata through.
9. Prove runtime rank ordering and other decomposition assumptions
   empirically, with a minimal diagnostic in the legacy language, before
   writing derivation logic that depends on them.
10. Never conflate IMPLEMENTED / COMPILED / GPU EXECUTED / NUMERICALLY
    VALIDATED — a documented audit in this project exists specifically
    because that conflation happened once and had to be unwound.
11. One bounded change per validation cycle; stage ABI/metadata plumbing
    separately from any behavior change, and validate each stage
    independently.
12. Every stopped/rejected approach deserves the same documentation rigor as
    a successful one — record what was tried, what was measured, and why it
    was stopped, so it is never redone from scratch.
13. All ABI evolution should be additive-only (trailing fields/args, safe
    defaults, version/size checks at init) — this is what makes staged,
    low-risk integration possible at all.
14. Default behavior must be provably unchanged "by construction" (the new
    code path is unreachable when its flag is unset), not just "by
    observation" (it happened not to trigger in the tests you ran).
15. A post-hoc resource snapshot (memory, GPU utilization) taken after a job
    exits is unreliable; measure concurrently, in-run, with synchronization
    across ranks where needed.
16. Cost-efficiency and scientific correctness are not in tension when done
    right: bounded debugging with explicit stop conditions, targeted
    inspection, and reuse-before-rebuild all reduce cost *and* reduce the
    risk of the kind of overclaiming documented in §4.
17. Keep a short, explicit "known benign artifact" list in the living status
    document so the same non-bug is never re-investigated from scratch.
18. A decision to stop/defer/keep-CPU is just as much a deliverable as a
    successful port — require the same evidence bar for all four decision
    types in §6.
