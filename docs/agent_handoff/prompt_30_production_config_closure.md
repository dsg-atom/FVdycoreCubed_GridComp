# Prompt 30: Production Configuration Closure Attempt

## 1. Objective

Determine whether the CUDA dyn_core production interface can execute the
remaining production-relevant configuration branches (`q_con != 0`,
`MOIST_CAPPA`, `USE_COND`, `a_imp <= 0.999`) without numerical differences,
and verify the Intel-Fortran <-> CUDA ABI. Not a performance task, not a new
numerical-algorithm task. Method: targeted source inspection (this project's
own fence/rejection code, its comments, and the referenced milestone
handoff), not new Slurm jobs — every question below was already answered
unambiguously and redundantly by existing, self-documenting source, so
running a job would only reproduce a already-confirmed rejection (contrary
to the cost-efficiency policy against redundant evidence-gathering).

## 2. Headline Finding

**MOIST_CAPPA, USE_COND, and `a_imp <= 0.999` are not unvalidated — they are
confirmed UNIMPLEMENTED in the CUDA backend, by explicit, intentional,
multiply-documented design.** This is not new: it was flagged at milestone
3A.9 (`milestone_3a9_gate_c1_closure.md:719,835`) as "must widen before
production integration," and the real production Fortran bridge
(`fv3_gpu_dyncore_bridge.F90:589-596`) already contains a comment stating
the exact same fact in the exact same words this prompt would have
produced. Prompt 30's contribution is confirming this is *still* true,
tracing every independent enforcement point, and discovering that Stage A
(`q_con != 0` alone) cannot be tested in isolation the way originally
planned — see §4.

## 3. Configuration-Gap Table (§1)

| Option | Authoritative Fortran | Current CUDA support | ABI/config path | Validated? | Test needed? |
|---|---|---|---|---|---|
| `q_con` (field, any value) | `dyn_core.F90:844-847` (halo group); read only inside `USE_COND`-gated code, `nh_utils.F90:377` (`Riem_Solver_c`) / equivalent in `Riem_Solver3` | Field registered (`FieldId::q_con`, halo group, mutable-state contract) but read **only** inside `if (a.use_cond)` (`fv3_riem_solver_c.cu:103-105`) — dead in the implemented config | Passed as ordinary state field via `fv3_gpu_initialize_state`/halo exchange | Validated **only at q_con ≡ 0** (Gate C2 + every subsequent depth/grid row in CURRENT_STATE §5) | Cannot be usefully tested alone — see §4 |
| `MOIST_CAPPA` | `nh_core.F90` Riemann-solver kappa closure (`cappa(i,j,k)` vs. constant `akap`); compiled unconditionally in this project's Fortran fixture (`@fvdycore/CMakeLists.txt:130`) | Explicit runtime reject, `fv3_gpu_runtime.cpp:1101-1102`; WP-D operators have a real `moist_cappa`-aware branch (`wp_d_operators.cpp:585`) that is unreachable because the top-level fence blocks it | ABI field exists (`fv3_gpu_api.h:177`, `fv3_gpu_numerical_config.moist_cappa`), plumbed to `cfg.moist_cappa` (`fv3_gpu_runtime.cpp:1599`), then rejected | **NOT IMPLEMENTED** (confirmed, not merely unvalidated) | Cannot pass — rejection is by design |
| `USE_COND` | Same Riemann-solver call, additional `gm2`/`q_con` terms (`nh_utils.F90:372-390`); also compiled unconditionally | Same reject (`fv3_gpu_runtime.cpp:1101-1102`); WP-B's `d_sw` driver additionally hard-fails at kernel-launch time (`wp_b_dsw_cuda.cu:889`) | Same ABI field, same plumbing, same reject | **NOT IMPLEMENTED** | Cannot pass — rejection is by design |
| `a_imp <= 0.999` | Vertical-solver branch selector (`nh_utils.F90:402-411`); three branches (SIM3p0, RIM_2D, SIM1); only SIM1 (`a_imp = 1.0`) implemented | Explicit reject, `fv3_gpu_runtime.cpp:1082-1083`; second independent reject inside the Riemann-solver launcher itself (`fv3_riem_solver_c.cu:247-251`) | ABI field `cfg.a_imp`, read from the **real** namelist value (`Atm%flagstruct%a_imp`, `fv3_gpu_dyncore_bridge.F90:512` — not hardcoded), gated by a third, Fortran-side `mpp_error(FATAL)` guard (`fv3_gpu_dyncore_bridge.F90:622-627`) | **NOT IMPLEMENTED** (SIM1/`a_imp=1.0` only) | Cannot pass — rejection is by design, triple-gated |
| `n_con` / tracer-count implications | `dyn_core.F90:405-412` derived ladder (`convert_ke`, `vtdm4`, `d2_bg_k1/k2`) | Faithfully reproduced, not hardcoded (`fv3_gpu_dyncore_bridge.F90:559-570`); passed into WP-D operators (`wp_d_contract.cpp:408`) | Derived value, not a flag — no gap | Validated (Gate C2 defect #11 fix) | None |
| `cappa`/`q_con` coupling | Only exists inside `MOIST_CAPPA`/`USE_COND` branches | `cappa` rides the halo-exchange group "only under MOIST_CAPPA" (`fv3_gpu_halo_mpi.cpp:160`) — correctly conditioned, but the condition is permanently false in this build | n/a | n/a (dead while MOIST_CAPPA=0) | Reopens automatically once MOIST_CAPPA is implemented |
| Intel-ABI-sensitive types | — | All ABI structs are `bind(C)` with fixed-width `iso_c_binding` kinds (`c_int32_t`/`c_int64_t`/`c_double`/`c_int8_t`/`c_ptr`/`c_intptr_t`); every entry point runtime-checks `abi_version`/`struct_bytes` via `header_ok` (`fv3_gpu_runtime.cpp:40,1430,1452,1489,1583,1693,2399`); MPI communicator crosses as an `integer(c_intptr_t)` token, converted with `MPI_Comm_f2c` (comment at `fv3_gpu_abi_mod.F90:14`, call site `fv3_gpu_dyncore_bridge.F90:259`) | Sound by construction | **PASS** — see §7 | None beyond what's already run |

## 4. Stage A (`q_con != 0` alone): why it cannot be isolated as planned

The plan called for a matched Fortran-vs-CUDA run with `q_con != 0` and
MOIST_CAPPA/USE_COND left off, to test whether nonzero `q_con` propagates
correctly as inert state. That test is **not constructible**:

- `@fvdycore/CMakeLists.txt:130`: `target_compile_definitions (${this)
  PRIVATE MOIST_CAPPA USE_COND)` — the authoritative standalone Fortran
  fixture used for every validation row in `CURRENT_STATE.md` §5 has
  `MOIST_CAPPA`/`USE_COND` compiled in **unconditionally**. There is no
  runtime flag on the Fortran side to turn them off; it is a preprocessor
  choice baked into the binary.
- Every PASS in the existing validation matrix relied on `q_con ≡ 0` making
  Fortran's moist branch degenerate algebraically to the dry formula (the
  documented dry/wet equivalence argument, `run_small_perf_bench.cpp:240`
  and `fv3_gpu_internal.hpp:149-159`, citing `fv_dynamics.F90:454`).
- Setting `q_con != 0` in this Fortran fixture therefore **activates real
  moist-thermodynamics formulas** in Fortran (the `USE_COND`/`MOIST_CAPPA`
  branches are live code in this binary, gated only by the *value* of
  `q_con`, not by a separate switch). CUDA has no matching code path — it
  is rejected outright if the flags are set, and dead if they are not
  (`fv3_riem_solver_c.cu:103-105`).
- Consequently, a nonzero-`q_con` Fortran run and a `use_cond=0` CUDA run
  are not computing the same physics by construction — any observed
  difference would not be a "discrepancy to localize," it would be the
  expected, already-understood consequence of the missing implementation.
  Running the job would not produce new information.

**Conclusion: Stage A is coupled to Stages B/C, not independent of them.**
The three cannot be closed piecemeal — closing `q_con != 0` support requires
implementing `MOIST_CAPPA`/`USE_COND` in CUDA, which is real numerical
kernel work (the Riemann solver's `gm2`/`cp2`/`peg` terms, `nh_utils.F90:
372-390`), explicitly outside this task's `NOT: a new numerical-algorithm
task` boundary and outside the "≤3 plumbing fixes" debug policy (§12).

## 5. Stages B (MOIST_CAPPA) / C (USE_COND): PASS/FAIL/NOT REACHED = **NOT REACHED**

Confirmed via source, triply independent and mutually consistent:

1. Top-level runtime fence: `fv3_gpu_runtime.cpp:1101-1102` rejects any
   config with `use_cond` or `moist_cappa` set, citing "a documented
   divergence from the real GEOS FV3 build ... see the milestone 3A.9
   handoff."
2. WP-B kernel-launch-time fence: `wp_b_dsw_cuda.cu:889` independently
   rejects `use_cond` at the point of use.
3. Production Fortran bridge: `fv3_gpu_dyncore_bridge.F90:595-596` **forces
   `cfg%use_cond = 0` / `cfg%moist_cappa = 0`** regardless of the real
   namelist value, with a comment stating this is "a REAL, recorded
   coverage gap, not a translation."
4. Existing unit-test coverage already asserts the reject fires:
   `test_wp_b5_dsw_cuda.cpp:2353-2354` (`use_cond=1` -> expect
   `FV3_GPU_UNSUPPORTED_CONFIGURATION`).

This is not ambiguous (hard-stop condition §13.1 does not apply — the
meaning of both flags is precisely documented at `nh_utils.F90:372-390` and
corroborated by three independent CUDA-side citations). It is a genuine,
scoped-out numerical implementation gap, not a config-plumbing defect, not
fixable within this task's 3-fix debug budget.

**Worst discrepancy: N/A — configuration rejected before any state mutation
(by design; `fence_production_config`'s explicit purpose per its own
comment, `fv3_gpu_runtime.cpp:1014-1016`, is exactly to fail before mutating
state).**

## 6. Stage D (`a_imp <= 0.999`): PASS/FAIL/NOT REACHED = **NOT REACHED**

Same pattern, triple-gated:

1. Top-level runtime fence: `fv3_gpu_runtime.cpp:1082-1083`.
2. Riemann-solver launcher fence: `fv3_riem_solver_c.cu:247-251`, citing
   exactly which branches are missing (`SIM3p0` for `a_imp < -0.01`,
   `RIM_2D` for `a_imp <= 0.5`; only `SIM1_solver` at `a_imp = 1.0` is
   implemented).
3. Fortran bridge: `fv3_gpu_dyncore_bridge.F90:622-627` — `if
   (Atm%flagstruct%a_imp <= 0.999) call mpp_error(FATAL, ...)`, with a
   message explaining the specific reason (`m_split`/`scale_m` become live
   only below that threshold, and this routine's initialization-time
   `m_split` predates `fv_dynamics.F90:616`'s recompute).

Unlike `use_cond`/`moist_cappa`, `a_imp` **is** read from the real namelist
value (`Atm%flagstruct%a_imp`, line 512) rather than hardcoded — the gate is
a genuine runtime check on a real config value, not a silent override. The
production first-run configuration mandated by `project_architecture.md`
is `a_imp = 1.0`, so this gap does not block the currently-required
first-run envelope; it blocks any *future* non-SIM1 configuration.

**Worst discrepancy: N/A — rejected before mutation, both by the Fortran
bridge (`FATAL`, before CUDA is even called) and by the CUDA runtime
independently.**

## 7. Stage F (Intel-Fortran ABI verification): **PASS**

- All structs crossing the ABI (`fv3_gpu_error`, `fv3_gpu_field_desc`,
  `fv3_gpu_create_desc`, `fv3_gpu_state_desc`, `fv3_gpu_step_desc`,
  `fv3_gpu_capabilities`, `fv3_gpu_numerical_config`) are `bind(C)` Fortran
  derived types built exclusively from `iso_c_binding` fixed-width kinds
  (`c_int32_t`, `c_int64_t`, `c_int8_t`, `c_int16_t`, `c_double`, `c_ptr`,
  `c_intptr_t`). No Fortran array descriptors, no `character` strings, no
  assumed-shape arguments cross the boundary — every array is a raw
  `type(c_ptr)` plus explicit extent/stride integers.
- `use_cond`/`moist_cappa`/etc. are `integer(c_int8_t)` on the Fortran side
  against `uint8_t` on the C side — signed vs. unsigned 8-bit, but every
  value used is 0 or 1, so the bit pattern is identical; not a real
  incompatibility.
- Every public entry point validates `abi_version`/`struct_bytes` via
  `header_ok(...)` against `sizeof()` on the C++ side
  (`fv3_gpu_runtime.cpp:40,1430,1452,1489,1583,1693,2399`) before touching
  the struct — this is a real, load-bearing runtime cross-check against
  compiler-specific struct packing/alignment drift, not a documentation
  claim.
- The MPI communicator crosses as `integer(c_intptr_t) :: communicator_token`
  (not a raw `MPI_Comm`), explicitly to dodge the MPICH-int-vs-OpenMPI-
  pointer representation difference; converted on the C++ side with
  `MPI_Comm_f2c` (comment: `fv3_gpu_abi_mod.F90:14`; call site:
  `fv3_gpu_dyncore_bridge.F90:259`).
- Empirical confirmation already exists and is extensive: every row in
  `CURRENT_STATE.md` §5's validation matrix (Gate C2 through Prompt 29,
  c8/c16/c32, R8/R4, up to 310-call depth) ran the real
  `StandAlone_FV3_Dycore.x` — an Intel `ifort`-built executable — linked
  against the GCC-built CUDA archives through exactly this ABI, with zero
  ABI-attributable failures across dozens of jobs. No new job was run for
  this stage; re-running one would not add information beyond what dozens
  of prior jobs already established.
- Known, pre-existing, unchanged caveat (not new): this proves the
  standalone-fixture ABI link, not the full production GEOS build's
  Intel-everywhere ABI (`CURRENT_STATE.md` §7's existing limitation stands
  as-is).

## 8. Stage E (combined production-like case): **NOT REACHED**

Per the task's own gating rule ("only if Stages A-D individually pass") —
A/B/C/D did not pass (A is coupled to B/C's gap; B/C/D are confirmed
unimplemented), so no combined case was attempted.

## 9. Remaining Hardcoded Production Assumptions (§10)

Swept the production Fortran bridge (`fv3_gpu_dyncore_bridge.F90:489-629`)
line by line. Everything is either read from the real `Atm%flagstruct`/
`Atm%gridstruct` namelist state, or is a documented, deliberate, currently-
inert placeholder:

- `cfg%use_cond = 0`, `cfg%moist_cappa = 0` — **the one live, documented,
  already-covered gap** (§5). Not new.
- `cfg%hord_height = 0` — documented "no Fortran counterpart... not the
  equivalent quantity," inert (not read by any implemented stage per
  `wp_b_dsw_cuda.cuh`'s field table).
- `cfg%vertical_solver = 0` — documented "no override, meaning use a_imp
  alone" — consistent with only SIM1 being implemented; becomes meaningful
  only if a second vertical solver is ever added.
- `cfg%rayleigh_ztop = 0.0` — documented "left zero rather than guessed;
  inert here because rayleigh_enabled is computed from the real gate
  (`RF_fast .and. tau > 0`) and is false in this configuration."
- `cfg%rot3 = 0`, `cfg%use_sg = 0` — documented "compile-time in the
  Fortran, absent from this build's flags" (i.e., there is no live namelist
  value to read; these are truly off in this build, not silently dropped).
- `cfg%flags = 0`, `cd%communicator_token = 0` (in `fv3_gpu_create_desc`,
  filled at create-time before the token is known) — placeholders filled
  later in the same routine or by design (`flags` is reserved).

**No other production-relevant hardcoded/synthetic value was found beyond
the already-known `use_cond`/`moist_cappa` forced-zero.** The
benchmark-only hardcode found during source review
(`run_small_perf_bench.cpp:2103`, `cfg.moist_cappa = 0;`) is inside the
standalone micro-benchmark harness, not the production bridge, and is out
of scope.

## 10. Numerical Comparison Policy (§11)

No comparison runs were executed this prompt — every stage that would have
required one is blocked before any kernel launch (rejected pre-mutation, by
design, per `fence_production_config`'s own stated purpose). §11's
BITWISE/ROUND-OFF/SMALL/SIGNIFICANT classification therefore does not apply
to any stage this prompt; `compare_dcx_backends.py` was not invoked (no new
savepoints were generated).

## 11. Fixes Applied

**None.** No source was modified. Per §12's debug policy, the only
available path to closing B/C/D is implementing missing numerical
formulations (the `MOIST_CAPPA`/`USE_COND` Riemann-solver terms and the
`RIM_2D`/`SIM3p0` vertical branches) — explicitly disallowed fix categories
for this task ("scientific algorithm changes, new numerical formulation").
**Numerical algorithm changed: NO.**

## 12. Regression (§15)

Not applicable — no source changed, so no regression run was performed.

## 13. Next Recommended Step

This is a real, scoped, well-understood engineering task, not a research
question: **implement `MOIST_CAPPA`/`USE_COND` in the CUDA vertical solver**
(`fv3_riem_solver_c.cu`'s `peg`/`gm2`/`cp2` terms, `nh_utils.F90:372-390`,
already partially scaffolded — the `use_cond`/`moist_cappa` struct fields,
the `q_con`/`cappa` field registrations, the conditional halo-group
composition, and even a real WP-D `moist_cappa`-aware branch
(`wp_d_operators.cpp:585`) already exist; only the Riemann-solver kernel
body and the two fence rejections need to change). This is WP-C/WP-B
numerical work requiring a dedicated implementation-and-validation prompt
(construct a real nonzero-`q_con` fixture, implement the missing formulas,
re-run Gate-C2-style validation) — appropriately a **separate, larger
prompt**, not a continuation of this one. `a_imp <= 0.999` (the `RIM_2D`/
`SIM3p0` vertical branches) is a second, independent, similarly-scoped
follow-on. Until either closes, "broader GEOS integration" remains bounded
to the already-validated production first-run subset (`hord_*=±6`,
`grid_type=0`, `hydrostatic=false`, `a_imp=1.0`, `use_cond=moist_cappa=0`).

## 14. Files Touched This Prompt

- `docs/agent_handoff/prompt_30_production_config_closure.md` (this file).
- `docs/agent_handoff/CURRENT_STATE.md` (updated; corrected the stale HEAD
  reference introduced during Prompt 29 finalization, added Prompt 30
  findings).

No CUDA/C++/Fortran source was modified. Not committed/pushed — awaiting
explicit user request per project policy.
