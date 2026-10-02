# Prompt 115 — FMS Within-Tile Rank Ordering Verification (layout=(2,2))

Date: 2026-10-01
Branch: `optimization/cuda-fv3-performance`
Scope: Fortran-only, test-only diagnostic. No CUDA source touched, no ABI
change, no gate relaxation, no topology-builder change, no numerical change,
no commit/push.

## 1. Goal

Empirically verify, using authoritative FMS/Fortran runtime metadata (not
inference), whether FMS's actual within-cubed-sphere-tile MPI rank ordering
under `layout=(2,2)` follows the x-fastest/row-major convention assumed (but
flagged UNVERIFIED) by Prompt 113
(`p113_fms_bridge_subtile_metadata_design_2026-09-30.md` §4) and Prompt 114
(`p114_fms_layout_bridge_abi_plumbing_2026-09-30.md` §10):

```
local_rank_within_tile 0 -> sub_i=0, sub_j=0
local_rank_within_tile 1 -> sub_i=1, sub_j=0
local_rank_within_tile 2 -> sub_i=0, sub_j=1
local_rank_within_tile 3 -> sub_i=1, sub_j=1
```

This is a blocking prerequisite before any future CUDA sub-tile
topology-builder work (roadmap item, Prompt 114 §11 "Option A").

## 2. Method

Added a single `get_environment_variable('RUN_FV3_LOG_FMS_LAYOUT_DIAG', ...)`
-gated diagnostic print inside `domain_decomp`
(`GEOSfvdycore_fortran_perf/.../@fvdycore/tools/fv_mp_mod.F90`), immediately
after `Atm%bd%jec = je` is set (i.e. after the real
`mpp_define_layout`/`mpp_define_mosaic`-derived compute/data domain bounds
and `Atm%tile`/`Atm%layout` are fully established for that rank). Default
unset -> zero production behavior change. One line per rank:

```
FV3_FMS_LAYOUT_DIAG rank=<gid> pe=<mpp_pe()> tile=<Atm%tile> layout=<layout(1)>x<layout(2)>
  compute=<is>:<ie>,<js>:<je> data=<isd>:<ied>,<jsd>:<jed> mpirank=<MPI_Comm_rank(commglobal)>
```

No new decomposition mechanism was built — `layout=(2,2)` was reached purely
via the existing, already-validated job-script resource path:
`FV_NX`/`FV_NY` (shell) -> `AGCM.rc` `NX:`/`NY:` -> `FV_StateMod.F90`'s
`Atm%layout(1)=nx`, `Atm%layout(2)=ny/6` -> `fv_init2` -> `run_setup` ->
`domain_decomp`. CUDA backend disabled (`RUN_CUDA_FV3: 0`) and Held-Suarez
forcing disabled (`RUN_HELD_SUAREZ: 0`) in both jobs — pure FMS decomposition
check, zero CUDA code path entered.

Rebuilt `fvdycore` + `StandAlone_FV3_Dycore.x` against the modified
`fv_mp_mod.F90` (Intel MPI production stack, same as P113/P114). Binary
sha256 `808c93c72a5550ce4a7bafdbc69bcce1236469c7da1ed5fcf5b44f2871d20ecc`.
Confirmed via `strings <bin> | grep -c FV3_FMS_LAYOUT_DIAG` -> `1` that the
new format string compiled in.

Grid: C8 (`AGCM_IM=8`), L10, R8, `COLDSTART=1`, 1 heartbeat step
(`RUN_DT=3600`, `JOB_SGMT='00000000 010000'`). Partition: `compute`
(CPU-only, no GPU resource requested — correct, since this is a pure-FMS
check with CUDA disabled). Two jobs:

| Job | Dir | Ranks | FV_NX/FV_NY | layout | Slurm job | Result |
|---|---|---|---|---|---|---|
| Control | `experiments/p115_c8_layout1x1_control/` | 6 | 1/6 | (1,1) | 58674719 | `rc=0` |
| Test | `experiments/p115_c8_layout2x2_test/` | 24 | 2/12 | (2,2) | 58675000 | `rc=0` |

## 3. Stage 4 — Control Run (6 ranks, layout=(1,1))

All 6 `FV3_FMS_LAYOUT_DIAG` lines present, one per rank, `tile = rank+1` for
every rank (1-6), `layout=1x1`, `compute=1:8,1:8` (full C8 face) on every
rank, `rank==pe==mpirank` identically. `Status = 0`, correct binary used
(`***** USING ****` line matches `scratch_.../StandAlone_FV3_Dycore.x`), one
untruncated `DYN Date` line, zero FATAL/error matches.

**Control: PASS — diagnostic reads sensible metadata.**

## 4. Stage 5 — Test Run (24 ranks, layout=(2,2))

All 24 lines present. `Status = 0`, correct binary, one untruncated `DYN
Date` line, zero FATAL/error matches, zero CUDA/GPU string matches anywhere
in the job's stdout (Stage 12, see §7).

Observed grouping by `tile`:

| tile | ranks | compute windows (is:ie, js:je) |
|---|---|---|
| 1 | 0,1,2,3 | (1:4,1:4) (5:8,1:4) (1:4,5:8) (5:8,5:8) |
| 2 | 4,5,6,7 | (1:4,1:4) (5:8,1:4) (1:4,5:8) (5:8,5:8) |
| 3 | 8,9,10,11 | (1:4,1:4) (5:8,1:4) (1:4,5:8) (5:8,5:8) |
| 4 | 12,13,14,15 | (1:4,1:4) (5:8,1:4) (1:4,5:8) (5:8,5:8) |
| 5 | 16,17,18,19 | (1:4,1:4) (5:8,1:4) (1:4,5:8) (5:8,5:8) |
| 6 | 20,21,22,23 | (1:4,1:4) (5:8,1:4) (1:4,5:8) (5:8,5:8) |

Full raw lines captured in job output
(`experiments/p115_c8_layout2x2_test/slurm-58675000.out` and matching
`scratch_*.log`).

## 5. Stage 6/7 — Derive and Test the Assumed Formula

Deriving `(sub_i, sub_j)` directly from each rank's own `(is,ie)`/`(js,je)`
compute-domain bounds (not inferred from a single combined index — this
structurally avoids any x/y ambiguity, see §8):

- `is:ie = 1:4` -> `sub_i=0`; `is:ie = 5:8` -> `sub_i=1`
- `js:je = 1:4` -> `sub_j=0`; `js:je = 5:8` -> `sub_j=1`

Local rank within tile (`local_rank = global_rank - tile_start`, tile_start
= 0,4,8,12,16,20 for tiles 1-6 respectively, confirmed in §6):

| local_rank | sub_i | sub_j | assumed formula (`sub_j*layout_x+sub_i`) |
|---|---|---|---|
| 0 | 0 | 0 | 0 — match |
| 1 | 1 | 0 | 1 — match |
| 2 | 0 | 1 | 2 — match |
| 3 | 1 | 1 | 3 — match |

Identical pattern on all 6 tiles (table in §4). **The assumed x-fastest/
row-major formula is VERIFIED exactly, with zero exceptions, on all 6
faces.**

## 6. Stage 8 — Tile Contiguity

Ranks group into exactly 6 contiguous blocks of 4: `[0-3]->tile1,
[4-7]->tile2, [8-11]->tile3, [12-15]->tile4, [16-19]->tile5,
[20-23]->tile6`. This matches the pre-existing source-confirmed formula
(`pe_start(n) = pelist(1) + (n-1)*layout(1)*layout(2)`, P113 §/source read)
with `pelist(1)=0` observed. **4-ranks-per-tile: PASS. Tile-major-contiguous:
YES.**

## 7. Stage 9 — Communicator Rank Agreement

Every one of the 30 collected lines (6 control + 24 test) has
`rank == pe == mpirank` (i.e. `gid == mpp_pe() == MPI_Comm_rank(commglobal)`)
identically, with zero offset or reordering observed at either rank count.

**Communicator check: IDENTICAL. No translation required.**

## 8. Stage 10 — x/y Ambiguity Check

Not applicable in the way a single combined "quadrant index" diagnostic
would need an uneven grid to disambiguate — this diagnostic prints `is:ie`
and `js:je` as two independent bound pairs per rank, so `sub_i` and `sub_j`
are each derived directly from their own axis, never from a combined scalar.
Ambiguity is structurally avoided by construction, independent of whether
the grid is square. (The C8/2=4-per-subtile grid is square, but this does
not weaken the result — the four distinct `(is:ie, js:je)` pairs observed
per tile are each uniquely attributable to one `(sub_i, sub_j)`.)

**Stage 10: PASS (ambiguity structurally excluded by the diagnostic's
design, not by grid asymmetry).**

## 9. Stage 11 — Optional Generalization Check

Skipped. The primary hypothesis (x-fastest/row-major, `layout=(2,2)`) was
verified with zero exceptions across all 6 tiles and both rank counts
tested (6 and 24) — the bounded-debugging stop condition (CLAUDE.md
"Debugging") was reached: hypothesis resolved, no ambiguity, no conflicting
evidence. An additional `(3,1)`/`(1,2)` layout run was judged redundant
confirmatory evidence rather than new information, so it was not run, in
keeping with the cost-efficiency guidance to stop once evidence is
sufficient. If a future task wants converging evidence under a second,
non-square layout, running `p115_c8_layout1x2_test` (12 ranks, `FV_NX=1`,
`FV_NY=12`) would reuse the same diagnostic unchanged.

## 10. Stage 12 — CUDA Non-Involvement

Both jobs set `RUN_CUDA_FV3: 0` and `RUN_HELD_SUAREZ: 0` in `AGCM.rc`.
`grep -i "cuda\|gpu"` against both jobs' full stdout returns zero matches.
No CUDA source file was read, compiled with new logic, or touched by this
task's diagnostic. **CUDA involvement: NO, confirmed by both configuration
and log inspection.**

## 11. Stage 13 — Cleanup / Production-Safety

The diagnostic is gated behind `get_environment_variable
('RUN_FV3_LOG_FMS_LAYOUT_DIAG', ...)`, unset by default. No separate removal
step is required: with the env var unset (the default, standard, and every
other standing job script's state), `p115_envlen == 0` and the print is
never executed — default production behavior is provably unchanged by
construction, not just by observation. The two-line declaration addition
(`p115_envval`, `p115_envlen`, `p115_mpi_rank`) and the ~10-line print block
are the only source changes; no existing line was modified or removed.

## 12. Stage 14 — Canonical Formula

**VERIFIED canonical formula** (supersedes "UNVERIFIED" language in P113 §4
and P114 §10):

```
tile_start(t)   = pelist(1) + (t-1) * layout_x * layout_y      [t = 1..6]
local_rank      = global_rank - tile_start(Atm%tile)
sub_i           = local_rank mod layout_x
sub_j           = local_rank  /  layout_x        (integer division)
```

Equivalently, the inverse used for verification:
`local_rank = sub_j * layout_x + sub_i` — confirmed exact, zero exceptions,
all 6 tiles, both rank counts (6, 24) tested.

## 13. Production Safety Statement

- CUDA production code changed: **NO**
- Bridge gates changed: **NO**
- MPI transport gates changed: **NO**
- Real 24-rank CUDA model run: **NO** (this was a 24-rank **Fortran-only**
  run, `RUN_CUDA_FV3=0`; no CUDA bridge, topology builder, or halo transport
  was invoked at any rank count)
- All four pre-existing >6-rank CUDA bridge/runtime gates
  (`fv3_gpu_dyncore_bridge.F90:168` `comm_size/=6` FATAL;
  `build_rank_of_tile`'s tile-map FATAL; `fv3_gpu_runtime.cpp:1585`
  `mpi_size!=cubed_sphere_tiles` check; `fv3_gpu_halo_mpi.cpp`'s
  `create_mpi_transport()` `comm_size==6` gate) were not read, built against,
  or exercised by this task at all — this task touched only
  `GEOSfvdycore_fortran_perf`'s `fv_mp_mod.F90`, zero files in this repo's
  `cuda/` tree.
- No commit, no push (both repos).

## 14. Unresolved / Honest Caveats

- Only `layout=(1,1)` (control) and `layout=(2,2)` (test) were run. The
  formula is verified for these two cases; Stage 11's optional third layout
  was skipped as redundant (§9) — a future task could still run it for
  additional confirmatory margin at near-zero cost if desired, but it is not
  required to proceed.
- This verifies FMS's within-tile rank ordering under the standalone
  `StandAlone_FV3_Dycore.x` harness's exact decomposition call path
  (`domain_decomp` via `mpp_define_layout`/`mpp_define_mosaic`) at C8. It
  does not re-verify this at every grid size — but `domain_decomp` is
  resolution-independent code, and the mechanism (`pe_start`/`mpp_define_*`)
  does not branch on grid size, so there is no structural reason to expect a
  different ordering at C512/C1024.
- `fv3_gpu_interface_mod.F90` (legacy mirror module, flagged
  not-independently-compiled since P114) remains unverified/unchanged —
  out of scope for this task.

## 15. Next Task

**DONE (Prompt 116, 2026-10-01): the recommended design task below was
executed.** See
`docs/performance/p116_subtile_id_derivation_design_2026-10-01.md` — derived
the canonical `subtile_id_from_rank(mpi_rank, tile, layout_x, layout_y)`
pure-function formula (restating §12 below with explicit bounds), confirmed
NO new ABI field is required, proved the layout=1×1 reduction is exact, and
recommended a bounded next implementation slice (derivation helper + 1×1
shadow-topology equivalence check + 24-rank test-only oracle unit test, no
gate relaxation). The recommendation text below is retained for historical
context.

**FMS within-tile rank ordering is VERIFIED, matching the exact formula
already assumed (but previously unverified) in P113/P114.** No metadata
translation redesign is needed before resuming sub-tile topology-builder
work. Recommended next step (new task, out of scope for this one): use the
now-VERIFIED formula from §12 to design (not yet implement) the CUDA
sub-tile topology-builder's rank -> `(sub_i, sub_j)` derivation inside the
bridge, strictly as a design/mapping task first (matching this project's
established pattern of design-then-implement for anything that touches the
bridge's >6-rank gates), before any gate relaxation is attempted.

Lower priority, deferred (unchanged from P113/P114): (a) Phase B
corner-transport design; (b) folding Prompt 109's duplicated vector-sign
formula into one shared helper.

```
FMS WITHIN-TILE RANK ORDERING VERIFIED — READY FOR SUB-TILE ID DERIVATION INTEGRATION
```
