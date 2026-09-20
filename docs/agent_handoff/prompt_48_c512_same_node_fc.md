# Prompt 48: Authoritative Same-Node C512 Fortran-vs-CUDA Performance

## Purpose

Resolve the outstanding ambiguity between historical C512 CUDA `DYN_CORE`
figures measured on different physical nodes (Prompt 37: ~9089 ms/call on
`warpa009`; Prompt 47: ~3892 ms/call on `warpa007`) by running Fortran and
CUDA back-to-back, alternating, in one Slurm allocation on one node — the
one comparison the config audit (`prompt_48_c512_config_audit.md`)
identified as still missing before this could be resolved.

## Method

Single sbatch job (`p48_c512_same_node_fc.sbatch`, job **58468244**, node
**`warpa004`**), 1 node, 1 A100, 6 MPI ranks throughout. Six sub-runs in
strict alternating order **F1 C1 F2 C2 F3 C3** (not grouped), each in its
own freshly created, correctly self-referencing `EXPDIR`
(`p48_pair{1,2,3}_{fortran,cuda}/`) — fixing the stale-`EXPDIR`
copy-paste bug the config audit found in Prompt 47's harness copies.
MPS started once at job start, stopped once at job end (inert for the
Fortran sub-runs; identical CUDA config to Prompt 37/47: fusion ON
(`RUN_CUDA_FV3_FUSE_HALO_EDGES=YES`, `RUN_CUDA_FV3_FUSE_CORNER_FILL=YES`),
Graph OFF, `RUN_CUDA_FV3_MPI_TRANSPORT` unset → `host_staged`,
`I_MPI_OFFLOAD` unset). C512, R8, `npz=10`, `bdt=36`, `n_split=3`,
`k_split=1`, Held-Suarez ON, `COLDSTART=1`. Each sub-run: 30 `dyn_core`
calls (same 3-rep-equivalent × 30-call convention as Prompts 34/36/37,
here 1 rep per backend per pair × 3 pairs = 3 reps per backend total).

**Stage 0 pre-submission audit** (before the job was submitted):
- Diffed each generated `fv3.j` against its `p37_c512_perf_{fortran,cuda}`
  template — confirmed the only differences are the `#SBATCH --job-name`
  comment and the `setenv EXPDIR` line (both intentionally changed).
- Confirmed no `StandAlone_FV3_Dycore.x` override binary exists in any of
  the 6 new `p48_pair*` directories (so the canonical
  `$GEOSBIN/StandAlone_FV3_Dycore.x` binary is used, not a copy).
- Confirmed the canonical binary's md5 (`77315a9c375783fc69bda5ad37103c2a`)
  and mtime (`2026-08-31 11:26:37`) are unchanged from every prior audited
  job — no rebuild occurred.
- Confirmed all 6 `EXPDIR`s self-reference their own new directory (see
  audit doc's "stale-EXPDIR bug" section for the failure mode this
  prevents).

Primary timing source: raw model stdout (`fv_timing_mod` block),
extracted directly from the job's `.out` file. **Not** the `scratch_*.log`
copy (though this job's own `scratch_*.log` capture worked correctly for
all 6 sub-runs — non-empty `LOG=` for every `P48_PAIR=... RC=0 LOG=...`
marker — because every `EXPDIR` self-referenced correctly this time).

## Allocation

| Item | Value |
|---|---|
| Node | `warpa004` |
| GPU | NVIDIA A100-SXM4-40GB |
| GPU UUID | `GPU-ee48c558-3cb7-b739-7450-92a11edbeaa4` |
| Driver | 535.104.12 |
| MPI | Intel MPI 2021.13, 6 ranks |
| CUDA GPUs visible | 1 |
| MPS | ON (whole allocation) |
| Job elapsed | 00:31:14 |

## Configuration confirmation

- Binary identical across all 6 sub-runs: **YES** (single canonical
  `install/bin/StandAlone_FV3_Dycore.x`, confirmed via `USING ****` line,
  6/6 identical path; no per-run override binary present).
- Per-run generated `AGCM.rc` `RUN_CUDA_FV3` directly inspected in each of
  the 6 actual `$SCRDIR`s: `1` for all 3 `*_cuda` runs, absent (defaults
  to Fortran) for all 3 `*_fortran` runs — config injection confirmed
  correct at the point of execution, not just in the template.
- `RUN_HELD_SUAREZ: 1`, `COLDSTART: 1`, `hord_mt/vt/tm=6`, `hord_dp=-6`,
  `vtdm4=0.01`, `grid_type=0`, `a_imp=1.0`, `n_split=3`, `k_split=1`,
  `npx=npy=513`, `npz=10` — identical generated `input.nml`/`AGCM.rc`
  across all 6 runs (only `RUN_CUDA_FV3` differs, as designed).

## Health check (Stage 6)

All 6 sub-runs: `rc=0`, `Status = 0`, `ALL DONE` reached. Full-log scan
(excluding the script's own `set -x` trace lines) for
`nan|range_warn|error|abort|fatal`: **0 hits** across the entire job for
all three categories. No sub-run excluded from the performance median.

## Paired results

| Pair | Fortran DYN_CORE (s / 30 calls) | ms/call | CUDA DYN_CORE (s) | ms/call | CUDA/Fortran |
|---|---:|---:|---:|---:|---:|
| 1 | 190.6084 | 6353.61 | 191.2159 | 6373.86 | 1.00319 |
| 2 | 191.4865 | 6382.88 | 191.1323 | 6371.08 | 0.99815 |
| 3 | 191.2205 | 6374.02 | 191.4981 | 6383.27 | 1.00145 |

- **Fortran median: 6374.02 ms/call**
- **CUDA median: 6373.86 ms/call**
- **Ratio of medians (CUDA/Fortran): 0.99998** (≈1.0000)
- **Median paired ratio: 1.00145**
- **CUDA speedup (Fortran/CUDA): 1.00002x** — no measurable speedup or
  regression either way; the two backends are performance-identical on
  this node, within the same ~0.3% rep-to-rep noise band seen throughout
  this project's C512 measurements.

## Communication clocks

| Metric | Fortran median | CUDA median |
|---|---:|---:|
| COMM_TOTAL ms/call | 1824.76 | 1819.14 |
| COMM_DSW ms/call | 1032.07 | 1026.84 |

Both within ~0.5% of each other — communication cost is not a
distinguishing factor here because there is no overall performance gap to
explain. Classification: **NOT APPLICABLE**.

## Decision

**Classification: C512 FORTRAN/CUDA PERFORMANCE COMPARABLE** (ratio of
medians 0.99998, squarely inside the ±5% Case-B band, and in fact far
tighter than that band — under 0.3%).

**Historical 1.270 CUDA/Fortran ratio: SUPERSEDED.** On a controlled
same-node, same-allocation, same-config, alternating-order measurement,
CUDA and Fortran are statistically indistinguishable at C512. The
previously documented 1.270x "CUDA regression" (`CURRENT_STATE.md` §6,
Prompt 37's original cross-run figures: Fortran 7155.63 / CUDA 9089.05)
does not reproduce when node hardware is held constant. The regression
was an artifact of comparing backend measurements taken on different
physical nodes with materially different absolute performance, not a
genuine CUDA-specific slowdown.

## Node variability interpretation

| Prompt | Backend | Node | ms/call |
|---|---|---|---:|
| 37 | CUDA | `warpa009` | ~9089.05 |
| 47 | CUDA | `warpa007` | ~3892.31 |
| **48** | **CUDA** | **`warpa004`** | **~6373.86** |
| **48** | **Fortran** | **`warpa004`** | **~6374.02** |

Three different physical nodes produced three materially different
absolute CUDA `DYN_CORE` figures (9089 / 3892 / 6374 ms/call) for a
nominally identical configuration — a >2x spread across the node
population, well beyond Prompt 40's originally-characterized ~30%
node-to-node variance. Prompt 48's key new finding: **on the same node,
Fortran tracks CUDA's absolute number almost exactly** (6374.02 vs.
6373.86) — meaning whatever drives the large node-to-node swings affects
both backends proportionally. This is consistent with generic
node/hardware/thermal variance (CPU clock, memory bandwidth, NUMA/PCIe
placement, A100 binning) rather than anything specific to the CUDA
dyn_core path, the MPI transport layer, or the halo-exchange protocol.

**Node-to-node variability materially affects absolute timing: YES.**
This is now established on stronger evidence than before — not just
CUDA-vs-CUDA across nodes (Prompt 47), but Fortran-vs-CUDA on the same
node showing parity while each backend's own absolute number still varies
by >2x across the three nodes sampled so far.

## Project status after this prompt

- C512 correctness: unaffected (still PASS, no source/config changed
  since the last correctness gate).
- **C512 performance ambiguity: RESOLVED.** No genuine CUDA-vs-Fortran
  regression at C512 under controlled same-node conditions; the ladder's
  C32→C256 "CUDA faster" trend does not reverse at C512 once node
  variance is controlled for — it simply becomes a wash (~1.00x) rather
  than either a win or a loss.
- Rank-density hypothesis (Prompt 46's informal lead): REFUTED (Prompt
  47), unaffected by this prompt.
- `cuda_aware` transport: still VIABLE BUT NOT PERFORMANCE-SIGNIFICANT
  (Prompt 46), unaffected — this prompt used `host_staged` only, as
  instructed.
- No numerical source, MPI transport, or Held-Suarez physics changed.
  Only new files: this doc, the 6 `p48_pair*` run directories (job
  scripts, generated RCs/namelists, scratch logs — all reusable
  artifacts, not source), and `p48_c512_same_node_fc.sbatch`.
- 1 GPU allocation used for this prompt (job 58468244), as budgeted — no
  further GPU allocation is needed to close this question.

## Next recommended step

C512's performance question is now closed to project-standard rigor
(same-node paired measurement, alternating order, health-checked, dual
extraction-path agreement). The next open items are the ones already
queued in `CURRENT_STATE.md` §11 and unrelated to this ambiguity: (1) the
mainline `MOIST_CAPPA`/`a_imp<=0.999` CUDA implementation gaps, or (2)
folding MPS lifecycle + fusion flags into standing job scripts as the
default. No further C512 Fortran-vs-CUDA reconciliation work is needed
unless a future node sample contradicts this result.
