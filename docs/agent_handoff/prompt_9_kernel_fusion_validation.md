# Prompt 9 — Kernel-Fusion Prototype: Correctness Validation + Perf Read

Git HEAD at time of this pass: `191157f` (`origin/cuda/gpu-bringup`), plus the
uncommitted Prompt 9 diff (see §4 below). Slurm job `58230026`
(`perf_rerun_logs/prompt9_fusion_58230026.out`, sbatch script
`perf_rerun_logs/prompt9_fusion.sbatch`), A100 node `warpa011`, 2026-09-03,
COMPLETED exit 0, walltime 00:01:29.

## 1. What was fused

`c_sw`'s per-level corner fills (`fv3_cgrid_predictor.cu`), gated behind
`NumericalConfig::fuse_corner_fill` / `--fuse-corner-fill` (opt-in, default
off):

- dir=1 (`sw_core.F90:206,236-237`): `fill2_4corners(delp, pt, 1)` +
  `fill_4corners(w, 1)` → one `launch_fill_corner_triplet(..., kCornerFillDirX)`.
- dir=2 (`sw_core.F90:257,284`): `fill2_4corners(delp, pt, 2)` +
  `fill_4corners(w, 2)` → one `launch_fill_corner_triplet(..., kCornerFillDirY)`.

Same per-field order (delp, pt, w) as the two-launch form; the new combined
launcher is `fv3_cgrid_corners.cu`/`.hpp`. Called per level (10 levels) per
direction (2) inside `c_sw`. This is a narrow, mechanical fusion — it does not
touch WP-B/C/D, the halo/corner-service family, or `riem_solver3`.

## 2. Correctness result — CLEAN

Section 3/4/5 of the sbatch script dump `dyn_core_exit` (`it=3`) for all 13
fields on all 6 tiles, once with the default path and once with
`--fuse-corner-fill`, then binary-diff the dumps:

- **78/78 `IDENTICAL`, 0 `DIFFERS`, 0 `MISSING`** (13 fields × 6 tiles).
- R8 ctest: 22/22 PASS.
- Default synthetic smoke (no fusion flag): PASS, exit 0.

This is a materially different outcome than Prompt 8's CUDA Graphs prototype,
which diverged on all 13 exit fields. **Numerical algorithm changed: NO** —
confirmed by bitwise dump equality, not just by inspection.

## 3. Performance result — small, and not cleanly resolved by this one job

nsys profile (`--warmup 5 --iters 20`, 25 total dyn_core calls, same reduced-count
methodology Prompt 7 used for its 17,730/call baseline):

| metric | Prompt 7 baseline | Prompt 9 fused | delta |
|---|---|---|---|
| kernel launches / call | 17,730 | 17,370 | −2.0% |
| kernel busy time / call | ≈74.0 ms | ≈73.0 ms | −1.4% |
| `cudaLaunchKernel` % of API time | 90.6% | 91.1% | unchanged (still launch-bound) |

Raw resident ms/step (32×32, `--warmup 20 --iters 100 --repeats 5`):

| repeat | baseline | fused |
|---|---|---|
| 1 | 44.501 | 22.806 |
| 2 | 44.373 | 41.996 |
| 3 | 35.089 | 44.287 |
| 4 | 44.345 | 39.880 |
| 5 | 40.339 | 39.238 |
| median | 44.345 | 39.880 |

The median-to-median read (~10%) is **not treated as a validated speedup**:
repeat-to-repeat spread is ~9 ms in *both* arms (comparable to the claimed
delta), consistent with ordinary run-to-run noise on this node, not with a
change of this size. The nsys-derived launch-count/kernel-time deltas (~1.4–2%)
are the more trustworthy signal because they come from direct instruction
counting rather than wall-clock timing, and they are the right order of
magnitude for fusing 2 of 17,730 launch-generating call sites. Do not quote the
~10% figure as an established result without a larger/steadier rerun (more
repeats, or a longer per-repeat window) to separate signal from node noise.

Dominant kernel families remain unchanged from Prompt 7 and are **not** touched
by this fusion: `a2b4_global_corner_kernel` (108 calls/dyn_core-call, 6.2% of
kernel time), `copy_corners_kernel` (1,140 calls/call, 5.1%),
`riem_solver3_kernel` (18 calls/call, 3.7%), `pack_edge_kernel` (1,056
calls/call, 3.6%). Any of these would be a larger-leverage fusion/consolidation
target than the one implemented here.

## 4. Uncommitted diff (this pass)

```
cuda/dyn_core/benchmarks/run_small_perf_bench.cpp |  36 +++++-   (--fuse-corner-fill flag, dump wiring)
cuda/dyn_core/cgrid/fv3_cgrid_corners.cu          |  41 +++++    (launch_fill_corner_triplet)
cuda/dyn_core/cgrid/fv3_cgrid_corners.hpp         |  14 +++      (declaration)
cuda/dyn_core/cgrid/fv3_cgrid_predictor.cu        |  54 ++++---  (call-site branch, both directions)
cuda/dyn_core/include/fv3_gpu_api.h               |   9 ++       (public config surface)
cuda/dyn_core/include/fv3_gpu_context.hpp         |  21 ++++
cuda/dyn_core/include/fv3_gpu_internal.hpp        |   7 ++       (fuse_corner_fill=0 field)
cuda/dyn_core/runtime/fv3_gpu_runtime.cpp         |  63 +++++--
```

`fuse_corner_fill` defaults to `0`; the default path is provably unchanged
(same code path as before this pass when the flag is off).

## 5. Status / next steps

- **Prompt 9 (narrow scope: fuse `c_sw`'s corner-fill pair): COMPLETE.**
  Correctness: PASS (bitwise). Performance: small, directionally positive on
  the trustworthy (launch-count) metric, not conclusively resolved on wall-clock
  in this one job.
- This does **not** close the "kernel fusion/batching" optimization candidate
  from Prompt 7 §6 — it validates the *approach* (fuse small repetitive kernels,
  verify bitwise, then measure) on one small pair. The larger-leverage targets
  identified above (`copy_corners_kernel`, `a2b4_global_corner_kernel`,
  `pack_edge_kernel`) remain unfused.
- If continuing kernel-fusion work: target `copy_corners_kernel` next (1,140
  calls/call is the largest count of the four families above), and rerun with
  more repeats / a steadier node to get a wall-clock number worth quoting.
- No numerical code was changed in a way that affects the default path when
  `fuse_corner_fill=0`. Gate C2 is unaffected.
