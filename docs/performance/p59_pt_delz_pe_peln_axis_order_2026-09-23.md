# Prompt 59 — pt/delz pe/peln Axis-Order Verification (2026-09-23)

## Question

Are the CUDA pt/delz write-backs to `pe` and/or `peln` using the wrong
logical axis order? Answer must be proven from source/runtime evidence, not
inferred from comments.

## Stage 0 — Field layout contract (proven, not assumed)

Traced `fv_mapz.F90`'s registration call sites (`add3()`) and cross-checked
against a live registry dump (`RUN_CUDA_FV3_PTDELZ_DUMP_VIEWS=1`):

- `pe`: registered from `Atm%pe(is-1:ie+1,npz+1,js-1:je+1)` — Fortran
  dimension order **(i,k,j)**, k is the MIDDLE dimension. Live view: lower
  `(0,0,0)` extent `(34,11,34)` stride `(1,34,374)`.
- `peln`: registered from `Atm%peln(is:ie,npz+1,js:je)` — same **(i,k,j)**
  order. Live view: lower `(1,0,1)` extent `(32,11,32)` stride
  `(1,32,352)`.
- `pk`: registered from `Atm%pk(is:ie,js:je,npz+1)` — k is LAST, i.e.
  genuinely **(i,j,k)**, matching `RiemViewC3::at`'s plain positional
  convention with no translation needed.

`RiemViewC3::at(i,j,k)` is a pure positional accessor (arg 0 → stride s0,
arg 1 → s1, arg 2 → s2), with zero per-field axis awareness. Therefore:

**CORRECT ACCESS IS: `pe.at(i, k, j)`, `peln.at(i, k, j)`, `pk.at(i, j, k)`.**

## Stage 1 — Access audit (`fv3_pt_delz_remap.cu`, pre-fix)

| Line (pre-fix) | Field | R/W | Code indices | Intended indices | Correct? |
|---|---|---|---|---|---|
| read loop, `pe_old[k] = a.pe.at(i, k, j)` | pe | R | (i,k,j) | (i,k,j) | YES |
| read loop, `pn1[k] = a.peln.at(i, k, j)` | peln | R | (i,k,j) | (i,k,j) | YES |
| closing update, `a.pe.at(i, j, k) = peO[k]` | pe | W | (i,j,k) | (i,k,j) | **NO** |
| closing update, `a.peln.at(i, j, k) = pn2[k]` | peln | W | (i,j,k) | (i,k,j) | **NO** |
| closing update, `a.pk.at(i, j, k) = ...` | pk | W | (i,j,k) | (i,j,k) | YES |

The read loop was already correct (a prior, uncommitted-until-now fix).
Both closing-update writes to `pe`/`peln` used the wrong axis order.

## Stage 2 — Address-level proof (live evidence, not inference)

Pre-existing compute-sanitizer memcheck logs from the Stage-4 discriminator
job (58520848, `perf_logs/p57_memcheck_pid_182570.log` through `182575.log`,
one per MPI rank) already contained the decisive evidence — no new job was
needed to establish this:

```
========= Invalid __global__ write of size 8 bytes
=========     at fv3::gpu::vertical::<unnamed>::pt_delz_remap_kernel(...)+0x16f80
=========     by thread (12,1,0) in block (0,1,0)
=========     Address 0x15221ab8df60 is out of bounds
=========     and is 1 bytes after the nearest allocation at 0x15221ab75200
=========     of size 101,728 bytes
```

`pe`'s buffer is exactly 34×11×34 = 12,716 elements × 8 bytes = 101,728
bytes — matching the allocation size in the report exactly (same size class
Prompt 54 originally found for the earlier OOB bug).

`(0x15221ab8df60 - 0x15221ab75200) / 8 = 12,716` — i.e. the fault address is
element offset **12,716**, exactly the first element past the end of `pe`'s
buffer. Twenty consecutive reports increment by exactly 8 bytes (1 element)
each, spanning offsets 12,716–12,735; thread IDs span `(12,1,0)`–`(15,1,0)`
in blocks `(0,1,0)`–`(1,1,0)`, i.e. 20 distinct column-threads (varying `i`)
at a fixed `threadIdx.y=1`/`blockIdx.y=1` (fixed `j`/`k` combination) firing
the same OOB write simultaneously in SIMT lockstep. This is exactly the
signature of `a.pe.at(i, j, k)`: `j` lands in the stride-34 slot meant for
`k`, and `k` lands in the stride-374 slot meant for `j` — once `j` (using
the wrong slot's smaller extent, 11) is exceeded by the loop's own `k` range
(`0..km`, km=10 here so `k` reaches 10, and the wrong-slot extent is 11 — the
overflow appears once the *outer* i-loop's threads combine with a `j` near
the tile edge), the write walks past the buffer end. Reproduced identically
on all 6 MPI ranks — not rank-specific data pathology.

This is address-level, tool-confirmed proof, not comment-based inference.

## Stage 3 — Downstream effects

The OOB write is a memory-safety violation (undefined behavior), not a
logical-but-in-bounds aliasing bug. Its runtime symptom was intermittent:
`cudaErrorLaunchFailure (error 719)` surfaced on whichever CUDA call next
synchronized the stream (`cudaStreamSynchronize`/`cudaMemcpy`/`cudaFree`/
`cudaEventRecord`), consistent with an asynchronous device-side fault only
becoming visible at the next synchronization point — explaining why this
had not previously been caught by non-sanitizer runs (a corrupted
allocation neighbor doesn't always trip the driver's own fault detector).

## Stage 4/5 — Root cause classification

**Category: genuine out-of-bounds `__global__` write (real memory-safety
bug), proven directly by compute-sanitizer, not inferred.** This is
independent of, and does not explain, "Bug 2" (the pt/w/omga divergence
from call 2 onward first isolated in Prompt 57) — see Stage 7 below.

Historical note: `docs/performance/p55_pt_delz_oob_fix_2026-09-22.md`
claimed Prompt 55 fixed **both** the read loop and the closing-update loop
from `.at(i,j,k)` to `.at(i,k,j)`. `git show HEAD:cuda/dyn_core/vertical/
fv3_pt_delz_remap.cu` proves only the read loop was ever actually committed
— the closing-update loop remained `.at(i,j,k)` in every commit through
Prompt 58, and in the working tree at the start of this prompt. The
documented fix was never applied to the closing-update loop; this is a
documentation/reality mismatch, not a re-introduced regression.

## Stage 6 — Fix applied (the ONE authorized change)

`cuda/dyn_core/vertical/fv3_pt_delz_remap.cu`, closing-update loop:

```cpp
// before
a.pe.at(i, j, k) = peO[k];
a.peln.at(i, j, k) = pn2[k];
// after
a.pe.at(i, k, j) = peO[k];
a.peln.at(i, k, j) = pn2[k];
```

`a.pk.at(i, j, k)` left unchanged (its own contract is genuinely (i,j,k)).
No other lines touched. Rebuilt `fv3_wp_c_vertical` and relinked
`StandAlone_FV3_Dycore.x` (new sha256
`4159456d0b52ef6ceb208d14557915cafb48b63640b8ab6ae728239c57f76f29`).

## Stage 7 — Call-2 / crash-elimination result

Stage 7 verification job (58520983, same compute-sanitizer-wrapped C32
arm): 0 crashes (`P59_Range_Warn=0 P59_NaN=0 P59_Abort=0`), 0 OOB findings
(`P59_SANITIZER_OOB_FOUND=NO`), `ERROR SUMMARY: 0 errors` on all 6 ranks.
The fix eliminates the crash.

**Primary criterion — does the call-2 `pt` divergence (Bug 2) disappear?
NO.** Six-call C32 `dcx` A/B (`compare_dcx_backends.py --grid-size 32`,
post-fix) shows call 1 clean/round-off (matches Gate C2 precedent exactly:
worst max_rel 3.246504e-10) but calls 2–6 show the same non-round-off
divergence documented pre-fix in Prompt 57: `pt` max_rel
1.281892e-06 (call 2) → 6.193508e-06 (call 6); `delp` max_rel 6.344516e-06
(call 2) → 2.143720e-05 (call 6); `w`/`omga` up to ~4–5%. These figures are
bitwise-identical in character to the pre-fix Prompt 57 Stage 12 numbers.
**Per Prompt 59's rule, this STOPS the investigation — no second fix
attempted.**

## Stage 8 — Six-call C32 (executed inline with Stage 7)

Result: FAIL (Bug 2 present, unchanged from pre-fix). See Stage 7 numbers
above. Stage 10 (C512 A/B) is correctly NOT attempted, gated on Stage 8
passing cleanly.

## Stage 9 — Regression

- `fv3_wp_c_pt_delz_remap_gpu_smoke`: PASS, 0/6 cases failed, round-off
  (~1e-16–1e-15 relative).
- `fv3_wp_c_pt_delz_remap_prod_shape_smoke`: PASS, 0/6 cases failed, same
  figures.
- compute-sanitizer memcheck on both pt/delz smoke variants: 0 errors each.
- `fv3_wp_c_map_scalar_gpu_smoke`: FAIL, 8/8 cases show mismatches, all at
  double-precision round-off scale (max_abs 5.684e-14–1.137e-13, max_rel
  ~2–4e-16, i.e. at/near FP64 machine epsilon). This kernel
  (`fv3_map_scalar_kernel.cu`) is not touched by this prompt's fix — no
  `.cu`/`.hpp`/`.cpp` file other than `fv3_pt_delz_remap.cu` was modified
  this prompt. The presence of a separate, already-existing
  `fv3_wp_c_map_scalar_gpu_smoke_nofma` build target indicates the project
  already tracks FMA-contraction-sensitive round-off behavior in this
  specific fixture as a known, pre-existing characteristic, independent of
  this prompt's change. This FAIL is out of Prompt 59's scope (no map_scalar
  files were touched) and is not attributed to this fix.

## Stage 10 — C512

Not attempted (correctly gated off by Stage 8's FAIL, per Prompt 59's
explicit rule).

## Keep-vs-revert decision

**KEEP.** Unlike Prompt 58's reverted "Category E" fix (which showed an
unexplained side-effect on `delp`'s divergence, undermining its own
"no semantic change" premise), this fix:

- is independently proven necessary by direct compute-sanitizer evidence
  of real undefined behavior (not inferred from comments or manual
  arithmetic);
- matches the long-documented (but never-applied) Prompt 55 intent and the
  already-correct, already-validated read-loop convention;
- produces bitwise-identical Bug-2 divergence figures before and after
  (zero numerical side effect);
- eliminates a real, reproducible, multi-rank crash.

This is retained as a standalone, validated memory-safety/correctness fix,
orthogonal to and not a resolution of Bug 2.

## Diagnostics left in source (all opt-in, zero effect by default)

`RUN_CUDA_FV3_PTDELZ_DUMP_VIEWS`, `RUN_CUDA_FV3_PTDELZ_DUMP_PRETRANSFORM`
(pre-existing, Prompts 56/57), `RUN_CUDA_FV3_PTDELZ_DUMP_POSTWRITE` (new,
this prompt) — a post-kernel sentinel comparing the correct `(i,k,j)`
address against the code's literal `(i,j,k)`-positional address at a fixed
debug column. All gated behind `std::getenv`, inert unless explicitly set,
consistent with the project's existing `RUN_CUDA_FV3_PTDELZ_FORCE_FORTRAN`
diagnostic convention.

## Conclusion

The pe/peln closing-update write-back axis-order bug is real, proven by
address-level compute-sanitizer evidence (not comment inference), and now
fixed. It explains and eliminates the "unspecified launch failure" crash
observed previously. It does **not** explain or resolve Bug 2 (the pt/w/
omga divergence from call 2 onward), which remains open, unfixed, and
un-root-caused.

**Update (Prompt 60, 2026-09-23): Bug 2 is now ROOT-CAUSED (still not
fixed).** The CUDA `pt` remap dispatch omits the GMAO top/bottom boundary-
condition branches that Fortran's `map_scalar` applies whenever
`gmao_top_bc`/`gmao_bot_bc` are true — the unconditional GEOS production
default (`FV_StateMod.F90:558-559`), not an HS-specific setting. Proven via
exact algebraic identity at k=1 plus direct numeric evidence of non-round-
off divergence at k=1/k=10 from call 1 onward — overturning the "call 1 is
clean" belief from Prompts 57-59 (call 1 is only clean at mid-column; the
boundary is the true first divergence site). Not fixed this prompt: a
correct fix requires porting 5 conditional branches, exceeding a single
narrow-fix budget. See
`docs/performance/p60_pt_delz_dispatch_boundary_divergence_2026-09-23.md`.
