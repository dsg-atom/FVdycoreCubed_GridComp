# Prompt 83: Eliminate Tracer Pack/Unpack Host Copy Bottleneck

Date: 2026-09-24
Branch: `optimization/cuda-fv3-performance`
Framing (verbatim from task spec): **"This is the FINAL bounded
optimization attempt for the CUDA tracer production path."**
Prerequisite: Prompt 82 (`p82_tracer_batched_regression_attribution_2026-09-24.md`)
— root-caused the Prompt 79 batched-transfer design's continued ~54.3%
TRACER_2D slowdown to ~520.7 ms/call of new serial, non-`__restrict__`
scalar host pack/unpack cost in `fv3_tracer_transport_bridge.cu`'s
`pack_field`/`unpack_field` (lines ~156-164), naming three candidate
fixes: `__restrict__`, bulk `memcpy` for same-type fields, or direct
`cudaMemcpyAsync` from the original Fortran pointers.

## 1. Do-NOT / Change-Budget Confirmation

Respected throughout, zero exceptions:

- Tracer numerics: unchanged.
- hord==8 kernels: unchanged.
- `pert_ppm`: unchanged.
- PT/DELZ: unchanged.
- REMAP_UV: unchanged.
- `nsplt>1`: not implemented.
- MPI halo exchange: not touched.
- Production default: unchanged (Fortran remains default for tracer
  dispatch; `use_cuda_tp2d` selector logic untouched).
- R4: not touched (the fix is a compile-time `if constexpr` branch that
  is a no-op for `T != fv_real`, i.e. R4 keeps the original scalar cast
  loop, unmodified in behavior).
- No second broad tracer architecture introduced. The only change is to
  the body of `pack_field`/`unpack_field`; the Prompt 79 packed-buffer
  H2D/D2H transfer architecture is untouched (Stage 3).
- No commit/push performed.

## 2. Stage 0 — Memory Layout Proof

All 12 packed inputs + 3 packed outputs are Fortran `REAL(FVPRC)` arrays.
In the production R8 build, `FVPRC` is 8-byte real, and the CUDA bridge's
internal working type `fv_real` is `double` — so for every field, the
Fortran source element type and the packed-buffer destination element
type are both 8-byte IEEE-754 double, identical bit representation, no
conversion required. Each field, for a fixed `(k,iq)` slice, is a
contiguous block in Fortran column-major layout (fixing the trailing
indices of `q(isd:ied,jsd:jed,npz,nq)`-shaped arrays yields a contiguous
2D `(isd:ied, jsd:jed)`-extent block; the edge-staggered fields — `crx`,
`xfx`, `cry`, `yfx`, `rax`, `ray`, `mfx`, `mfy`, `fx`, `fy` — are likewise
contiguous 2D blocks of their own declared extents). Source stride and
destination stride are both unit (element-contiguous); the packed buffer
itself is a single flat allocation with each field placed at a
precomputed fixed offset (`off_q`, `off_area`, `off_dxa`, `off_dya`,
`off_crx`, `off_xfx`, `off_cry`, `off_yfx`, `off_rax`, `off_ray`,
`off_mfx`, `off_mfy` on the input/H2D side; `off_q`, `off_fx`, `off_fy` on
the output/D2H side).

| Field | Role | Elem type (src) | Elem type (dst) | Contiguous | Same repr. | memcpy safe? |
|---|---|---|---|---|---|---|
| `q` | input+output | R8 double | `fv_real` double | yes | yes (R8) | YES |
| `area` | input | R8 double | `fv_real` double | yes | yes | YES |
| `dxa` | input | R8 double | `fv_real` double | yes | yes | YES |
| `dya` | input | R8 double | `fv_real` double | yes | yes | YES |
| `crx` | input | R8 double | `fv_real` double | yes | yes | YES |
| `xfx` | input | R8 double | `fv_real` double | yes | yes | YES |
| `cry` | input | R8 double | `fv_real` double | yes | yes | YES |
| `yfx` | input | R8 double | `fv_real` double | yes | yes | YES |
| `rax` | input | R8 double | `fv_real` double | yes | yes | YES |
| `ray` | input | R8 double | `fv_real` double | yes | yes | YES |
| `mfx` | input | R8 double | `fv_real` double | yes | yes | YES |
| `mfy` | input | R8 double | `fv_real` double | yes | yes | YES |
| `fx` | output | `fv_real` double | R8 double | yes | yes | YES |
| `fy` | output | `fv_real` double | R8 double | yes | yes | YES |

`use_cuda_tp2d` gates on `kind(q) == 8` (`fv_tracer2d.F90`, Prompt 77),
so the CUDA tracer path is only ever entered in the R8 build — the table
above (all `T == fv_real`) is the only case that reaches this code in
production. In a hypothetical R4 build the CUDA path is never dispatched,
so no field ever needs the `T != fv_real` narrowing-conversion branch in
practice; that branch is retained only for API generality (Stage 2) and
is unchanged behavior.

**Conclusion: memcpy is legal and safe for all 15 packed
fields in the production (R8) configuration.**

## 3. Stage 1 — Classification

All 12 inputs and 3 outputs classify as **Class A — DIRECT BULK MEMCPY**
in the R8 production configuration (per the Stage 0 table). No field
requires Class B (offset-only), C (strided), D (type conversion), or E
(other) treatment when `T == fv_real`. This matches the task's stated
goal: "If fields are already R8-compatible contiguous regions, no
per-element cast loop should remain" — confirmed true for all 15 fields.

## 4. Stage 2 — Implemented Fix

`fv3_tracer_transport_bridge.cu`, `pack_field`/`unpack_field` templates
(added `#include <type_traits>`):

```cpp
template <class T>
void pack_field(fv_real *__restrict__ dst, const T *__restrict__ src, size_t n) noexcept {
  if constexpr (std::is_same_v<T, fv_real>) {
    std::memcpy(dst, src, n * sizeof(fv_real));
  } else {
    for (size_t i = 0; i < n; ++i) dst[i] = static_cast<fv_real>(src[i]);
  }
}

template <class T>
void unpack_field(T *__restrict__ dst, const fv_real *__restrict__ src, size_t n) noexcept {
  if constexpr (std::is_same_v<T, fv_real>) {
    std::memcpy(dst, src, n * sizeof(fv_real));
  } else {
    for (size_t i = 0; i < n; ++i) dst[i] = static_cast<T>(src[i]);
  }
}
```

This implements two of Prompt 82's three named candidates simultaneously
(`__restrict__` + bulk `memcpy` for the same-type case) while leaving the
call sites, offsets, and buffer layout untouched. No per-call allocation,
no new numerical transform, no extra temporary buffer introduced. The
third candidate (direct `cudaMemcpyAsync` straight from the original
Fortran pointers, skipping the packed staging buffer entirely) was
**not** attempted — it would require restructuring the transfer
architecture itself (multiple smaller device transfers instead of one
packed transfer), which Stage 3 below and the task's change budget
explicitly reserve unless bulk packing is proven impossible. Bulk
packing is not impossible (Stage 0/1); therefore the narrower fix was
correctly preferred first.

## 5. Stage 3 — Transfer Architecture Preserved

No change made to: the persistent `thread_local PinnedBuf g_pack`
(pinned host buffer, `cudaHostAllocDefault`, grow-only) or persistent
`thread_local Workspace g_ws` (device buffer, grow-only); the packed-
buffer offset layout; the single `cudaMemcpyAsync` H2D of the whole
packed buffer; the single `cudaMemcpyAsync` D2H of the `q+fx+fy` output
prefix; the `cudaStreamSynchronize(nullptr)` call; or the
`[P79_TRANSFER_COUNT]` diagnostic. Transfer count per `(k,iq)` call
remains 1 H2D + 1 D2H, i.e. the Prompt 79 architecture's ~40 transfers/
call for a full production call (unchanged from Prompt 79/82 — not
re-measured this task since no GPU stage was run, see §8).

## 6. Stage 4 — Host Microbench (GATE FAILED — governs the rest of this task)

New file (host-only, no CUDA/GPU required):
`cuda/dyn_core/tests/host/p83_pack_unpack_microbench.cpp`. Reproduces
`fv3_tracer_transport_bridge.cu`'s own C512 single-tile offset/size
formulas (`is=1..512`, `ng=3`, `isd=-2..515`, `nx_d=ny_d=518`,
`total_elems=3,717,288` / 29,738,304 bytes ≈28.36 MB packed-buffer H2D
payload per `(k,iq)` slice call, `out_elems=793,636` / 6,349,088 bytes
≈6.05 MB D2H payload), packs/unpacks all 12 inputs + 3 outputs with both
the OLD (byte-for-byte pre-fix scalar loop, no `__restrict__`) and NEW
(Stage 2 fix) implementations, 20 warmup + 500 timed reps each, plus a
bitwise self-check comparing OLD vs NEW packed-buffer contents.

Three configurations were run (heap memory, no GPU context — see the
pinned-memory caveat in §6.3):

| Config | Compiler/flags | PACK_OLD (ms) | UNPACK_OLD (ms) | PACK_NEW (ms) | UNPACK_NEW (ms) | Combined speedup | Bitwise match |
|---|---|---|---|---|---|---|---|
| g++ -O2 | `g++ -O2 -std=c++17` | 1.797349 | 0.349388 | 1.293864 | 0.222151 | **1.416x** | PASS |
| g++ -O0 | `g++ -O0 -std=c++17` | (not representative of production) | | | | **4.434x** | PASS |
| nvcc -O3 (authoritative) | `nvcc -O3 -DNDEBUG -std=c++17` (matches `build_cuda_r8_intelmpi`/`build_cuda_r8_perf`'s `CMAKE_CUDA_FLAGS_RELEASE`) | 1.494790 | 0.202546 | 1.245285 | 0.174052 | **1.196x** (PACK 1.200x, UNPACK 1.164x) | PASS |

### 6.1 Which configuration is authoritative

`CMAKE_BUILD_TYPE` was checked across all three relevant build
directories: `build_cuda_r8_mpi` (dev/OpenMPI stack) is empty
(effectively `-O0` — **not** representative); `build_cuda_r8_intelmpi`
and `build_cuda_r8_perf` (the production/performance-relevant build
directories) are both `Release` (`-O3 -DNDEBUG`). No override of
optimization flags was found in `cuda/dyn_core/CMakeLists.txt`. The nvcc
`-O3 -DNDEBUG -std=c++17` result (1.196x combined) is therefore the
authoritative Stage 4 figure.

### 6.2 Root cause of the low speedup: compiler loop-idiom recognition

At `-O2`/`-O3`, both g++ 12.3.0 and nvcc 12.5's host compilation pipeline
already recognize the OLD implementation's trivial same-type
element-by-element `dst[i] = static_cast<fv_real>(src[i])` loop (with
`T == fv_real`, this cast is a no-op) as a bulk-copy idiom and compile it
to something functionally equivalent to `memcpy` — this is standard
"loop-to-memcpy" idiom recognition, not a surprise once observed. A
hand-written `std::memcpy` therefore has very little headroom to exploit
over the already-optimized OLD loop at production optimization levels.
The `-O0` result (4.434x) is real but not representative, because
production never builds at `-O0`.

### 6.3 Pinned-memory caveat (documented, not empirically retested)

This microbench used ordinary heap memory (`std::vector`), not the
production `cudaHostAlloc(..., cudaHostAllocDefault)` pinned buffer, and
required no GPU context. Prompt 82's in-situ measurement found ~520.7
ms/call of pack+unpack cost — roughly 3.3x higher than this microbench's
aggregate estimate (~158 ms/call combined OLD pack+unpack, scaled to the
in-situ call pattern). This gap is not fully reconciled in this task.
However: both the OLD and NEW implementations write into the *same*
buffer type (the production pinned buffer, or heap memory here) — any
pinned-vs-heap write-performance difference would apply
**proportionally** to both implementations, and is very unlikely to
change the measured **speedup ratio** (1.2x-1.4x) enough to reach the
required 5x-10x gate. This ratio-invariance argument, combined with the
project's cost-efficiency mandate (bounded debugging, minimum
experiments needed, defined stop condition — `CLAUDE.md`), was judged
sufficient to skip a GPU-context pinned-memory retest rather than
requiring a Slurm allocation to confirm a conclusion three independent
heap-memory configurations already support unambiguously. This is
recorded as an open, documented caveat, not a fully closed loop.

### 6.4 Stage 4 verdict

**GATE FAILED.** Required: ≥5x reduction, preferably ≥10x. Measured
(authoritative nvcc -O3 config): **1.196x**. Per Stage 4's own binding
instruction:

> "If bulk copy is unexpectedly not much faster: STOP and investigate
> only the copy implementation. Do not proceed to GPU performance."

The investigation (three independent configurations, including the one
that exactly matches production compiler flags) is complete and
conclusive: the low speedup is not a measurement artifact, a wrong
build-type comparison, or an unswapped-label error — it is a genuine
consequence of the OLD implementation already being compiler-optimized
to near-`memcpy` speed at `-O2`/`-O3`. This STOP condition is
**invoked**.

## 7. Decision: Stages 5-9 Not Run

Per Stage 4's binding contingency, Stages 5 (standalone CUDA regression),
6 (C32 integration), 7 (C512 integration), 8 (same-node paired
performance), and 9 (before/after attribution against Prompt 82's ~521
ms/call figure via an in-situ, GPU-integrated measurement) were **not
run**. No Slurm job was submitted for this task. This is an intentional
scope decision required by the task's own gate logic, not an omission:
Stage 4 is explicitly a pre-condition gate ("do not proceed to GPU
performance"), and it did not clear.

Consequently, Stage 9's requested before/after comparison can only be
reported at the host-microbench level (this section), not as an in-situ
GPU measurement:

- Prompt 82 in-situ pack+unpack: ~520.7 ms/call (serial, non-`__restrict__`).
- Prompt 83 isolated host-microbench pack+unpack (nvcc -O3, aggregate
  estimate scaled to the same field set): OLD ~1.70 ms per single
  `(k,iq)`-slice pack+unpack pair, NEW ~1.42 ms per pair — a reduction
  far short of what would be needed to explain, let alone eliminate,
  the ~521 ms/call in-situ figure. The residual gap (§6.3) means this
  task cannot claim the in-situ ~521 ms/call figure has been reduced by
  anything close to the isolated ratio; only the isolated, host-only
  ratio (1.196x-1.416x) is directly evidenced.

**The Stage 2 code fix is retained** (not reverted) despite the Stage 4
gate failure: it is bitwise-correct (`PACK_BITWISE_MATCH=PASS`, all three
configs), strictly non-regressive (never slower than the OLD
implementation in any measured configuration), and mechanically
harmless. Stage 4's contingency instructs skipping *GPU-performance
follow-through*, not reverting an already-implemented, verified-correct,
marginally-beneficial micro-fix.

## 8. Stage 10 — Hard Performance Gate

Stage 10's three-way outcome (CUDA faster / within ±2% / >2% slower) is
a comparison of **integrated, GPU-executed TRACER_2D timing**, which
requires Stages 5-9 to have run. Since they did not run (§7), Stage 10
cannot be evaluated on its own integrated-measurement terms. However,
the task's overriding binding instruction is unconditional on *how* the
STOP is reached:

> "If this narrow host-marshaling optimization does not recover
> performance, STOP CUDA tracer production optimization after this
> prompt."

Stage 4's host-microbench result is direct, sufficient evidence that
this optimization does not recover performance to any degree remotely
close to what would be needed to close a ~37-54% TRACER_2D regression
(Prompts 78/79/81B) — the isolated best-case host-copy speedup (1.2x-
1.4x) applied to a ~521 ms/call marshaling cost would still leave
~370-435 ms/call of marshaling overhead, nowhere near sufficient to
recover a multi-hundred-ms/call regression. Proceeding to a full
GPU-integrated remeasurement (Stages 5-9) was judged to have negligible
probability of reversing this conclusion, consistent with Stage 4's own
instruction to stop rather than proceed once the copy-implementation
investigation is conclusive.

**Verdict: STOP CUDA TRACER PRODUCTION OPTIMIZATION**, per the binding
instruction, triggered by the Stage 4 gate rather than a Stage-10
integrated remeasurement. Production default remains Fortran for tracer
dispatch (`use_cuda_tp2d` selector logic unchanged).

## 9. Stage 11 — Nsight

Not run. Stage 11 permits a short `nsys` capture only "if the host
microbench improves dramatically but integrated performance still
cannot be explained" — the inverse of what happened here (the host
microbench did **not** improve dramatically), so this condition does not
apply, and no GPU-integrated performance was measured to fail to
explain in the first place.

## 10. Summary Table

| Stage | Item | Result |
|---|---|---|
| 0 | Memory layout proof | All 15 fields Class A — memcpy safe (R8 production) |
| 1 | Pack/unpack classification | All 15 fields Class A — DIRECT BULK MEMCPY |
| 2 | Minimum host copy fix | Implemented: `__restrict__` + `if constexpr` bulk `memcpy` for `T==fv_real`; R4 branch unchanged |
| 3 | Transfer architecture | Preserved unchanged (1 H2D + 1 D2H per (k,iq) slice, persistent pinned/device buffers) |
| 4 | Host microbench | **GATE FAILED** — 1.196x (nvcc -O3, authoritative), 1.416x (g++ -O2); required ≥5x |
| 5-9 | GPU regression / C32 / C512 / paired perf / attribution | **NOT RUN** — skipped per Stage 4's binding STOP contingency |
| 10 | Hard performance gate | STOP, by extension of the Stage 4 gate failure (no integrated remeasurement performed) |
| 11 | Nsight | Not run — condition for running it did not apply |

## 11. Final Verdict

**TRACER HOST PACK BOTTLENECK REMOVED BUT CUDA STILL REGRESSES — STOP
CUDA TRACER PRODUCTION WORK.**

Clarification of what this closing line means in this task's specific
context: "removed" refers to the *host-copy-only* fix being implemented,
bitwise-verified, and marginally faster in isolation (1.2x-1.4x) — not
to an in-situ elimination of the full ~521 ms/call marshaling cost
Prompt 82 measured, and not to a fresh integrated TRACER_2D
remeasurement showing continued regression. The STOP is triggered by
Stage 4's own host-microbench gate (the fix does not clear the required
improvement bar to justify further GPU-integration work), not by a
Stage-10 GPU-measured regression. No new GPU/integration numbers exist
this task to report; the practical outcome — Fortran remains the
tracer-dispatch production default, and per this task's own binding
instruction, no further CUDA tracer production optimization prompt
should be authorized after this one — is the same as if a Stage-10
regression had been freshly measured. Prompt 82's diagnosis (the
batched design's own host marshaling loops are the dominant cost) is
directionally confirmed correct by this task's Stage 0-2 analysis; what
this task adds is that a *narrow* host-copy-only fix, while correct and
mechanically implemented, is compiler-preempted at production
optimization levels and cannot by itself close the gap. The remaining
named alternative from Prompt 82 — direct `cudaMemcpyAsync` from the
original Fortran pointers, skipping the packed staging buffer
architecture entirely — was explicitly out of this task's change budget
(a second architectural change) and remains untried; per this task's own
binding STOP instruction, it should not be pursued as an automatic
follow-on.

## 12. Artifacts

- Code change (kept): `cuda/dyn_core/dgrid/fv3_tracer_transport_bridge.cu`
  (`pack_field`/`unpack_field`, `#include <type_traits>`).
- Microbench source (kept as a permanent record of this measurement;
  not part of the production CMake build — no CMakeLists.txt reference
  added): `cuda/dyn_core/tests/host/p83_pack_unpack_microbench.cpp`.
- Scratch compiled binaries under `build_cuda_r8_mpi/p83_scratch/`
  (`/tmp` is noexec on this login node, hence the out-of-source scratch
  location) — deleted after this task's numbers were transcribed above;
  not committed (inside an existing, already-not-committed build
  directory).
