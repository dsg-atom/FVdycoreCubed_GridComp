# Prompt 18 — Six-rank MPI halo transport determinism

Branch `cuda/gpu-bringup`; HEAD at start `eb6d689` (working tree carries the
uncommitted Prompt 15/16/17 work). Platform: Discover `gpu_a100`, `rome`,
1 node / 6 MPI ranks / 1 A100, OpenMPI 4.1.6 + GCC 12.3.0 + CUDA 12.5, R8.

**Result: the defect is a CUDA stream-ordering bug in the TRANSPORT TEST's own
device-buffer setup, not in the MPI transport, and not an MPI contract
violation.** It is fixed, the mechanism is proven by both observation and
intervention, and the six-rank transport test now passes **24/24 consecutively**
under OpenMPI with `checks: 16692  failures: 0` every run. R8 CTest is
**23/23 PASS** (it was 22/23, the one failure being this test).

---

## 1. Three recorded premises were wrong, and correcting them mattered

Each of these was carried into this pass from `CURRENT_STATE.md` or the task
prompt itself. All three are now corrected on measurement.

| recorded premise | measured reality |
|---|---|
| "Intel MPI: 5/5 PASS" (task prompt); OpenMPI is the broken stack | **Inverted.** Intel MPI 2021.13 = **5/5 FAIL** (jobs 58244300, 58244365); OpenMPI 4.1.6 = **3/5 PASS / 2/5 FAIL** (job 58244385). Intel MPI was the *higher-hit-rate reproducer*, never a clean reference. |
| "the failing value is always exactly 0.0 (unwritten)" | **Not established, and false here.** A `0xFF`-filled R8 word is NaN; `std::llround(NaN)` → `LLONG_MIN`, which decodes through the oracle to `tile 0 (-8,-8,0)` — *textually identical* to a genuine 0.0. The old output could not tell them apart. With the new `classify()`, every observed failure is **POISON-REMAINS**, not zero. |
| "a pre-existing intermittent *transport* defect"; blast radius unknown, possibly one cell | **It is a TEST-HARNESS defect**, and the blast radius is an entire message: **72 of 72 written cells** of one 105-element slot, with the slot census reading **poison 105 / zero 0 / other 0**. Not one cell, not a short prefix. |

The "one cell" reading was an artifact of the test's own
`goto next_field` suppression (`test_halo_mpi_transport.cu:328`), which reported
only the first failure per `(edge, field)` and skipped the rest — which is also
exactly what produced the historical check-count drops (16692 → 16621/16609/
16538). With the suppression removed, **failing runs report the full 16692
checks too**, and the drops are fully explained.

---

## 2. Root cause

`test_halo_mpi_transport.cu` creates its exchange stream with
`cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)`, which by definition
does **not** implicitly synchronize with the legacy default stream. Two pieces of
the test's own setup were nevertheless issued on the legacy stream:

1. **`cudaMemset(send/recv, 0xFF, …)`** — the §6 poison fill. A plain
   `cudaMemset` on *device* memory is asynchronous with respect to the host and
   is issued on the legacy stream. It was therefore free to land **after** the
   transport's staging H2D (`cudaMemcpyAsync(recv, h_recv_, used_bytes, …, cs)`)
   and before the unpack read that slot — re-poisoning correctly-arrived data.
2. **`TileArray::create()`'s `cudaMemcpy(dev, host.data(), …, H2D)`** — a
   *synchronous* `cudaMemcpy` H2D from **pageable** memory returns once the
   source has been staged for DMA, not once the device write has landed, and it
   too lands in the legacy stream. So `launch_halo_pack` could read the field
   array before it was populated. Fresh `cudaMalloc` memory commonly reads as
   0.0, and 0.0 decodes to `tile 0 (-8,-8,0)` — the historical signature.

Both are pure test-code hazards. **The transport itself does not have this
class of bug**: its two buffer clears are `std::memset(h_send_/h_recv_, 0, bytes)`
on **host pinned** memory at attach (`fv3_gpu_halo_mpi.cpp:137-138`) — no CUDA
stream is involved at all.

### The observed signature (job 58246330, repeat 2, `warpa011`)

```
[rank 2] FAIL: tile 3 edge N field 0 h=1 a=1 k=1
      got  tile 0 (-8,-8,0) sign +1  [POISON-REMAINS (0xFF; device recv slot
                                      not overwritten by the H2D)]
[rank 2] BLOCK FAIL 72/72 cells | group 7 H_DSCALAR (q_con,delp,pt,cappa;
      4 fields, width 3) | my tile 3 edge N field 0 <- donor tile 5 rank 4
      edge W slot 0 | tag 112 | my recv slot offset 315 of 1680 elems,
      slot 105 elems | slot census: poison 105, zero 0, other 0
```

Reading it against §1's discriminators:

- **A full field/edge message block is missing** — all 72 written cells
  (`width 3 × along 6 × nk 4`) of a 105-element slot
  (`3 × (max(nx,ny)+1 = 7) × (npz+1 = 5)`). Not one cell, not a prefix.
- **The whole slot still holds the test's own poison** — so nothing overwrote
  it late; the poison arrived late.
- **Offsets are correct**: 315 = 3 × 105 (field 0, edge N) and, in repeat 6,
  105 = 1 × 105 (field 0, edge E). This *empirically* confirms the §7 slot
  arithmetic rather than merely reasoning about it.
- **The afflicted `(edge, field)` moves between runs** (repeat 2 = N, repeat 6 =
  E) — a race, not a static mis-mapping.
- **Zero `OWNED-BOX MOVED` lines.** My own prior inference that the failure came
  from the `"owned cells outside this group's destinations are untouched"`
  assertion was **wrong**; every failure came from the decode loop. Recorded
  because the inference is a plausible one to make again.

### Proof by intervention

Same binary, same node, same allocation, the only difference being whether the
test's device-buffer setup is stream-ordered (`FV3_HALO_TEST_LEGACY_SETUP`):

| job | setup | repeats | result |
|---|---|---|---|
| 58246302 A | legacy (unordered) | 12 | **6 FAIL** (84, 156, 72, 84, 72, 72 failures) |
| 58246302 B | stream-ordered | 24 | **24 PASS**, 16692 / 0 each |
| 58246330 | legacy, full diagnostics | 12 | **6 FAIL** — the `BLOCK FAIL` lines above |
| 58246343 A | stream-ordered, post-fix binary | 24 | **24 PASS**, 16692 / 0 each |
| 58246343 B | legacy control, post-fix binary | 6 | **3 FAIL** (84, 72, 324) — defect still reproducible on demand |

This is why the failure appeared under *both* MPI stacks at different rates: the
race is in CUDA stream ordering, so the MPI implementation only shifts the
timing, never the semantics. **Classifying it as an OpenMPI bug would have been
wrong**, and so would an OpenMPI-specific workaround.

---

## 3. What the audits cleared (§3, §4, §5, §7)

Recorded so the next pass does not redo them.

- **§3 pack → D2H ordering: CORRECT.** `fv3_gpu_runtime.cpp:912-948` runs
  `launch_halo_pack` → `tr->transfer` → `launch_halo_unpack` all on
  `d.transfer_stream`; inside `transfer()` the D2H is `cudaMemcpyAsync` on that
  same stream, followed by `cudaEventRecord`/`cudaEventSynchronize` on it before
  any `MPI_Isend`. Stream-ordered by construction.
- **§4 request storage: NOT the bug, and the leading hypothesis is retired.**
  `req` is `std::array<MPI_Request, 2*4*max_halo_group_fields>` (32 slots),
  value-initialized `{}` — **not** a `std::vector`. `req.data()` cannot
  reallocate, only the `nreq` written entries reach `MPI_Waitall`, and
  `nreq` is incremented exactly once per posted request. Do not re-pursue
  "vector reallocation after `MPI_Isend`".
- **§5 message matching: symmetric and collision-free.** Send tag is
  `message_tag(group, mine, n)`; the matching recv tag is
  `message_tag(group, nb.edge, sn)` with `sn = field_slot(spec, fs.partner)`
  for reversed non-scalar fields. Within one exchange every recv has a distinct
  `(source, tag)`; across exchanges, MPI non-overtaking plus exactly one post
  per `(peer, tag)` per exchange gives correct pairing. Max tag 207, inside the
  guaranteed `MPI_TAG_UB ≥ 32767`.
- **§7 counts/offsets: airtight, and now also confirmed empirically.**
  `group_slot_elements(g,w) = w*(max(nx,ny)+1)*(npz+1)`;
  `slot_base(g,w,field,e) = (field*4+e)*group_slot_elements`;
  `group_buffer_elements = field_count*4*group_slot_elements`. The maximum slot
  base plus one slot equals `group_buffer_elements()` equals the byte count the
  H2D overwrites, so **every slot lies inside the copied region** — no transport
  offset or count error can leave a stale cell there. Measured group geometry
  from the `[dbg]` lines: `width=3 slot=105 buffer=420/840/840/1680` (1/2/2/4
  fields) and `width=1 slot=35 buffer=280` (2 fields).

**MPI contract violation: NO.** Nothing in the transport modifies a send buffer
before completion, overlaps send/recv regions, relies on same-tag ordering, or
lets request storage move while requests are active.

---

## 4. Fixes applied

### 4.1 Test instrumentation and stream ordering — `cuda/dyn_core/tests/integration/test_halo_mpi_transport.cu`

Explicitly allowed by §9 ("test instrumentation"), so **not** counted against
the 3-substantial-fix budget.

- The two poison memsets → `cudaMemsetAsync(..., stream)`; `TileArray::create()`
  takes the stream and uses `cudaMemcpyAsync(..., stream)`. **This is the fix.**
- `FV3_HALO_TEST_LEGACY_SETUP=1` retains the racy form *on purpose*, so the
  defect can be demonstrated rather than asserted. Default is the ordered path.
- Observability (§1/§6): the `goto next_field` suppression is removed and every
  cell is counted; `classify()` distinguishes **POISON-REMAINS** / **ZERO** /
  **WRONG-VALUE** from the raw bit pattern; a per-block summary reports rank,
  group, my tile/edge/field, donor tile/rank/edge/slot, tag, recv slot offset and
  size, and a poison/zero/other census of the whole slot; the owned-box check
  reports the first offending index with want/got decode plus a total moved-cell
  count instead of early-exiting on a boolean. Per-cell prints have their own
  budget (`may_print_cell()`, cap 24) so a bulk-failing block cannot starve the
  per-block summary — that starvation is why the first attempt saw 72–156
  failures with no failure text.
- `FV3_HALO_TEST_DEBUG=1` enables the per-cell detail and the geometry line.
  Neither variable is set in the CTest registration, so CTest exercises the
  ordered, quiet path.

### 4.2 Production hardening — `cuda/dyn_core/runtime/fv3_gpu_runtime.cpp:1290-1291`

**Counts as 1 of the 3 permitted substantial transport fixes** (§9 category
"synchronization / H2D–D2H ordering").

`attach_halo_transport`'s one-time slot-tail clear was
`cudaMemset(send/recv, 0, bytes)` — the *same* legacy-stream hazard, one layer
over in the production runtime. It was only *probably* ordered before the first
pack (two MPI collectives and a lot of host work intervene), never *provably*
so. Now `cudaMemsetAsync(..., d.transfer_stream)`, i.e. the same stream every
pack, staging transfer and unpack runs on — a stream guarantee, at zero cost.
The clear's purpose (silencing 24587 benign initcheck reports on the never-read
slot tail, milestone 3A.10 §20) is unchanged.

**Numerical algorithm changed: NO.** No halo semantics, no edge rotations or
signs, no topology, no CUDA numerical kernel, no change to the transport
abstraction.

---

## 5. Acceptance evidence

- **§10 determinism, OpenMPI 4.1.6:** job 58246343, node `warpa011` —
  **24 / 24 consecutive PASS**, `checks: 16692  failures: 0` on every run, no
  poison remaining, no MPI errors, no CUDA errors. (Job 58246302 Phase B
  independently gave 24/24 on the pre-hardening binary.)
- **Legacy control on the same post-fix binary:** 3 of 6 FAIL — the defect is
  still reproducible on demand, which is what makes the 24/24 attributable to
  the ordering change rather than to luck.
- **R8 regression:** full CTest in the same job — **23/23 PASS, 0 failed**
  (previously 22/23, the failure being `fv3_halo_mpi_transport`).
- **Intel MPI 2021.13 comparison reference:** job 58246352 — **5/5 PASS**, was
  **5/5 FAIL** pre-fix. See §7.1.
- **§11 standalone CUDA short run:** job 58246353 — 6 dyn_core calls, entry
  BITWISE 6/6, exit 102/102 ROUND-OFF or SMALL, `delp` positive, zero NaN/Inf.
  See §7.2.

---

## 6. Reproduction recipe

```bash
# Build (OpenMPI, R8, MPI halo ON)
module purge; module use -a /discover/swdev/gmao_SIteam/modulefiles-SLES15
module load GEOSenv comp/gcc/12.3.0 mpi/openmpi/4.1.6/gcc-12.3.0 \
            nvidia/nvhpc-byo-compiler/24.7
cmake --build build_cuda_r8_mpi -j 12

# Determinism acceptance + legacy control + full CTest, one job
sbatch p18_accept.sbatch          # -> p18_accept_<jobid>.out

# Reproduce the defect with full diagnostics
sbatch p18_phaseA.sbatch          # -> p18_phaseA_<jobid>.out
#   equivalently, interactively inside a 6-rank/1-GPU allocation:
#   FV3_HALO_TEST_LEGACY_SETUP=1 FV3_HALO_TEST_DEBUG=1 \
#     mpirun -np 6 ./fv3_halo_mpi_transport_test

# Intel MPI comparison reference
sbatch p18_intelmpi.sbatch        # -> p18_impi_<jobid>.out

# Standalone dual-backend short run (rebuild both halves first)
bash p17_rebuild_all.sh
cp <fixture>/build/bin/StandAlone_FV3_Dycore.x <fixture>/experiments/p17_cuda/
cp <fixture>/build/bin/StandAlone_FV3_Dycore.x <fixture>/experiments/p17_fortran_ref/
sbatch <fixture>/experiments/p17_both.sbatch
python3 tools/savepoints/compare_dcx_backends.py --stage dce   # entry
python3 tools/savepoints/compare_dcx_backends.py --stage dcx   # exit
```

The experiment directories hold their **own copy** of
`StandAlone_FV3_Dycore.x`, which `fv3.j:427-433` prefers over `$GEOSBIN` — a
rebuild that is not copied in will silently run the old binary.

---

## 7. Standalone regression and Intel MPI reference

### 7.1 Intel MPI 2021.13 comparison reference — job 58246352, node `warpa011`

**5 / 5 PASS**, `checks: 16692  failures: 0` on every repeat. Pre-fix, this
*exact* configuration measured **5/5 FAIL** (jobs 58244300, 58244365).

This is the strongest single piece of cross-stack evidence: the fix removes the
failure under the stack that reproduced it *most* reliably, which is only
consistent with a shared, MPI-independent mechanism (CUDA stream ordering) and
not with an OpenMPI-specific one.

### 7.2 §11 standalone CUDA short run — job 58246353, one node / 6 ranks / 1 A100

Both backends from the same `StandAlone_FV3_Dycore.x` in one allocation,
differing only in `AGCM.rc`'s `RUN_CUDA_FV3`; c8 / npz=10 / n_split=3 /
k_split=1 / R8; `FV_NX=1`/`FV_NY=6`. Both arms exit **rc=0**.

- **6 dyn_core calls** on both arms (`dce_it1..it6`, `dcx_it1..it6`), matched
  savepoint inventory 648 entry + 648 exit files per arm.
- **dyn_core ENTRY, call 1: all 13 state fields BITWISE IDENTICAL, 6/6 ranks**
  (`max_abs = max_rel = 0.000000e+00`, full-array bitwise 6/6). Unchanged from
  Prompt 17. The four flux capacitors (`mfxL`/`mfyL`/`cxL`/`cyL`) are dyn_core
  *outputs*, so at entry they hold uninitialized memory — identical NONFINITE
  counts on both sides for `mfyL`/`cyL`, differing garbage for `mfxL`. Not part
  of the entry claim and not a defect.
- **dyn_core EXIT, all 6 calls × 17 fields = 102 verdicts: 78 ROUND-OFF SCALE +
  24 SMALL NUMERICAL DIFFERENCE. Zero SIGNIFICANT, zero NONFINITE.** Worst
  `max_rel` over all six calls **2.034576e-11** (`w`, call 5), which is `w`'s own
  cancellation floor. Per-call worst: 1.716e-11, 1.463e-11, 1.889e-11,
  1.956e-11, 2.035e-11, 2.008e-11 — no divergence trend, and `u` grows
  3.517e-13 → 8.562e-13 across the six calls, i.e. round-off accumulation.
- **`delp` positivity: PASS.** Owned-box (`i,j = 1..8`, `k = 1..10`) minimum over
  all 6 ranks, both arms **identical to all printed digits**:
  9967.182763 → 9964.152658 → 9963.438405 → 9960.048555 → 9959.672696 →
  **9957.518684 Pa**. Zero non-finite cells.
- **Zero halo-transport diagnostics** in the run output (no `BLOCK FAIL`, no
  `POISON`, no `OWNED-BOX MOVED`), and zero NaN/Inf anywhere in either arm's log.

Every figure here **reproduces Prompt 17's recorded result to the digit**
(worst `max_rel` 2.008e-11 → 2.008e-11 at call 6; `u` 3.5e-13 → 8.6e-13; owned
min `delp` 9967.18 → 9957.52 Pa). Neither the test fix nor the production
`cudaMemsetAsync` hardening perturbed the numerics — which is the expected
result, since the hardening only strengthens an ordering that was already
holding in practice.

---

## 8. Lessons worth carrying forward

- **A "0.0" or "unwritten" reading is a claim about a bit pattern, and the
  oracle here could not make it.** `0xFF` poison and a genuine 0.0 decoded to
  the identical text for three passes. One `classify()` helper that reads the
  raw bytes turned an ambiguous signature into a mechanism.
- **Remove the suppression before theorizing about blast radius.** One
  `goto next_field` had been silently converting "an entire message is missing"
  into "one cell is wrong", and had also manufactured the check-count drops that
  were themselves recorded as a mystery.
- **Give the summary line its own print budget.** The first diagnostic run
  produced 72–156 failures with no visible failure text, purely because per-cell
  prints exhausted a shared cap and starved the per-block summary — the one line
  that carried the mechanism.
- **Keep the racy form behind a switch.** `FV3_HALO_TEST_LEGACY_SETUP` turns
  "we changed something and it stopped failing" into a controlled A/B on one
  binary in one allocation — which is what makes 24/24 a fix rather than a
  coincidence.
- **`cudaStreamNonBlocking` means what it says.** Any plain `cudaMemset` or
  synchronous `cudaMemcpy` on device memory issued alongside such a stream is
  unordered against it. A synchronous pageable H2D in particular returns when the
  *source* is safe to reuse, not when the device write has landed. Grep for both
  forms whenever a non-blocking stream is in play.
- **Audit the layer you are accusing before you accuse it.** Four audit sections
  cleared the transport (ordering, request lifetime, tag symmetry, slot
  coverage), and it was the *coverage* argument — every slot provably inside the
  region the H2D overwrites — that made "the transport left this stale" logically
  impossible and pointed at the only other writer of that buffer: the test.
