# Halo Exchange / Domain-Decomposition Modernization

## Purpose

Scale a domain-decomposed model's boundary-exchange ("halo exchange")
mechanism onto a new backend, and scale the decomposition itself (more
ranks per domain unit) without conflating physical-domain topology with
communication/transport identity.

## Use When

- Porting a domain-decomposed model's boundary exchange to a new backend
  (e.g. GPU-resident halo exchange).
- Scaling beyond a fixed, assumed rank-to-domain-unit mapping (e.g. more
  than one rank per sub-domain/face/tile).

## Inputs Needed

- The existing decomposition library's actual layout/rank-ordering
  metadata (e.g. an FMS-equivalent domain-decomposition library) — check
  whether it already supports the finer decomposition you need before
  building anything new.
- A way to empirically verify rank-ordering assumptions at runtime (a
  diagnostic print of real decomposition metadata), not just source
  inspection or inference.
- An existing, validated halo-transport mechanism for the current (coarse)
  decomposition, to use as the correctness reference.

## Procedure

1. Before building any new decomposition/communication mechanism, check
   whether the underlying legacy communication/domain library already
   supports the finer decomposition you need (e.g. a layout parameter
   already present but unused by your new backend) — you may only need to
   plumb existing metadata through, not invent a mechanism.
2. Separate **physical topology identity** (which domain unit, e.g. which
   cubed-sphere face/tile, am I) from **transport/communication identity**
   (which rank do I actually send to) as two distinct fields from the
   start — do not assume these are interchangeable, even if they happen
   to coincide 1:1 in your current (coarse) decomposition. This
   distinction only matters once more than one rank can share one
   physical domain unit, but build it in before you need it, since
   retrofitting it after an identity-conflation bug ships is more
   expensive.
3. Empirically verify any assumed rank-ordering formula (e.g. "sub-rank
   ordering is row-major within a domain unit") using a minimal, legacy-
   language-only diagnostic that reads real runtime decomposition metadata
   (actual compute-domain index bounds per rank), independent on each
   axis to avoid ambiguity — never assume ordering from a single combined
   index or from documentation alone.
4. Stage the integration: first plumb new decomposition metadata through
   the ABI/bridge additively (no behavior change, cross-checked at init
   e.g. via an allgather consistency check), separately from any later
   step that actually uses the metadata to change topology/routing
   behavior.
5. Validate the new topology/transport logic at a synthetic/test rank
   count first (e.g. a scale that exercises the finer decomposition
   without requiring full production-scale resources), checking routing
   correctness exhaustively (every expected neighbor pair communicates
   correctly, zero unexpected failures).
6. Keep the production entry point's existing rank-count gate (e.g. "must
   be exactly N ranks") intact and unmodified until the new mechanism has
   been independently verified end-to-end — do not relax a production gate
   as a side effect of adding new capability.
7. If corner/diagonal-junction communication exists in your decomposition
   (multiple domain units meeting at a point, not just an edge), treat it
   as a separate, explicitly tracked problem — don't assume edge-halo
   correctness implies corner correctness.

## Validation / PASS Criteria

- Physical topology identity and transport identity are stored as
  distinct fields/values in the new mechanism, even where they currently
  coincide.
- Any rank-ordering formula used by the new mechanism has been verified
  by an actual runtime diagnostic, not assumed.
- The new topology/transport logic has been validated at a real (not
  simulated) multi-rank run with zero unexpected transfer failures.
- Existing production rank-count gates are confirmed unchanged after the
  new mechanism is added, by direct re-inspection of the gate condition.

## STOP Conditions

- A rank-ordering assumption has not been empirically verified — do not
  build derivation/routing logic on top of it; verify first via a minimal
  diagnostic.
- The new mechanism has only been validated at a synthetic/test scale, not
  real production scale — do not relax the production rank-count gate
  based on synthetic validation alone; treat gate relaxation as a
  separate, later decision requiring its own evidence.
- A conflation is found between physical-domain identity and transport
  identity in existing code — stop and fix it (separate the two fields)
  before extending the decomposition further; this is a known, previously
  realized bug pattern, not a hypothetical.

## Common Failure Modes

- Indexing a transport/peer-send table directly by physical-domain
  identity (e.g. "which face") — works perfectly while there is exactly
  one rank per domain unit, and silently breaks the moment more than one
  rank shares a domain unit, because the table can no longer distinguish
  which specific rank within that unit should receive a given message.
- Assuming a decomposition library's rank ordering (e.g. row-major within
  a domain unit) without checking real runtime metadata — a reasonable-
  sounding assumption can simply be wrong, or right in one axis and wrong
  in another, which only a per-axis empirical check catches.
- Treating a successful synthetic-scale validation (e.g. correct routing
  at a small test rank count) as sufficient evidence to relax a production
  gate — synthetic validation proves the mechanism works, not that it is
  safe to enable for real traffic yet.
- Forgetting corner/diagonal-junction communication because edge-halo
  logic looks complete.

## Cost-Control Guidance

- Check for existing decomposition-library support (step 1) before
  estimating any implementation cost — this can turn a "build a new
  mechanism" task into a much cheaper "plumb existing metadata through"
  task.
- Verify rank-ordering (step 3) with the smallest multi-rank test that can
  unambiguously resolve it (e.g. a non-square sub-decomposition if a
  square one would be ambiguous) — do not run a larger confirmatory test
  once the hypothesis is already resolved with zero exceptions.
- Validate new routing logic at a synthetic/moderate rank count before
  requesting the larger resource allocation a full production-scale test
  would need.

## Reusable Prompt Template

```
Extend <halo exchange mechanism> to support <finer decomposition, e.g.
more than one rank per domain unit>. First check whether <existing
decomposition library> already supports this layout parameter — if so,
plumb its metadata through the ABI/bridge additively, with an init-time
consistency check, before changing any routing behavior. Separate
physical-domain identity from transport/rank identity as distinct fields
if not already distinct. Empirically verify any rank-ordering assumption
via a minimal <legacy-language>-only diagnostic reading real
decomposition metadata, independent per axis. Validate new routing logic
at a synthetic multi-rank scale with an exhaustive correctness check (every
expected neighbor pair, zero failures) before considering any production
rank-count gate relaxation — treat gate relaxation as a separate,
later decision.
```

## FV3 Example

**(FV3-specific illustration, not a universal rule.)** FV3's CUDA bridge
hardcodes `comm_size==6` (one rank per cubed-sphere face). Scaling beyond
C512 needs sub-tile decomposition (more than one rank per face), which
FMS already supports via `Atm%layout(1:2)` — this metadata was plumbed
through the ABI additively (`layout_x`/`layout_y`, Prompt 114) with zero
behavior change, cross-checked via `MPI_Allgather`. A real bug was found
and fixed (Prompt 111->112): `launch_halo_transfer` indexed its peer-send
table by `.tile` (physical face), conflating it with transport-rank
identity; the fix added a distinct `.rank` field. FMS's within-tile rank
ordering was *assumed* row-major in Prompts 113-114 but explicitly
flagged UNVERIFIED, then empirically verified by a Fortran-only, zero-CUDA
24-rank `layout=(2,2)` diagnostic (Prompt 115) deriving `sub_i`/`sub_j`
independently from each axis's own `(is:ie)`/`(js:je)` compute-domain
bounds, confirming `local_rank_within_tile = sub_j*layout_x+sub_i` with
zero exceptions. The topology/transport-peer-identity layer was then
GPU-validated at real 24-rank MPI scale (0/22224 failures) — but all four
production `comm_size==6`-style gates remain intentionally unrelaxed; no
real >6-rank production path exists yet (`docs/agent_work_summary.md`
§9). Corner transport remains metadata-only, separately tracked, deferred
multiple times.
