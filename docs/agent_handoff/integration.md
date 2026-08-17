# FV3 dyn_core CUDA Integration Handoff

## Integration baseline

| Component | Commit | Branch |
|-----------|--------|--------|
| Base | `ac43844` | `feature/cuda-port-experiments` |
| WP-E v3.1 | `b92c5ff` | `cuda/wp-e-runtime` |
| WP-A | `1f59b40` | `cuda/wp-a-cgrid` |
| WP-B | `6116d7b` | `cuda/wp-b-dgrid` |
| WP-C | `aefc1c5` | `cuda/wp-c-vertical` |
| WP-D | `d04592a` | `cuda/wp-d-operators` |

## Commit ancestry analysis

Each numerical WP branch contains cherry-picked WP-E runtime commits. The
integration must use one canonical WP-E v3.1 history and apply only
package-specific commits from WP-A/B/C/D.

### WP-E canonical history (4 commits)

```
965c8ca  Add FV3 GPU runtime and state contract skeleton       [v1]
7cecdfc  Add FV3 GPU internal contract version 2               [v2]
73efe42  Close FV3 GPU numerical contract version 3            [v3/Gate A.2]
b92c5ff  Add FV3 GPU internal contract v3.1 hotfix             [v3.1]
```

### WP-A commits (7 total)

| Commit | Type | Description |
|--------|------|-------------|
| `b095bd5` | WP-E cherry-pick | v1 skeleton |
| `226e2dc` | **WP-A specific** | Partial C-grid pressure gradient |
| `32fa2ba` | WP-E cherry-pick | v2 contract |
| `1677951` | **WP-A specific** | Connect pressure to v2 |
| `6013df2` | WP-E cherry-pick | v3 contract |
| `9ebc072` | **WP-A specific** | Adapt pressure to Gate A.2 |
| `1f59b40` | **WP-A specific** | Complete C-grid predictor |

WP-A package-specific commits: `226e2dc`, `1677951`, `9ebc072`, `1f59b40`

### WP-B commits (7 total)

| Commit | Type | Description |
|--------|------|-------------|
| `9834f77` | WP-E cherry-pick | v1 skeleton |
| `d67626f` | **WP-B specific** | D-grid translation foundation |
| `a2334ed` | WP-E cherry-pick | v2 contract |
| `1f8c10b` | **WP-B specific** | Connect to v2 |
| `f633546` | WP-E cherry-pick | v3 contract |
| `cf3ff5c` | **WP-B specific** | Consume Gate A.2 |
| `6116d7b` | **WP-B specific** | Extend with PPM/damping/heat |

WP-B package-specific commits: `d67626f`, `1f8c10b`, `cf3ff5c`, `6116d7b`

### WP-C commits (7 total)

| Commit | Type | Description |
|--------|------|-------------|
| `51e1146` | WP-E cherry-pick | v1 skeleton |
| `2f36897` | **WP-C specific** | Vertical column primitives |
| `0cd0c9f` | WP-E cherry-pick | v2 contract |
| `a89bef6` | **WP-C specific** | Resume against v2 |
| `28e4bc3` | WP-E cherry-pick | v3 contract |
| `6254a4d` | **WP-C specific** | Advance against Gate A.2 |
| `aefc1c5` | **WP-C specific** | Complete column solvers |

WP-C package-specific commits: `2f36897`, `a89bef6`, `6254a4d`, `aefc1c5`

### WP-D commits (7 total)

| Commit | Type | Description |
|--------|------|-------------|
| `6dc3aba` | WP-E cherry-pick | v1 skeleton |
| `456b457` | **WP-D specific** | Pressure/diffusion translations |
| `56b9ebf` | WP-E cherry-pick | v2 contract |
| `a368b97` | **WP-D specific** | Connect to v2 |
| `1b57d64` | WP-E cherry-pick | v3 contract |
| `e728e9e` | **WP-D specific** | Complete milestone-one translations |
| `d04592a` | **WP-D specific** | Close blockers with v3.1 |

WP-D package-specific commits: `456b457`, `a368b97`, `e728e9e`, `d04592a`

## Integration strategy

1. Apply WP-E v3.1 complete history first (canonical runtime)
2. Cherry-pick WP-A package-specific commits
3. Cherry-pick WP-D package-specific commits
4. Cherry-pick WP-C package-specific commits
5. Cherry-pick WP-B package-specific commits
6. Resolve any conflicts favoring WP-E v3.1 contract

This order integrates complete packages (WP-A, WP-D) before partial ones
(WP-C, WP-B) and avoids replaying obsolete WP-E versions.

## Integration status

Status will be updated as integration proceeds.

### WP-E integration

Status: PENDING

### WP-A integration

Status: PENDING

### WP-B integration

Status: PENDING
Note: Translation is PARTIAL per handoff documentation

### WP-C integration

Status: PENDING
Note: Column primitives complete; CUDA/update_dz_d partial

### WP-D integration

Status: PENDING

## Build and test status

| Test | Status |
|------|--------|
| WP-E R4/R8 tests | PENDING |
| WP-A R4/R8 tests | PENDING |
| WP-B R4/R8 tests | PENDING |
| WP-C R4/R8 tests | PENDING |
| WP-D R4/R8 tests | PENDING |
| Integrated CPU compilation | PENDING |
| CUDA compilation | NOT RUN |
| GPU execution | NOT RUN |
| Numerical validation | NOT RUN |

## Conflicts and resolutions

(To be recorded during integration)

## Implementation gaps

### WP-B documented gaps

- Complete PPM orders 2-7, 9-13
- Complete fv_tp_2d directional sequencing
- pert_ppm
- True smag_corner (currently approximated)
- Contravariant edge corrections
- Inline tracer transport (inline_q)
- CUDA dispatch launcher

### WP-C documented gaps

- CUDA kernel wrapping
- Workspace arena integration
- Complete update_dz_d horizontal transport
- C-grid vertical entry for WP-A

## Cross-WP boundaries

### update_dz_d cross-WP resolution

WP-C owns vertical/update_dz_d logic but requires WP-B transport operators.
Resolution documented in dyn_core_cuda_integrated_sequence.md.

## Remaining blockers

(To be identified during integration)
