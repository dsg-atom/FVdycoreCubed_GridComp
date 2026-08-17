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

### WP-E integration

Status: **PASS**

Commits applied:
- `a936aef` — WP-E v1 skeleton
- `7593d92` — WP-E v2 internal contract
- `7fd72ac` — WP-E v3 Gate A.2
- `2e2f529` — WP-E v3.1 hotfix

All WP-E runtime tests pass (R4 and R8).

### WP-A integration

Status: **PASS**

Commits applied:
- `b85c9dd` — Partial C-grid pressure gradient
- `c157b5c` — Connect to v2
- `b3b152a` — Adapt to Gate A.2
- `5cbbe93` — Complete C-grid predictor

All WP-A view tests pass (R4 and R8).

### WP-B integration

Status: **PARTIAL**

Commits applied:
- `294b20e` — D-grid translation foundation
- `f3768f5` — Connect to v2
- `c51ba46` — Consume Gate A.2
- `8d4c379` — Extend with PPM/damping/heat

Translation is PARTIAL per handoff documentation. Missing:
- Complete PPM orders 2-7, 9-13
- Full `fv_tp_2d` directional sequencing
- `pert_ppm` limiters
- True `smag_corner`
- Contravariant edge corrections
- Inline tracer transport
- CUDA dispatch launcher

All WP-B kernel/contract tests pass (R4 and R8).

### WP-C integration

Status: **PARTIAL**

Commits applied:
- `ed6b338` — Vertical column primitives
- `3bcbdb8` — Resume against v2
- `71b9c38` — Advance against Gate A.2
- `bdc2964` — Complete column solvers

Column primitives complete (all 5 solver variants). Missing:
- CUDA kernel wrapping
- Workspace arena integration
- Complete `update_dz_d` horizontal transport
- C-grid vertical solver dispatch wiring

All WP-C column tests pass (R4 and R8).

### WP-D integration

Status: **PASS**

Commits applied:
- `daa05b5` — Pressure/diffusion translations
- `d780fa3` — Connect to v2
- `722f1ab` — Complete milestone-one translations
- `008d7fe` — Close blockers with v3.1

All WP-D operator/contract tests pass (R4 and R8).

## Build and test status

| Test | Status |
|------|--------|
| WP-E R4/R8 tests | **PASS** (4/4 each) |
| WP-A R4/R8 tests | **PASS** (1/1 each) |
| WP-B R4/R8 tests | **PASS** (2/2 each) |
| WP-C R4/R8 tests | **PASS** (1/1 each) |
| WP-D R4/R8 tests | **PASS** (2/2 each) |
| Integrated CPU compilation | **PASS** |
| Integrated tests | **PASS** (10/10 R8, 10/10 R4) |
| CUDA compilation | **NOT RUN** |
| GPU execution | **NOT RUN** |
| Numerical validation | **NOT RUN** |

## Conflicts and resolutions

No conflicts encountered. All cherry-picks applied cleanly by separating
WP-E runtime commits from package-specific commits.

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

### For first runnable configuration

1. **WP-B transport orders 2-7, 9-13**: Only `hord=1,8` implemented. If GEOS
   milestone-one requires other orders, they must be completed first.

2. **`update_dz_d` horizontal transport**: Cross-WP boundary between WP-C
   (owns sequence) and WP-B (owns transport operators) is not wired.

3. **Authoritative savepoints**: No Fortran savepoints exist for numerical
   validation.

4. **CUDA compilation untested**: nvcc unavailable in current environment.

### For production validation

1. Complete PPM order coverage
2. True `smag_corner` implementation
3. Inline tracer transport (`inline_q`)
4. `pert_ppm` limiters
5. Contravariant edge corrections
6. Full `fv_tp_2d` directional sequencing

## Source layout

```
cuda/dyn_core/
  include/           WP-E public and internal headers
  runtime/           WP-E runtime implementation
  cgrid/             WP-A C-grid predictor
  dgrid/             WP-B D-grid dynamics
  vertical/          WP-C vertical solvers
  operators/         WP-D pressure/diffusion
  tests/
    runtime/         WP-E tests
    cgrid/           WP-A tests
    dgrid/           WP-B tests
    vertical/        WP-C tests
    operators/       WP-D tests
  cmake/             Platform-specific CMake modules
```

## First runnable configuration

See `docs/architecture/dyn_core_cuda_supported_config.md`.

Summary:
- `hydrostatic=.false.`
- `nested=.false.`
- `hord_*=1` or `8` only
- `inline_q=.false.`
- All vertical solvers supported
- All damping orders supported
- All pressure variants supported

**WARNING:** If GEOS milestone-one requires transport orders other than 1 or 8,
the first runnable configuration cannot yet represent it.

## Runtime version

WP-E internal contract version: 3 (with v3.1 extensions)
Public ABI version: 1 (unchanged)

## Integration commit

Final integration commit: `470c95d`

Total commits from base: 20
- WP-E: 4 commits
- WP-A: 4 commits
- WP-D: 4 commits
- WP-C: 4 commits
- WP-B: 4 commits
- Integration: 1 commit (CMake + docs)
