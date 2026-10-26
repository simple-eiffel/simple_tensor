# CLASS DESIGN: simple_tensor

Naming: no collisions found in `D:\prod\simple_*/src` for TENSOR*, NN_*, GRAD_STORE, GC_POINT, SAFETENSORS_FILE, TOKEN_BATCHER (grep, 2026-10-25).
- Ops are `TENSOR_OP_*`.
- Modules are `NN_*`.
- Backends are `*_BACKEND`.

## Class Inventory

### Core
| Class | Role | Single Responsibility |
|-------|------|----------------------|
| SIMPLE_TENSOR | Facade | Factories (zeros, ones, from_array, normal, parameter, index tensors) on an explicit backend |
| TENSOR_SHAPE | Value | Dims, count, rank; broadcast and reshape rules |
| TENSOR | Data + API | A shape over a storage; every op is a query returning a new TENSOR; `backward` |
| INDEX_TENSOR | Data | INTEGER_32 indices/targets on the C heap, with shape; never differentiable |
| TENSOR_STORAGE | Deferred | REAL_32 block on one backend: count, element access (slow path), bulk copy |
| CPU_STORAGE | Effective | MANAGED_POINTER on the C heap (D-002) |
| GRAD_STORE | Data | Gradients by tensor id; accumulate, norm, scale |
| GC_POINT | Mixin | `gc_point`: empty `C blocking` external (S-3 law) |
| TENSOR_RANDOM | Service | Seeded uniform/normal/bernoulli fills (simple_randomizer + G-2) |

### Backends (Burn pattern: autograd above, kernels below)
| Class | Role |
|-------|------|
| TENSOR_BACKEND | Deferred kernel interface: one deferred feature per primitive kernel (forward and the few non-composable backward kernels) |
| REFERENCE_BACKEND | Plain Eiffel; REAL_64 accumulation; GC point per outer loop; **the oracle twin** |
| CPU_BACKEND | C inline `C blocking` kernels on CPU_STORAGE pointers; threaded matmul (SHOULD → MUST for M2) |
| VULKAN_BACKEND | M3: GLSL kernels via simple_shaderc; VULKAN_STORAGE (inherits TENSOR_STORAGE) |

Both REFERENCE_BACKEND and CPU_BACKEND use CPU_STORAGE (same memory), so twin tests compare without copies.

### Autograd ops (all inherit TENSOR_OP)
| Group | Classes |
|-------|---------|
| Binary, broadcast | TENSOR_OP_ADD, _SUB, _MUL, _DIV |
| Unary | _NEG, _SCALE (by constant), _ADD_SCALAR, _EXP, _LOG, _SQRT, _TANH, _SIGMOID, _RELU, _GELU (exact erf), _SILU |
| Reduce | _SUM (axis, keep_dims), _MEAN (axis, keep_dims), _MAX (axis; backward to argmax) |
| Shape | _RESHAPE (view), _PERMUTE (copy), _SLICE (axis, start, length), _CONCAT (axis) |
| Linear algebra | _MATMUL (batched; transpose_a/transpose_b flags) |
| Neural | _SOFTMAX (last axis), _LOG_SOFTMAX (last axis), _MASKED_FILL (INDEX_TENSOR mask, value), _DROPOUT (p, seeded mask) |
| Index | _EMBEDDING (gather rows; scatter-add backward) |
| Loss | _CROSS_ENTROPY (fused log-softmax + NLL, ignore index, mean), _MSE |

That's **33 primitives**. LayerNorm, RMSNorm and attention are compositions in M1 (03 A-S08); fused versions come later, each with its composition as oracle.

### Modules
| Class | Role |
|-------|------|
| NN_MODULE | Deferred: `forward (x: TENSOR): TENSOR`, `parameters`, `named_parameters`, `children`, `freeze`/`unfreeze`, `set_training` |
| NN_LINEAR | W (out×in), optional bias |
| NN_EMBEDDING | Table (rows × dim); `forward_indices (INDEX_TENSOR)` |
| NN_LAYER_NORM | Gamma, beta, eps (composition) |
| NN_RMS_NORM | Gamma, eps (composition) |
| NN_DROPOUT | p; identity when not training |
| NN_MLP | Linear → GELU → Linear → Dropout |
| NN_CAUSAL_SELF_ATTENTION | Fused QKV linear, heads, causal mask, out projection (PyTorch MultiheadAttention layout, so RixGPT weights map 1:1) |
| NN_SEQUENTIAL / NN_MODULE_LIST | Containers |
| NN_LORA_LINEAR | Wraps NN_LINEAR (frozen); A (r×in, normal), B (out×r, zeros), scale alpha/r; `merge` |

### Training
| Class | Role |
|-------|------|
| TENSOR_OPTIMIZER | Deferred: `step (grads)`, `parameters`, `step_count`, `schedule` |
| SGD_OPTIMIZER | Plain SGD (+ momentum SHOULD) |
| ADAMW_OPTIMIZER | Decoupled weight decay; m, v per parameter (on the parameter's backend) |
| LR_SCHEDULE | Deferred: `rate (step): REAL_64` |
| CONSTANT_SCHEDULE, WARMUP_COSINE_SCHEDULE | PyTorch-recipe-exact formula (03 NFR-003) |
| TOKEN_BATCHER | Random windows over an INTEGER_32 sequence; seeded; train/val split |
| TOKEN_BATCH | inputs, targets (INDEX_TENSOR B×T) |
| TENSOR_TRAINER | SHOULD: loop helper (eval every N, log, checkpoint) |

### I/O
| Class | Role |
|-------|------|
| SAFETENSORS_FILE | Read/write: JSON header (simple_json) + raw little-endian data; F32/F16/BF16 → REAL_32 on load, F32 on save |
| TENSOR_DTYPE_CODEC | fp16/bf16 ↔ REAL_32 conversion (C inline, blocking) |

### Testing support (testing/ cluster)
| Class | Role |
|-------|------|
| GRADIENT_CHECKER | Central-difference check of any TENSOR → scalar function on REFERENCE_BACKEND |
| TWIN_CHECKER | Same op on two backends; worst relative error |
| GC_PROBE | S-3 allocation-latency probe as a test (UC-004) |

## Facade Design: SIMPLE_TENSOR

**Purpose:** the single entry point that creates tensors on a chosen backend.

```eiffel
class SIMPLE_TENSOR
create
	make, make_with_backend
feature -- Access
	backend: TENSOR_BACKEND
feature -- Factories
	zeros (a_dims: ARRAY [INTEGER]): TENSOR
	ones (a_dims: ARRAY [INTEGER]): TENSOR
	filled (a_dims: ARRAY [INTEGER]; a_value: REAL_32): TENSOR
	from_array (a_values: ARRAY [REAL_32]; a_dims: ARRAY [INTEGER]): TENSOR
	normal (a_dims: ARRAY [INTEGER]; a_mean, a_std: REAL_64; a_random: TENSOR_RANDOM): TENSOR
	parameter (a_tensor: TENSOR): TENSOR      -- leaf copy with requires_grad
	indices (a_values: ARRAY [INTEGER]; a_dims: ARRAY [INTEGER]): INDEX_TENSOR
end
```

**Hides:** storage allocation and the backend kernels.

## Engine Design: TENSOR_BACKEND (deferred)

```eiffel
deferred class TENSOR_BACKEND
feature -- Storage
	new_storage (a_count: INTEGER): TENSOR_STORAGE
	owns (a_storage: TENSOR_STORAGE): BOOLEAN
feature -- Kernels (out-of-place; `a_out' preallocated by the op)
	add (a, b: TENSOR_STORAGE; a_shape_a, a_shape_b, a_shape_out: TENSOR_SHAPE; a_out: TENSOR_STORAGE)
	matmul (a, b: TENSOR_STORAGE; ...; a_transpose_a, a_transpose_b: BOOLEAN; a_out: TENSOR_STORAGE)
	reduce_to_shape (a_grad: TENSOR_STORAGE; a_from, a_to: TENSOR_SHAPE; a_out: TENSOR_STORAGE)  -- broadcast backward
	... one feature per kernel
end
```

Kernels are commands writing into a preallocated output, which keeps CQS and lets the GPU backend keep outputs on the device.

## Data Class Design: TENSOR

```eiffel
class TENSOR
feature -- Access
	shape: TENSOR_SHAPE
	storage: TENSOR_STORAGE
	id: INTEGER_64
	requires_grad: BOOLEAN
	origin: detachable TENSOR_OP
	backend: TENSOR_BACKEND
feature -- Status
	is_tracked: BOOLEAN          -- requires_grad or origin /= Void
	is_scalar: BOOLEAN
feature -- Ops (queries: each returns a NEW tensor; CQS holds)
	plus alias "+" (o: TENSOR): TENSOR
	minus alias "-" (o: TENSOR): TENSOR
	times alias "*" (o: TENSOR): TENSOR
	divided alias "/" (o: TENSOR): TENSOR
	matmul (o: TENSOR): TENSOR
	matmul_transposed (o: TENSOR; a_ta, a_tb: BOOLEAN): TENSOR
	... (33 ops)
	detached: TENSOR
	to_backend (a_backend: TENSOR_BACKEND): TENSOR
feature -- Autograd
	backward: GRAD_STORE
feature -- Element access (slow path: tests, small data)
	item (a_index: INTEGER): REAL_32
	to_array: ARRAY [REAL_32]
end
```

`backward` is a query that returns a new GRAD_STORE and changes no tensor, so CQS holds.

## Inheritance Hierarchy

```
TENSOR_STORAGE           TENSOR_BACKEND                 TENSOR_OP
   ├─ CPU_STORAGE           ├─ REFERENCE_BACKEND          ├─ TENSOR_OP_ADD
   └─ VULKAN_STORAGE (M3)   ├─ CPU_BACKEND                ├─ ... (33)
                            └─ VULKAN_BACKEND (M3)        └─ TENSOR_OP_CROSS_ENTROPY
NN_MODULE                                     TENSOR_OPTIMIZER        LR_SCHEDULE
   ├─ NN_LINEAR ── NN_LORA_LINEAR (client,    ├─ SGD_OPTIMIZER        ├─ CONSTANT_SCHEDULE
   ├─ NN_EMBEDDING   not heir: wraps one)     └─ ADAMW_OPTIMIZER      └─ WARMUP_COSINE_SCHEDULE
   ├─ NN_LAYER_NORM, NN_RMS_NORM, NN_DROPOUT
   ├─ NN_MLP, NN_CAUSAL_SELF_ATTENTION
   └─ NN_SEQUENTIAL, NN_MODULE_LIST
GC_POINT (mixin) ── inherited by REFERENCE_BACKEND, TENSOR_RANDOM, TOKEN_BATCHER, SAFETENSORS_FILE
```

**Inheritance Justification:**
| Child | Parent | IS-A Valid? | Liskov OK? |
|-------|--------|-------------|------------|
| REFERENCE_BACKEND / CPU_BACKEND | TENSOR_BACKEND | Each is a way to run every kernel | YES (same pre/post; twin tests check it) |
| CPU_STORAGE | TENSOR_STORAGE | A REAL_32 block | YES |
| TENSOR_OP_* | TENSOR_OP | Each is a graph node with backward | YES |
| NN_* layers | NN_LAYER (→ NN_MODULE) | Each maps a tensor to a tensor and owns parameters | YES |
| NN_EMBEDDING | NN_MODULE only | Owns parameters, but maps indices (not tensors) to a tensor | YES (no `forward` inherited to weaken; see below) |
| NN_LORA_LINEAR | NN_MODULE (not NN_LINEAR) | A LoRA layer is not a plain linear (it has extra leaves and a merge) | Client of NN_LINEAR: composition over inheritance |

**NN_EMBEDDING and Liskov:** NN_MODULE.forward takes a TENSOR. Embedding takes indices, so NN_EMBEDDING does **not** inherit `forward`. NN_MODULE splits into:
- **NN_MODULE:** parameters, freeze, training mode.
- **NN_LAYER:** inherits NN_MODULE and adds `forward (TENSOR): TENSOR`.

NN_EMBEDDING inherits NN_MODULE only and offers `forward_indices`. No precondition strengthening.

## Generic Classes
| Class | Type Parameter | Constraint | Purpose |
|-------|----------------|------------|---------|
| none in M1 | n/a | n/a | Element type is fixed at REAL_32 (D-005). A generic TENSOR [G -> NUMERIC] would multiply every kernel by the type count for no current need |

## Class Diagram

```
┌──────────────────────┐ creates ┌────────────────────────────────────────┐
│ SIMPLE_TENSOR        │────────>│ TENSOR                                 │
│ backend              │         │ shape, storage, id, requires_grad      │
└──────────────────────┘         │ origin: detachable TENSOR_OP           │
                                 │ ops (33)  ─── each builds a TENSOR_OP ─┼──┐
                                 │ backward: GRAD_STORE                   │  │
                                 └───────┬────────────────────────────────┘  │
                                         │ storage on                       │
                                         ▼                                  ▼
                          ┌─────────────────────────┐        ┌────────────────────────┐
                          │ TENSOR_BACKEND (def.)   │<───────│ TENSOR_OP (deferred)   │
                          │ kernels                 │ calls  │ inputs, saved values   │
                          ├─────────────────────────┤        │ backward (grad, store) │
                          │ REFERENCE │ CPU │ VULKAN│        └────────────────────────┘
                          └─────────────────────────┘
NN_MODULE ── parameters ──> TENSOR (leaves)     ADAMW_OPTIMIZER ── step (GRAD_STORE)
```
