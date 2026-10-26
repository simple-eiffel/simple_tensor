# DOMAIN MODEL: simple_tensor

## Domain Concepts

### Concept: Shape
- **Definition:** the dimensions of a tensor, outermost first (row-major).
- **Attributes:** dims (1..rank), count (product), rank.
- **Behaviors:** equality; broadcast with another shape; reshape compatibility; drop/insert an axis.
- **Will become:** TENSOR_SHAPE (immutable).

### Concept: Storage
- **Definition:** a contiguous block of REAL_32 on one device.
- **Attributes:** element count; device (backend); C-heap block (CPU/REFERENCE) or GPU buffer (VULKAN).
- **Behaviors:** read/write one element (slow path, tests); bulk copy to and from ARRAY [REAL_32]; upload/download across devices.
- **Will become:** TENSOR_STORAGE (deferred), CPU_STORAGE (MANAGED_POINTER), VULKAN_STORAGE (M3).

### Concept: Tensor
- **Definition:** a shape over a storage, plus an autograd identity.
- **Attributes:** shape, storage, id (unique, process-wide), requires_grad, origin (the op that produced it, if tracked).
- **Behaviors:** every op (returns a new tensor); `backward` (when scalar).
- **Will become:** TENSOR.

### Concept: Index tensor
- **Definition:** integer data that indexes (token ids, targets, gather indices). Never differentiable.
- **Will become:** INDEX_TENSOR (INTEGER_32 on the C heap, with shape).

### Concept: Operation (graph node)
- **Definition:** a primitive computation that knows its inputs, any values its backward needs, and how to push an output gradient back to its inputs.
- **Will become:** TENSOR_OP (deferred) with ~30 effective descendants (OP_ADD, OP_MATMUL, …).

### Concept: Gradient store
- **Definition:** the gradients of one backward pass, keyed by tensor id.
- **Behaviors:** accumulate (`+=`), look up, global norm, scale (for clipping and accumulation), merge another store.
- **Will become:** GRAD_STORE.

### Concept: Backend
- **Definition:** where and how kernels run.
- **Variants:**
  - REFERENCE: plain Eiffel, REAL_64 accumulation, GC points; the oracle.
  - CPU: C inline, `blocking`.
  - VULKAN: GLSL through simple_vulkan.
- **Will become:** TENSOR_BACKEND (deferred) → REFERENCE_BACKEND, CPU_BACKEND, VULKAN_BACKEND.

### Concept: Module
- **Definition:** a reusable piece of a model owning named parameters (leaf tensors that require gradients).
- **Behaviors:** forward; list parameters (with dotted names for checkpoints); freeze/unfreeze; load from a checkpoint.
- **Will become:** TENSOR_MODULE (deferred), LINEAR, EMBEDDING, LAYER_NORM, RMS_NORM, CAUSAL_SELF_ATTENTION, MLP, LORA_LINEAR, MODULE_LIST.

### Concept: Optimizer
- **Definition:** updates parameters from a GRAD_STORE.
- **Attributes:** parameters, step count, learning-rate schedule, per-parameter state (AdamW's m and v).
- **Will become:** TENSOR_OPTIMIZER (deferred), SGD_OPTIMIZER, ADAMW_OPTIMIZER; LR_SCHEDULE (deferred), CONSTANT_SCHEDULE, WARMUP_COSINE_SCHEDULE.

### Concept: Checkpoint
- **Definition:** named tensors on disk.
- **Will become:** SAFETENSORS_FILE (read/write), RIXG_WRITER.

### Concept: Batch source
- **Definition:** yields (inputs, targets) INDEX_TENSOR pairs from a token sequence.
- **Will become:** TOKEN_BATCHER, TOKEN_BATCH.

### Concept: Random source
- **Definition:** a seeded generator for initialization and batching.
- **Will become:** TENSOR_RANDOM. It uses simple_randomizer; gap fill **G-2**: add a Gaussian draw to simple_randomizer.

### Concept: GC point
- **Definition:** an empty `C blocking` external that long Eiffel loops call.
- **Will become:** GC_POINT (mixin with `gc_point`).

## Concept Relationships
```
TENSOR ──has-a──> TENSOR_SHAPE
TENSOR ──has-a──> TENSOR_STORAGE ──belongs to──> TENSOR_BACKEND
TENSOR ──has-a (detachable)──> TENSOR_OP ──has-many──> TENSOR (inputs)
TENSOR.backward ──produces──> GRAD_STORE
TENSOR_MODULE ──has-many──> TENSOR (parameters), TENSOR_MODULE (children)
LORA_LINEAR ──wraps──> LINEAR
TENSOR_OPTIMIZER ──uses──> GRAD_STORE, LR_SCHEDULE
REFERENCE_BACKEND, CPU_BACKEND, VULKAN_BACKEND ──is-a──> TENSOR_BACKEND
OP_* ──is-a──> TENSOR_OP
LINEAR, EMBEDDING, ... ──is-a──> TENSOR_MODULE
```

## Domain Rules
| Rule | Description | Enforcement |
|------|-------------|-------------|
| DR-001 | Elementwise ops need broadcast-compatible shapes | Precondition `shapes_broadcast` |
| DR-002 | matmul: inner dims agree; batch dims broadcast | Precondition `inner_dims_agree` |
| DR-003 | Operands share a backend | Precondition `same_backend` |
| DR-004 | `backward` only from a scalar (count = 1) | Precondition `is_scalar` |
| DR-005 | A tensor is tracked iff it requires gradients or has an origin | Invariant `tracked_definition` (O(1)) |
| DR-006 | A gradient has its tensor's shape | GRAD_STORE.put precondition; `backward` postcondition |
| DR-007 | Leaves have no origin | Invariant `leaf_has_no_origin: requires_grad implies origin = Void` |
| DR-008 | Storage count = shape count (contiguous, no offset) | Invariant `storage_fits` (O(1)) |
| DR-009 | Index values lie in range for their use (embedding rows, classes) | Precondition (DbC build walks; lean skips) |
| DR-010 | Long Eiffel loops pass a GC point; long C kernels are `blocking` on C-heap | Code review + S-3 probe test in the suite |

## Glossary
| Term | Definition |
|------|------------|
| Broadcast | Stretching size-1 or missing leading dims to match (numpy rules) |
| Leaf | A tensor created directly (not by an op); parameters are leaves with requires_grad |
| Origin | The op that produced a tracked tensor |
| Tracked | Takes part in autograd: requires_grad, or computed from a tracked input |
| Twin | The REFERENCE_BACKEND version of a kernel, used as oracle |
| GC point | Empty `C blocking` call that lets a pending collection pass this thread |
| LoRA | Low-rank adapter: W x + (alpha/r)·B(A x), with W frozen |
