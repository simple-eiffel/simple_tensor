# INTERFACE DESIGN: simple_tensor

## Public API Summary

### Creation
| Feature | Purpose | Typical Use |
|---------|---------|-------------|
| SIMPLE_TENSOR.make | Facade on the CPU_BACKEND | `create st.make` |
| SIMPLE_TENSOR.make_with_backend | Facade on a given backend (REFERENCE for oracle tests, VULKAN in M3) | `create st.make_with_backend (create {REFERENCE_BACKEND})` |
| zeros / ones / filled / from_array | Plain tensors (untracked leaves) | `x := st.from_array (<<1, 2, 3, 4>>, <<2, 2>>)` |
| normal | Seeded Gaussian init | `w := st.normal (<<384, 384>>, 0.0, 0.02, rng)` |
| parameter | Leaf with requires_grad | `w := st.parameter (st.normal (...))` |
| indices | INDEX_TENSOR | `t := st.indices (tokens, <<64, 256>>)` |

### Ops (queries on TENSOR, each returns a new TENSOR)
| Feature | Alias | Purpose |
|---------|-------|---------|
| plus / minus / times / divided | + - * / | Broadcast elementwise |
| scaled (c), shifted (c), negated | | Constant ops |
| exp, log, sqrt, tanh, sigmoid, relu, gelu, silu | | Unary |
| sum (axis, keep), mean (axis, keep), max (axis, keep), sum_all | | Reductions |
| reshape (dims), permute (order), transposed (last two), slice (axis, start, len), concat (o, axis) | | Shape |
| matmul (o), matmul_transposed (o, ta, tb) | | Batched matmul |
| softmax, log_softmax | | Last axis |
| masked_fill (mask, v), dropout (p, rng, is_training) | | Neural |
| cross_entropy (targets, ignore), mse (o) | | Losses → scalar |
| detached, to_backend (b) | | Graph and device |

### Autograd
| Feature | Returns | Purpose |
|---------|---------|---------|
| backward | GRAD_STORE | From a scalar loss |
| GRAD_STORE.gradient (t) | TENSOR | Gradient of a parameter |
| GRAD_STORE.clip_global_norm (max) | (command) | Clipping |
| GRAD_STORE.merge (o) | (command) | Gradient accumulation across micro-batches |

### Modules
| Feature | Purpose |
|---------|---------|
| NN_LINEAR.make (st, in, out, has_bias, rng) | Linear layer, init normal(0, 0.02), bias zero (RixGPT recipe) |
| NN_LAYER.forward (x) | Apply |
| NN_EMBEDDING.forward_indices (idx) | Lookup |
| NN_MODULE.parameters / named_parameters | For optimizers and checkpoints |
| NN_MODULE.freeze / unfreeze / set_training | Training control |
| NN_LORA_LINEAR.make (base, r, alpha, rng) / merge | LoRA |

### Training
| Feature | Purpose |
|---------|---------|
| ADAMW_OPTIMIZER.make (params, schedule, b1, b2, eps, wd).step (grads) | Update |
| WARMUP_COSINE_SCHEDULE.make (peak, warmup, total, floor_ratio) | The rixgpt.py `lr_at` |
| TOKEN_BATCHER.make (tokens, block, batch, seed); forth; batch | Batches |
| SAFETENSORS_FILE.load / tensor (name) / save | Checkpoints |

## Usage Example (UC-001, one training step)

```eiffel
local
	st: SIMPLE_TENSOR
	rng: TENSOR_RANDOM
	model: RIXGPT_TRAINABLE          -- composed in simple_rixgpt's training app
	opt: ADAMW_OPTIMIZER
	batcher: TOKEN_BATCHER
	loss: TENSOR
	grads: GRAD_STORE
do
	create st.make
	create rng.make (1337)
	create model.make (st, config, rng)
	create opt.make (model.parameters,
		create {WARMUP_COSINE_SCHEDULE}.make (1.0e-3, 150, 5000, 0.1),
		0.9, 0.99, 1.0e-8, 0.1)
	create batcher.make (train_tokens, 256, 64, 1337)
	from until opt.step_count = 5000 loop
		batcher.forth
		loss := model.forward_indices (batcher.batch.inputs).cross_entropy (batcher.batch.targets, -1)
		grads := loss.backward
		grads.clip_global_norm (1.0)
		opt.step (grads)
	end
end
```

## Error Handling Pattern
- **Programming errors** (shape mismatch, wrong backend, out-of-range index) are **precondition violations**. That's the I-001 shape system; it fails at the op that made the error, with a named tag.
- **Environmental errors** (file missing, bad safetensors header, GPU allocation failure) use the fleet's `has_error` / `last_error` pattern on SAFETENSORS_FILE and VULKAN_BACKEND, never exceptions for control flow.

```eiffel
create f.load ("model.safetensors", st.backend)
if f.has_error then
	print (f.last_error)
else
	w := f.tensor ("blocks.0.attn.in_proj_weight")
end
```

## Command-Query Separation
| Feature | Type | Modifies State? | Returns Value? |
|---------|------|-----------------|----------------|
| TENSOR ops | Query | NO (new tensor; inputs untouched) | YES |
| TENSOR.backward | Query | NO (new GRAD_STORE) | YES |
| GRAD_STORE.accumulate / scale_all / clip_global_norm / merge | Command | YES | NO |
| OPTIMIZER.step | Command | YES (parameters in place, state, count) | NO |
| TOKEN_BATCHER.forth / batch | Command / Query | forth YES / batch NO | batch YES |
| NN_MODULE.freeze / set_training | Command | YES | NO |
| TENSOR_BACKEND kernels | Command | YES (writes `out`) | NO |
| SAFETENSORS_FILE.load / save | Command | YES (file or state) | NO |

Parameters are the one place tensors change in place: `OPTIMIZER.step` writes through the parameter's storage. TENSOR exports no mutator to general clients. `{TENSOR_OPTIMIZER, NN_MODULE, SAFETENSORS_FILE}.write_storage` is selectively exported, so arbitrary code can't change tensor data behind the graph.
