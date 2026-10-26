# INNOVATIONS: simple_tensor

### I-001: Contracts as the shape system
- **Problem solved:** shape bugs (wrong axis, broadcast mismatch) are the most common silent failure in ML code. PyTorch catches some at run time with opaque messages, and many not at all (broadcasting hides them).
- **Approach:**
  - Every op carries preconditions on shapes and dtypes (`same_inner_dim: a.shape.last = b.shape [b.rank - 1]`) and postconditions on result shape.
  - Every module carries invariants on its parameter shapes (O(1) checks only).
- **Novelty:** no surveyed framework (llm.c, candle, ggml, Burn, tinygrad) states shape rules as checkable contracts.
- **Design impact:** the DbC build is the "shape-checking" build; the lean build runs at full speed.

### I-002: Every fast kernel has a slow Eiffel twin that proves it
- **Problem solved:** fast kernels (C inline, GLSL) are where numerical bugs hide.
- **Approach:**
  - Each op has a plain-Eiffel reference implementation (REAL_64 mode available).
  - The test suite runs fast against slow on random tensors, and gradient checks run on the slow path.
- **Novelty:** llm.c checks against PyTorch, and needle-rs against JAX. Here the oracle is in the **same language and library**, with no outside runtime.

### I-003: Autograd you can read in a contract
- **Approach:** each op's backward has a postcondition linking the gradient shape to the input shape, and the tape's `backward` has the postcondition "every leaf with requires_grad has a gradient of its own shape".
- **Benefit:** a missing or wrongly shaped gradient fails at the op that produced it, not 30 layers later.

### I-004: One library from char-GPT to Needle
- **Approach:** RixGPT (standard ops) and Needle 3 (engram hashing, Hadamard MLP, Sinkhorn lanes) are both just compositions of the same primitive ops; neither needs its own backward code.
- **Benefit:** "use against anything I like". A new architecture is a new composition, not a new engine.

### I-005: Trained in Eiffel, served by the vendor engine
- **Approach:** a Needle fine-tune is exported to the vendor's .cact format and run by the vendor's libneedle.dll, which we already bind. The vendor's own engine is the outside judge of our training.

### I-006: GC-safe training
- **Approach:** tensors on the C heap with every long kernel and GPU wait marked `blocking`, so a GUI or server processor in the same app keeps running while a model trains. That's the fleet's GC law applied to ML.
- **Novelty:** Eiffel-specific; no other framework has this problem or this answer.

## Differentiation from Existing Solutions
| Aspect | Existing | Our Approach | Benefit |
|--------|----------|--------------|---------|
| Shape errors | Runtime exception or silent broadcast | Preconditions per op | Fails at the source with a named tag |
| Kernel correctness | Compared against PyTorch/JAX | Compared against an in-library Eiffel twin | No outside runtime needed to test |
| GPU | CUDA (most), Vulkan (Burn, ggml partial) | Vulkan through the fleet's own simple_vulkan | Cross-vendor; owned stack |
| Language | Python / C / Rust | Eiffel | First of its kind (web search found none) |
