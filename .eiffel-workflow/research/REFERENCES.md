# REFERENCES: simple_tensor

All fetched 2026-10-25.

## Documentation Consulted
- https://huggingface.co/Cactus-Compute/needle3: Needle 3 card; LoRA path; per-turn tool retrieval when more than 5 tools are declared
- https://cactuscompute.com/needle: platforms (windows-x86_64), deployment
- https://pypi.org/project/cactus-needle/: package 3.2.8 (installed and read locally: finetune.py, architecture.py, export.py, quantize.py, tokenizer.py)
- https://docs.jax.dev/en/latest/installation.html: no native NVIDIA GPU for JAX on Windows
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/include/ggml-opt.h: ggml training API
- https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md: Vulkan backend backward ops
- https://github.com/ggml-org/llama.cpp/tree/master/examples/training: training is WIP; CPU/CUDA named
- https://arxiv.org/abs/2106.09685: LoRA (Hu et al., 2021); frozen base, trainable low-rank A and B

## Repositories Examined
- https://github.com/karpathy/micrograd: scalar tape autograd
- https://github.com/karpathy/llm.c: hand-written backward; oracle test (train_gpt2.c, test_gpt2.c)
- https://github.com/karpathy/llm.c/discussions/481: A100 ~178K tok/s
- https://github.com/tinygrad/tinygrad: small primitive set, lazy fusion
- https://github.com/ggml-org/ggml: C tensor library
- https://github.com/tracel-ai/burn: autodiff as a backend decorator; Vulkan via CubeCL
- https://github.com/huggingface/candle (candle-core/src/backprop.rs): tensor-level tape + GradStore
- https://github.com/KomputeProject/kompute: thin Vulkan compute layer, sequences and fences
- https://github.com/geekgineer/needle-rs: token-exact non-Python Needle 3 inference; f32-oracle lesson
- https://github.com/afshinm/cactus-needle: Needle in browser/Node
- https://github.com/ZeLarpMaster/Neural-Network: the only Eiffel NN repo found (2016, GPL-2.0)

## Articles/Papers
- https://rits.shanghai.nyu.edu/ai/cactus-needle-3-one-set-of-weights-an-8-29-mb-model-at-any-depth/: Needle 3 architecture and per-depth benchmarks; "arithmetic of a 50M model"; 2-bit artifact is Platform-only
- https://arxiv.org/html/2609.28568: numbat (Zig training stack): RTX 3090 43K tok/s, 3060 23K tok/s on GPT-2 124M
- https://huggingface.co/rhysjones/gpt2-124M-edu-fineweb-10B: llm.c GPT-2 124M on one RTX 4090, ~20 h
- https://www.mindstudio.ai/blog/cactus-needle-3-on-device-tool-calling-model: use cases

## Local evidence (this session)
- `D:\prod\simple_needle\README.md`: binding spike, 8/8 parity with Python
- `D:\prod\simple_needle\spike\python\*.py`: domain-fit tests (desktop 9/10, Bible 6–7/16)
- `D:\prod\simple_tensor\spikes\matmul\`: S-1 CPU throughput
- Fleet survey (sub-agent, file:line citations): simple_rixgpt, simple_nn, simple_linalg, simple_vulkan, simple_shaderc, simple_onnx, simple_grpc
