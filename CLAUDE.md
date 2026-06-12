# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

bitnet.c is a C11 GGUF inference engine for dense, MoE, and hybrid
SSM/attention LLMs. It is CPU-first, with scalar, ARM NEON/SDOT, x86 AVX2,
x86 AVX512 BW/VNNI, and WASM SIMD CPU paths plus optional Metal, wgpu-native
WebGPU, and CUDA backends. The current architecture separates model anatomy,
quant formats, backend-resident state, transformer planning, CPU execution, GPU
op emission, KV/logits helpers, and generation APIs.

## Build And Test

```bash
make clean
make bitnet
make test

make debug
make asan
make avx2-check
make avx512-check
make test_avx512_quant
make pgo

make fetch-wgpu
make BN_ENABLE_WEBGPU=1 bitnet test_gpu_wgpu
make BN_ENABLE_METAL=1 bitnet test_coherence
make BN_ENABLE_CUDA=1 bitnet test_cuda_backend
make BN_ENABLE_ROCM=1 bitnet test_rocm_backend
make BN_ENABLE_ROCM=1 ROCM_ARCH=gfx1030 bitnet  # RX 6000 series
```

`BN_ENABLE_GPU=1` is a compatibility alias for WebGPU. Prefer
`BN_ENABLE_WEBGPU=1` in new docs and commands.

CUDA build uses `NVCC ?= /usr/local/cuda-13.2/bin/nvcc` and `CUDA_ARCH ?= sm_120` by default; override with `NVCC=...` and `CUDA_ARCH=...` as needed.

ROCm build uses `HIPCC ?= /opt/rocm/bin/hipcc` and `ROCM_ARCH ?= gfx1100` (RDNA3) by default; override `HIPCC=...` and `ROCM_ARCH=...` as needed. Requires `libamdhip64` and `librocblas` from ROCm 5.x+.

Individual tests include:

```bash
make test_architecture
make test_backend_matrix
make test_model_matrix
make test_gguf
make test_quant
make test_tokenizer
make test_transformer
make test_generate
make test_session
make test_prompt_cache
make test_threadpool
make test_safety
make test_arena
make test_ssm
make test_gguf_fuzz
make test_moe
make test_qwen36
make test_gemma4
make test_turboquant
make test_gpu_backend
make test_gpu_graph_ir
make test_cuda_backend
```

Coherence tests require a real GGUF model:

```bash
make BN_ENABLE_METAL=1 test_coherence
./test_coherence models/qwen2.5-3b-instruct-q4_0.gguf --metal

make BN_ENABLE_WEBGPU=1 test_coherence
./test_coherence models/model.gguf --webgpu

make test_coherence
./test_coherence models/model.gguf
```

Benchmark / acceptance gates (require a real model):

```bash
make BN_ENABLE_METAL=1 bench_llama_topk_server   # top-k coherence + throughput vs llama-server
./test/backend_matrix.sh                           # backend capability matrix
```

**Minimum gates before merging** (from `docs/transformer-behavior-map.md`):

```bash
make test_transformer
make test
make clean && make bitnet
make BN_ENABLE_METAL=1 test_coherence
./test_coherence models/qwen2.5-3b-instruct-q4_0.gguf --metal
./test/backend_matrix.sh
```

## Architecture Boundaries

**Dependency order** (no upstream imports allowed): `platform` → `gguf` → `quant` → `turboquant` → `model_arch` → `model` → `backend_layout`/`backend_model` → `tokenizer` → `moe` → `session` → `transformer` → `sampler` → `threadpool` → `bn_alloc` → `prompt_cache` → `generate` → `gpu_wgpu`/`gpu_metal`/`gpu_cuda` → `main`

Module responsibilities:

1. `platform` — mmap/buffer abstraction, timing
2. `gguf` — GGUF parser
3. `quant` — format metadata, dequantization, CPU kernels, backend capability declarations
4. `turboquant` — compressed KV support
5. `model_arch` — model-family rules and tensor-role mapping
6. `model` — config, immutable CPU-visible weights, model loading
7. `backend_layout` / `backend_model` — backend-owned uploads, packed/fused layouts, backend session state
8. `tokenizer` — BPE tokenizer
9. `moe` — expert routing, loading, cache, and sparse FFN compute (split across `moe_route`, `moe_io`, `moe_cache`, `moe_execute`, `moe_prefill`, `moe_math`, `moe_stats`)
10. `session` — per-request mutable KV, activations, SSM, and MoE scratch
11. `transformer` — planning and CPU/GPU execution
12. `sampler` — token sampling
13. `threadpool` — persistent pthread workers
14. `bn_alloc` — allocator vtable
15. `prompt_cache` — shared KV prefix cache
16. `generate` — library API, prefill/generate/chat/SSE/logprobs/stop strings
17. `gpu_wgpu` / `gpu_metal` / `gpu_cuda` — optional backend implementations
18. `main` — CLI wiring

Key transformer files:

| File | Responsibility |
|---|---|
| `src/transformer.c` | Top-level forward orchestration only (~144 lines). |
| `src/transformer/plan.c` | Layer/block plans and placement decisions. |
| `src/transformer/cpu.c` | CPU execution for attention, SSM, FFN, MoE, RoPE, residuals. |
| `src/transformer/gpu.c` | GPU-resident execution and CPU fallback boundaries. |
| `src/transformer/gpu_emit.c` | Emits semantic `BnGPUValueGraph` ops via `BnTransformerGPUEmitContext`. |
| `src/transformer/gpu_policy.c` | GPU forward eligibility, fallback reasons, binding-limit decisions. |
| `src/transformer/gpu_fallback.c` | CPU fallback flush/read/write sequences for SSM, MoE, and logits. |
| `src/transformer/gpu_resources.c` | Pre-resolved resource structs for QKV, attention, SSM, dense FFN, MoE. |
| `src/transformer/kv.c` | FP32, FP16, TurboQuant KV helpers. |
| `src/transformer/logits.c` | CPU logits routing. |
| `src/transformer/prefill.c` | Batch prefill. |

The GPU graph IR pipeline: `gpu_emit.c` appends semantic ops to a `BnGPUValueGraph` (defined in `include/gpu_graph.h` and `include/gpu_graph_ir.h`). The graph is lowered to backend shader commands by each backend privately via `src/gpu_shader.h` (not a public header). `BnGPUOp` is the lowered backend command format — it is not directly built by transformer emission code.

Quant kernel file naming: `src/quant/{format}_{backend}.c` (e.g. `q4k_avx2.c`, `q4k_neon_sdot.c`, `q4k_scalar.c`). Common dispatch, batch, matmul, and registry logic lives in `src/quant/*.c` without a backend suffix.

## Ownership Rules

- `BnModel` is shared and immutable after load. It owns config, architecture
  metadata, CPU-visible weights, file state, thread pool, and shared MoE I/O.
- `BnSession` is per request. It owns KV cache, activations, SSM state, MoE
  scratch, and generation position.
- `BnBackendModel` and backend session state own GPU/backend-resident buffers,
  stacked QKV/gate-up/SSM layouts, fused buffers, activation buffers, and future
  CUDA backend state. CPU SIMD kernels, including AVX512, stay in `src/quant/`
  and do not attach handles to model weights.
- `BnQWeight`, `BnLayerWeights`, and `BnWeights` must not expose backend handles.
- `BnQuantFormatOps` owns quant block geometry, sizing, CPU hooks, repack/native
  layout support, split/fused capability, and backend capability metadata.
- `BnModelArchOps` owns model-family rules: tensor names, tensor roles,
  activation/norm choices, SSM/MoE/shared-expert rules, and architecture flags.
- Public GPU graph code uses `BnGPUOpKind`, `BnGPUOpCode`, and `BN_GPU_VALUE_*`.
  `BN_GPU_SHADER_*` IDs are backend-private in `src/gpu_shader.h`.

## Testing Strategy

- Unit tests use synthetic data — no model files required unless noted.
- Each test builds only its module + dependencies, not the whole project.
- Tests use `assert()` — they crash on failure and print "PASSED" on success.
- The coherence test (`test/test_coherence.c`) validates GPU vs CPU forward pass
  (5 greedy tokens, first 3 must match), SIMD vs scalar matvec (max_diff < 2.0),
  and GPU standalone matvec vs CPU scalar when available.
- Architecture boundary tests in `test_architecture` / `test_backend_matrix`
  verify ownership invariants without loading any model file.

## Code Style

- C11, `-Wall -Wextra`, no GNU extensions unless already isolated behind a build guard.
  Linux builds add `-D_GNU_SOURCE` for `strdup`, `qsort_r`, `clock_gettime`.
- No external dependencies for CPU builds beyond libc/libm/pthread.
- Public functions use module prefixes.
- Internal helpers are `static`.
- Use `_init` / `_free` pairs for caller-owned structs. `BnSession` is created
  with `bn_session_create` and freed with `bn_session_free`.
- Keep platform-specific code behind feature macros such as `__EMSCRIPTEN__`,
  `BN_ENABLE_WEBGPU`, `BN_ENABLE_METAL`, and `BN_ENABLE_CUDA`.
- Avoid global mutable state in library modules. `main.c` and `wasm/api.c` are
  exceptions for application-level state.
- Error handling: return -1 or NULL on failure, print to stderr.

## Common Tasks

- Add a model-family rule: update `include/model_arch.h`, then add synthetic
  architecture tests.
- Add a quant format: update `include/quant.h`, `src/quant/registry.c`, format
  kernels in `src/quant/`, Makefile sources, and quant capability tests.
- Add a CPU SIMD kernel: keep it orthogonal under `src/quant/`, declare it in
  the matching `include/quant_kernels_*.h`, route it from dispatch/batch/multi
  code with feature guards, and compare it against scalar and existing SIMD
  references.
- Add backend layout behavior: update `include/backend_layout.h` and
  `src/backend_layout.c`; keep model load backend-neutral.
- Modify transformer behavior: update the relevant plan/execution module under
  `src/transformer/`, not just `src/transformer.c`.
- Add GPU behavior: append semantic ops to `BnTransformerGPUEmitContext` in
  `gpu_emit.c`, then lower in `src/gpu_metal.m`, `src/gpu_wgpu.c`, or
  `src/gpu_cuda.cu`. Keep shader IDs private in `src/gpu_shader.h`.
- Modify MoE expert dispatch: update `include/moe.h` and the relevant
  `src/moe_*.c` file.
- Add a sampling strategy: extend `src/sampler.c`.
- Add a CLI flag: update `src/main.c` and docs.
- Export a WASM API: add the `EMSCRIPTEN_KEEPALIVE` wrapper in `wasm/api.c` and
  update `wasm/build.sh`.
- Integrate as a library: include `generate.h` and `session.h`, load a
  `BnModel`, create one `BnSession` per request, then call `bn_prefill` and
  `bn_generate`.

## CLI Flags

- `--pread` — force pread for MoE expert loading (lower RSS)
- `--cache-mb N` — expert LRU cache budget in MB (default 4096, pread only)
- `--madvise` — madvise-guided mmap (experimental)
- `--draft PATH` — draft model for speculative decoding (greedy, same tokenizer)
- `--draft-k N` — draft tokens per iteration (default 5)
- `--flash` — flash attention (online softmax)
- `--kv16` — FP16 KV cache (halves KV storage)
- `--kv-tq 2|3|4` — TurboQuant compressed KV cache (recommended: 3 bits, ~8.9x compression)
- `--no-prefill` — disable batch prefill
- `--metal` — enable Metal inference (requires `BN_ENABLE_METAL=1` build)
- `--webgpu` / `--gpu` — enable WebGPU inference (requires `BN_ENABLE_WEBGPU=1` build)
- `--maxseq N` — cap sequence length; important on GPU to limit KV allocation
- `-t N` — thread count

## Performance Notes

- Inference on large models is DRAM bandwidth-bound before all cores are useful.
  Prefer measured thread counts over blindly using every core.
- AVX512 quant kernels live beside AVX2/scalar kernels in `src/quant/` and are
  dispatched from quant routing only; keep them format-local.
- Atomic work-stealing thread dispatch in threadpool for load balancing.
- MoE expert LRU cache with open-addressing hash + intrusive LRU list (pread mode).
- TurboQuant KV compression uses Randomized Hadamard Transform (O(d log d)),
  NEON-vectorized FWHT + popcount + FMA, and precomputed QJL to avoid
  per-key redundancy.
- Batch prefill uses fused Q4_K matmul kernel for dense models.
- KV cache is pre-allocated to `--maxseq`; set this explicitly on GPU.

## Runtime Notes

- MoE I/O modes are mmap, `--pread --cache-mb N`, and experimental `--madvise`.
- `--maxseq` is important on GPU and large-context models because KV allocation
  follows the selected sequence cap.
- `--kv16` halves KV cache storage.
- `--kv-tq 2|3|4` enables TurboQuant compressed KV cache.
- `--draft PATH` enables speculative decoding with a same-tokenizer draft model.
- WebGPU depends on wgpu-native adapter availability. Runtime checks may skip on
  machines with no suitable adapter.
- Metal is macOS-only and uses system Metal/Foundation frameworks. It is
  functional but can lag the CPU SIMD paths on local benchmarks; keep CPU
  fallback boundaries explicit when adding GPU coverage.

## WASM

WASM builds use `platform_load_buffer()` rather than mmap and are constrained by
wasm32 memory limits (max 2 GB). `wasm/api.c` is allowed to use global
application state for the browser demo. Keep exported functions listed in
`wasm/build.sh`.

## GPU on WSL2

WSL2 lacks a native NVIDIA Vulkan ICD. GPU inference uses Mesa's **dzn** (Dozen)
driver, which translates Vulkan to D3D12. Stock dzn is missing extensions
wgpu-native requires (`VK_EXT_robustness2`, etc.), so a patch is provided in
`patches/mesa-dzn-wgpu-compat.patch`.

**Setup:**
1. Install deps: `sudo apt-get install -y meson libdrm-dev libelf-dev llvm-dev libexpat1-dev directx-headers-dev ninja-build python3-mako libvulkan-dev && pip3 install --user meson`
2. Run `./patches/build-dzn.sh` (clones Mesa, applies patch, builds dzn)
3. Run with: `LD_LIBRARY_PATH=/usr/lib/wsl/lib VK_ICD_FILENAMES=/tmp/mesa-dzn/build/src/microsoft/vulkan/dzn_devenv_icd.x86_64.json ./bitnet model.gguf --gpu --maxseq 4096`

**What the patch does:** Adds `VK_EXT_robustness2` + `VK_EXT_image_robustness`,
raises `maxStorageBufferRange` to 2GB-1, sets robustness2 alignment properties,
and bumps the conformance version so wgpu accepts the adapter.
