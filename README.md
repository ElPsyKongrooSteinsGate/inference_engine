# QSTI

## Quantized SIMD Transformer Inference Engine

A CPU-oriented, autoregressive Transformer inference runtime featuring per-tensor INT8 weight quantization, AVX2/FMA-accelerated matrix multiplication, OpenMP parallel execution, rotary positional embeddings (RoPE), RMSNorm, SwiGLU feed-forward layers, and KV-cached causal self-attention. The project provides separate inference programs for the TinyStories 15M- and 110M-parameter checkpoints:

- `inference_engine.cpp` — 15M model (`Stories15MEngine`).
- `inference_engine_110m.cpp` — 110M model (`Stories110MEngine`).

Both programs load a `llama2.c`-format model checkpoint and tokenizer, quantize model weights to INT8 at load time, and generate text using a transformer with causal self-attention, RoPE, RMSNorm, and a SwiGLU feed-forward network. AVX2/FMA intrinsics accelerate matrix multiplication, and OpenMP parallelizes larger matrix operations.

## Requirements

- A C++17 compiler with `g++`.
- CPU and compiler support for AVX2 and FMA (`-march=native` enables supported native instructions).
- OpenMP support (`-fopenmp`).
- `curl` to download model/tokenizer files.

## Download model and tokenizer files

Run these commands from the project directory. The tokenizer is shared by both model sizes.

```sh
curl -L -o tokenizer.bin https://github.com/karpathy/llama2.c/raw/master/tokenizer.bin
curl -L -o stories15M.bin https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
curl -L -o stories110M.bin https://huggingface.co/karpathy/tinyllamas/resolve/main/stories110M.bin
```

The `.bin` files are ignored by Git because they are downloaded/generated assets and can be large. They need to be present locally when running inference.

## Compile

Compile the 110M version with the requested optimized command:

```sh
g++ -std=c++17 -O3 -march=native -mtune=native -fopenmp -DNDEBUG inference_engine_110m.cpp -o inference_engine_110.exe
```

To compile the 15M version, use the same options with its source file:

```sh
g++ -std=c++17 -O3 -march=native -mtune=native -fopenmp -DNDEBUG inference_engine.cpp -o inference_engine.exe
```

On Windows, run these commands from a GCC environment that supports OpenMP and AVX2/FMA, such as an appropriately configured MinGW-w64/MSYS2 installation. The `.exe` suffix is optional on non-Windows systems.

## Run

The 110M program defaults to `stories110M.bin`, `tokenizer.bin`, the prompt `Once upon a time`, 40 generated tokens, temperature `0.85`, and top-k `25`:

```sh
./inference_engine_110.exe
```

Example with a custom prompt and generation settings:

```sh
./inference_engine_110.exe --input "The little fox" --tokens 60 --temp 0.8 --top_k 30
```

Supported 110M options:

| Option | Description |
| --- | --- |
| `-i`, `-input`, `--input` | Prompt text |
| `-n`, `-num_tokens`, `--tokens` | Maximum number of generated tokens |
| `-t`, `-temp`, `--temp` | Sampling temperature |
| `-k`, `-top_k`, `--top_k` | Top-k sampling limit |
| `-m`, `--model` | Model checkpoint path |
| `-z`, `--tokenizer` | Tokenizer path |

The 15M program also supports prompt, token-count, temperature, and top-k options. Its model and tokenizer filenames are currently hard-coded as `stories15M.bin` and `tokenizer.bin` in the source.

Both versions stop generation at an end-of-sequence token or when the model context limit is reached, and print elapsed inference time and generated tokens per second.

## Notes and limitations

- The tokenizer in these sources is an approximate score-based BPE implementation, not the official SentencePiece implementation; tokenization and output may differ from reference inference.
- The 110M loader currently supports checkpoints where the number of key/value heads equals the number of attention heads.
- INT8 quantization reduces model-weight memory use but can affect output quality compared with full-precision inference.
- `-march=native` builds for the machine doing the compilation. The resulting executable may not run on older CPUs lacking the instructions enabled by the build host.
