# nano-GPT

A from-scratch reimplementation of a small GPT (decoder-only Transformer), written as a learning project while studying [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT). The model is trained on the Tiny Shakespeare dataset and is sized to run on a laptop CPU.

## Features

- GPT-2 style architecture: token + learned position embeddings, pre-LayerNorm Transformer blocks, causal multi-head self-attention, GELU MLP
- Fused Q/K/V projection (`c_attn`) and output projection (`c_proj`), as in GPT-2
- Weight tying between the token embedding (`wte`) and the language-modeling head (`lm_head`)
- GPT-2 style weight initialization (std 0.02, with scaled initialization for residual projections)
- Training loop with AdamW, gradient clipping, optional mixed precision on CUDA, periodic train/test evaluation and throughput logging
- Autoregressive sampling with temperature and top-k
- A small benchmark for choosing the number of CPU threads

## Project structure

```
nano-GPT/
├── config.py          # GPTConfig: model hyperparameters
├── model.py           # Model: CausalSelfAttention, MLP, Block, nanoGPT
├── prepare.py         # Tokenizes data/input.txt with the GPT-2 BPE tokenizer
├── train.py           # Training loop; saves the model to data/model.pt
├── sample.py          # sample(): autoregressive text generation
├── bench_threads.py   # Measures training speed for different CPU thread counts
├── check_threads.py   # Prints PyTorch / OpenMP / MKL threading information
├── requirements.txt
└── data/
    └── input.txt      # Tiny Shakespeare (~1.1 MB)
```

## Getting started

### 1. Install dependencies

```bash
python -m venv .venv
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Prepare the data

```bash
python prepare.py
```

This splits `data/input.txt` 90/10 into train and test sets, tokenizes both with the GPT-2 tokenizer (`tiktoken`), and writes `data/train.npy` and `data/test.npy` (about 302K training tokens).

### 3. Train

```bash
python train.py
```

Training runs on CUDA if available, otherwise on the CPU. It prints the train/test loss and tokens per second at each evaluation interval and saves the final weights to `data/model.pt`.

## Configuration

Model hyperparameters live in `config.py`; training hyperparameters (`batch_size`, `max_iters`, `eval_interval`, learning rate, ...) are set at the top of the training loop in `train.py`.

Default model configuration (small enough for a laptop CPU):

| Parameter | Value | Meaning |
|---|---|---|
| `block_size` | 128 | Maximum context length |
| `vocab_size` | 50304 | GPT-2 vocabulary (50257) padded up to a multiple of 64 |
| `n_layer` | 4 | Number of Transformer blocks |
| `n_head` | 4 | Attention heads per block |
| `n_embd` | 256 | Embedding dimension |
| `dropout` | 0.0 | Dropout rate |
| `bias` | False | Whether Linear / LayerNorm layers use a bias |

## Results

All runs below were done on a Windows laptop CPU (8 cores / 12 logical processors) in float32.

### CPU baseline

`batch_size=16`, `block_size=128`, AdamW with a constant learning rate of 1e-3, 500 iterations.

| Step | Train loss | Test loss | Tokens/s |
|---|---|---|---|
| 100 | 5.4854 | 5.6717 | 1795 |
| 200 | 4.7444 | 5.3490 | 1713 |
| 300 | 4.3843 | 5.1583 | 1811 |
| 400 | 4.0223 | **4.7368** | 1828 |
| 500 | 3.9613 | 4.8521 | 1733 |

- Average throughput: about **1776 tokens/s** (about 1.15 s per step).
- The best test loss is reached at step 400; after that the test loss rises while the train loss keeps falling, i.e. the model starts to overfit the small dataset.

### CPU thread count

`bench_threads.py` measures forward + backward + optimizer step on random data with the same shapes as training (100 timed steps per setting):

| Threads | ms/step | Tokens/s |
|---|---|---|
| 2 | 1561 | 1312 |
| 4 | 1060 | 1932 |
| 6 | 1002 | 2044 |
| **8** (PyTorch default) | **972** | **2107** |
| 12 | 1042 | 1966 |

- PyTorch's default (one thread per physical core) is the fastest setting on this machine, so no manual `torch.set_num_threads` is needed.
- Using all logical processors (12) is slower: hyper-threads share a core's vector units and caches.
- Pure model compute (2107 tokens/s) is about 16% faster than the full training loop (1776 tokens/s); most of the gap comes from the Python-level batch loading in `get_batch`.

## Roadmap

- [ ] Vectorize `get_batch` to remove per-sequence Python overhead
- [ ] Use `F.scaled_dot_product_attention` (fused attention)
- [ ] Try `torch.compile`
- [ ] Learning-rate warmup and cosine decay
- [ ] Save the checkpoint with the best test loss (early stopping) and add dropout to reduce overfitting
- [ ] Add a command-line entry point for `sample.py` (load the checkpoint, encode a prompt, decode the output)
- [ ] Character-level tokenizer to shrink the vocabulary and the cost of `lm_head`

## Acknowledgements

- [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) and the "Let's build GPT" lecture by Andrej Karpathy
- The Tiny Shakespeare dataset from [karpathy/char-rnn](https://github.com/karpathy/char-rnn)
