# baby-llm: Minimal GPT Training Script

This project is a compact, educational implementation of a decoder-only language model in PyTorch.
It is intentionally small so you can read all of it in one sitting, run it locally, and understand the core mechanics of autoregressive language model training.

Core file:
- `train_llm_simple.py`

Sample data file:
- `corpus.txt`

## 1. What This Script Is (and Is Not)

This script **does** implement:
- A GPT-style decoder-only transformer
- Causal self-attention
- Next-token prediction objective (cross-entropy)
- Random mini-batch training
- Train/validation loss estimation
- Autoregressive text generation
- Checkpoint saving

This script **does not** implement many production LLM requirements (distributed training, subword tokenization, data filtering, instruction tuning, safety alignment, etc.). Those are described in detail later in this README.

## 2. Quick Start

## 2.1 Requirements

- Python 3.9+
- PyTorch
- tqdm (recommended, for progress bars)

Install PyTorch (example):

```bash
pip install torch
```

With progress bars:

```bash
pip install torch tqdm
```

## 2.2 Train

```bash
python3 train_llm_simple.py --corpus corpus.txt
```

## 2.3 Train with larger settings

```bash
python3 train_llm_simple.py \
  --corpus corpus.txt \
  --max-iters 20000 \
  --batch-size 128 \
  --block-size 256 \
  --n-layer 8 \
  --n-head 8 \
  --n-embed 512
```

## 2.4 Output artifacts

After training, the script writes:
- `simple_llm.pt` (or your `--out` path), containing:
  - model parameters (`model_state_dict`)
  - vocabulary mappings (`stoi`, `itos`)
  - training arguments (`args`)

The script also prints a generated sample to stdout.

## 3. Data Format and Structure

Input is a **single UTF-8 text file**.

Important points:
- The model is **character-level** in this script.
- Every unique character becomes a token.
- Spaces, punctuation, newlines, digits, and symbols are all part of the vocabulary.

Example valid corpus content:
- paragraphs
- dialogue
- logs
- lists
- key-value lines

As long as it is plain UTF-8 text, the script can train on it.

### 3.1 How the script turns text into training data

1. Read entire file as one string.
2. Build sorted unique character set.
3. Build mappings:
   - `stoi`: character -> integer token id
   - `itos`: integer token id -> character
4. Convert full text to token id sequence.
5. Split 90% train / 10% validation.
6. Sample random windows of length `block_size`.
7. Predict next character for each position in the window.

So for each training sample:
- input `x` is tokens `[t0, t1, ..., t(n-1)]`
- target `y` is `[t1, t2, ..., t(n)]`

This one-token shift is the standard autoregressive language modeling objective.

## 4. Underlying Model in This Script

Architecture: Decoder-only transformer (GPT-like), character-level vocabulary.

Main components:
- Token embedding table: maps token ids to vectors
- Positional embedding table: adds position information
- Stack of transformer blocks
  - LayerNorm
  - Multi-head causal self-attention
  - Feed-forward MLP
  - Residual connections
- Final LayerNorm
- Linear language-model head to vocabulary logits

### 4.0 Compact architecture diagram

```text
Input text
  -> character tokenizer (stoi/itos)
  -> token IDs [B, T]
  -> token embedding + positional embedding
  -> N x Transformer Block
       -> LayerNorm
       -> Causal Multi-Head Self-Attention
       -> Residual add
       -> LayerNorm
       -> Feed-Forward MLP
       -> Residual add
  -> final LayerNorm
  -> linear LM head
  -> logits [B, T, V]
  -> cross-entropy with shifted targets
```

### 4.1 Causal attention

Attention mask is lower-triangular, which enforces:
- token at position `i` can attend only to positions `<= i`
- no future-token leakage during training

### 4.2 Objective

Loss is cross-entropy over next-token prediction.

If logits are `z` and target class is `y`, token-level loss is:

`L = -log(softmax(z)_y)`

The script averages this over all batch and time positions.

## 5. Training Process Implemented Here

The training loop in `train_llm_simple.py` follows the standard pretraining skeleton:

1. Initialize model and optimizer (`AdamW`).
2. Repeat for `max_iters`:
   - sample random batch from training split
   - compute logits and cross-entropy loss
   - backpropagate gradients (`loss.backward()`)
   - optimizer step (`optimizer.step()`)
3. Every `eval_interval`, estimate train/val loss over `eval_iters` batches.
4. Save checkpoint.
5. Run autoregressive generation for `generate_tokens` steps.

### 5.1 Why validation is used

Validation loss helps detect overfitting and whether learning is progressing beyond memorizing training slices.

## 6. Hyperparameters and What They Control

- `--batch-size`: samples per optimizer step
- `--block-size`: context window length
- `--max-iters`: number of optimization steps
- `--lr`: learning rate
- `--n-embed`: embedding/model width
- `--n-head`: attention heads per layer
- `--n-layer`: number of transformer blocks
- `--dropout`: regularization strength
- `--eval-interval`: how frequently losses are reported
- `--eval-iters`: averaging batches per evaluation
- `--generate-tokens`: sample length after training

Parameter count roughly scales with `n_layer * n_embed^2` (plus embeddings/projections), so increasing width/depth rapidly increases memory and compute cost.

### 6.1 Practical parameter ranges for this script

These are practical starting points for this exact character-level implementation.

| Scenario | Corpus size (characters) | Suggested model | Suggested training |
|---|---:|---|---|
| Smoke test / debugging | 50K to 500K | `n_embed=128`, `n_layer=2`, `n_head=4`, `block_size=128` | `batch_size=32`, `max_iters=1K to 5K`, `lr=3e-4` |
| Small experiment | 1M to 10M | `n_embed=256 to 384`, `n_layer=4 to 6`, `n_head=6`, `block_size=128 to 256` | `batch_size=64`, `max_iters=10K to 50K`, `lr=3e-4` |
| Better quality (single-GPU) | 20M to 100M+ | `n_embed=512`, `n_layer=8 to 12`, `n_head=8`, `block_size=256 to 512` | `batch_size=64 to 128`, `max_iters=50K+`, `lr=1e-4 to 3e-4` |

Notes:
- In this script, 1 token = 1 character, so “corpus size” and “token count” are approximately the same.
- Absolute minimum for code to run is roughly `len(corpus) > block_size + 1`, but that is not enough for useful learning.
- A practical floor is at least `100 x block_size` characters; for non-trivial outputs, target `10,000 x block_size` or more.

## 7. End-to-End LLM Development Stages

This section separates:
- stages implemented in this project
- stages required in real-world LLM systems but omitted here

## 7.1 Stages included in this script

1. **Corpus ingestion**: Read text corpus from disk.
2. **Tokenization (character-level)**: Build vocabulary and encode text.
3. **Model definition**: Build a decoder-only transformer.
4. **Pretraining objective**: Next-token prediction with cross-entropy.
5. **Optimization**: Train using AdamW + backprop.
6. **Evaluation loop**: Periodic train/val loss estimates.
7. **Checkpointing**: Save model state and token mappings.
8. **Inference sampling**: Autoregressive generation.

## 7.2 Critical stages used in production but not included

1. **Data collection pipeline**
- Crawl/sync diverse text/code sources
- Track dataset provenance and licenses
- Filter legal/compliance-restricted content

2. **Data quality pipeline**
- Deduplication (exact + near-duplicate)
- Language identification
- Toxicity/spam/pornography filtering
- Heuristic and model-based quality scoring

3. **Advanced tokenization**
- BPE/WordPiece/SentencePiece tokenizer training
- Vocabulary design and special token strategy
- Long-tail coverage and multilingual tradeoffs

4. **Large-scale distributed training**
- Data parallelism, tensor parallelism, pipeline parallelism
- Mixed precision (FP16/BF16), gradient scaling
- Activation checkpointing, optimizer sharding
- Fault-tolerant distributed checkpointing

5. **Training stabilization and scaling controls**
- Learning-rate warmup/decay schedules
- Gradient clipping
- Weight decay tuning
- Curriculum and sampling strategies

6. **Rigorous evaluation suite**
- Benchmark tasks (reasoning, coding, QA, multilingual)
- Calibration and uncertainty checks
- Robustness/adversarial evaluations
- Memorization and privacy leakage audits

7. **Post-training (alignment)**
- Supervised fine-tuning (SFT) on instruction data
- Preference optimization (RLHF, DPO, variants)
- Safety tuning and policy shaping

8. **Model compression and serving preparation**
- Quantization
- Distillation
- Throughput/latency optimization
- KV-cache and serving runtime optimizations

9. **Deployment, governance, and monitoring**
- Online safety filters
- Prompt injection/jailbreak defenses
- Rate limits and abuse detection
- Drift monitoring, rollback strategy, retraining cadence

In short: this repo covers the **core pretraining mechanics**, not the full product-grade LLM lifecycle.

## 8. Expert Notes and Design Tradeoffs

1. Character-level tokenization keeps implementation simple but is data/compute inefficient compared to subword tokenization.
2. No LR scheduler is used; convergence quality may be lower than cosine/linear-decay schedules with warmup.
3. The model is trained from random initialization on a small corpus, so generation quality is expected to be limited.
4. The script performs full-sequence attention with quadratic time/memory in `block_size`.
5. Checkpoint currently stores only final state, not periodic snapshots or optimizer state.

## 9. Practical Path from This Script to a Real LLM

1. Replace character tokenizer with a trained subword tokenizer (SentencePiece/BPE).
2. Build robust data ETL (dedup/filter/quality scoring/provenance).
3. Add scheduler, gradient clipping, and richer logging.
4. Add periodic checkpointing with resume support.
5. Move to distributed training stack (DDP/FSDP/DeepSpeed equivalents).
6. Add comprehensive evaluation harness.
7. Add instruction tuning + preference optimization.
8. Add safety and deployment controls.

## 10. Common Failure Modes

- `len(corpus) < block_size`: batch sampling can fail.
  Concrete threshold: keep corpus length comfortably above `block_size + 1`; in practice, use much more (see section 6.1).
- Too-large model for hardware: CUDA OOM.
  Typical trigger in this script: increasing `n_embed`, `n_layer`, `batch_size`, and `block_size` together.
  Fast mitigation order: lower `batch_size` first, then `block_size`, then `n_embed`, then `n_layer`.
- High learning rate: unstable loss/divergence.
  For this AdamW setup, `3e-4` is a strong default. If loss spikes or becomes `nan`, try `1e-4`.
- Tiny corpus: memorization and repetitive outputs.
  What “tiny” often means here:
  - Very tiny: `<100K` characters (for example, a few pages of text).
  - Small: `100K to 1M` characters (for example, a short document collection).
  With these sizes, the model often overfits style fragments and repeats loops.
  Better targets:
  - Minimum for basic diversity: `1M+` characters.
  - Better for coherent generation: `20M+` characters.
- Mixed-domain corpus with little data: incoherent style generation.
  Example: combining recipes, legal text, logs, and dialogue in only `200K` characters.
  Fix options:
  - increase corpus size per domain, or
  - train on one domain at a time for clearer style behavior.

## 11. Reproducibility Notes

- Script sets `torch.manual_seed(args.seed)`.
- Exact reproducibility still varies by hardware/backend and nondeterministic kernels.
- For stricter reproducibility in research setups, add deterministic backend flags and fixed software versions.

## 12. Summary

`train_llm_simple.py` is a compact educational reference for the backbone of autoregressive language model pretraining.
It demonstrates the mechanics that underpin modern LLMs, while intentionally omitting the industrial-scale systems and alignment stages needed for production-quality models.

## 13. Code-to-README Stage Map

This maps the conceptual stages in this README to concrete code in `train_llm_simple.py`.

1. **CLI and run configuration**
- `parse_args()` in `train_llm_simple.py:17`
- Consumed in `main()` in `train_llm_simple.py:286`

2. **Corpus ingestion**
- `args.corpus.read_text(...)` in `train_llm_simple.py:300`

3. **Tokenization and vocabulary construction (character-level)**
- character set + vocab size in `train_llm_simple.py:301`
- lookup tables (`stoi`, `itos`) in `train_llm_simple.py:303`
- encoding/decoding helpers in `train_llm_simple.py:306` and `train_llm_simple.py:317`

4. **Train/validation split and batch formation**
- tensorized corpus and split in `train_llm_simple.py:328`
- random contiguous windows in `get_batch()` at `train_llm_simple.py:333`

5. **Model definition**
- `Head` in `train_llm_simple.py:52`
- `MultiHeadAttention` in `train_llm_simple.py:97`
- `FeedForward` in `train_llm_simple.py:134`
- `Block` in `train_llm_simple.py:168`
- `GPTLanguageModel` in `train_llm_simple.py:205`
- model instantiation in `train_llm_simple.py:372`

6. **Causal attention mechanics**
- causal mask creation in `Head.__init__` at `train_llm_simple.py:73`
- mask application in `Head.forward` at `train_llm_simple.py:90`

7. **Pretraining objective (next-token prediction)**
- shifted targets created in `get_batch()` at `train_llm_simple.py:347`
- cross-entropy loss in `GPTLanguageModel.forward()` at `train_llm_simple.py:261`

8. **Optimization loop**
- optimizer creation in `train_llm_simple.py:380`
- forward/backward/update in `train_llm_simple.py:390`

9. **Evaluation loop**
- evaluation helper in `estimate_loss()` at `train_llm_simple.py:351`
- periodic evaluation trigger in `train_llm_simple.py:384`

10. **Checkpointing**
- checkpoint dict in `train_llm_simple.py:396`
- checkpoint save in `train_llm_simple.py:402`

11. **Autoregressive generation**
- generation method in `GPTLanguageModel.generate()` at `train_llm_simple.py:265`
- generation call + decode in `train_llm_simple.py:405`

12. **Stages discussed in README but not implemented in code**
- see `README.md` section `7.2 Critical stages used in production but not included`
