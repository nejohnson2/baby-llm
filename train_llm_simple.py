#!/usr/bin/env python3
"""
Minimal script to train a tiny GPT-style language model on a text corpus.

Usage:
  python train_llm_simple.py --corpus /path/to/corpus.txt
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def parse_args():
    """Parse command-line arguments used for training and text generation.

    Returns:
        argparse.Namespace: Parsed values for data path, model size, training loop
        settings, and output checkpoint location.
    """
    parser = argparse.ArgumentParser(
        description="Train a minimal GPT-style language model on a text corpus."
    )
    parser.add_argument("--corpus", type=Path, required=True, help="Path to a UTF-8 text file")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size")
    parser.add_argument("--block-size", type=int, default=256, help="Context window length")
    parser.add_argument("--max-iters", type=int, default=5000, help="Number of training steps")
    parser.add_argument(
        "--eval-interval", type=int, default=500, help="How often to print train/val loss"
    )
    parser.add_argument(
        "--eval-iters", type=int, default=100, help="Batches to average per evaluation"
    )
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate (AdamW)")
    parser.add_argument("--n-embed", type=int, default=384, help="Embedding size")
    parser.add_argument("--n-head", type=int, default=6, help="Attention heads per block")
    parser.add_argument("--n-layer", type=int, default=6, help="Transformer block count")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout probability")
    parser.add_argument(
        "--generate-tokens", type=int, default=300, help="Number of tokens to sample after training"
    )
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    parser.add_argument(
        "--out", type=Path, default=Path("simple_llm.pt"), help="Checkpoint output path"
    )
    return parser.parse_args()


class Head(nn.Module):
    """Single causal self-attention head.

    This module projects token embeddings into key/query/value vectors and
    performs masked scaled dot-product attention so each token only attends to
    itself and earlier tokens.
    """

    def __init__(self, head_size, n_embed, block_size, dropout):
        """Initialize projection layers and a fixed causal mask.

        Args:
            head_size (int): Dimensionality for this attention head.
            n_embed (int): Input embedding width.
            block_size (int): Maximum context length for the causal mask.
            dropout (float): Dropout probability for attention weights.
        """
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Run causal self-attention over an input sequence.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, time, channels).

        Returns:
            torch.Tensor: Attention output tensor with shape
            (batch, time, head_size).
        """
        b, t, c = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (c ** -0.5)
        wei = wei.masked_fill(self.tril[:t, :t] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v


class MultiHeadAttention(nn.Module):
    """Causal multi-head attention layer.

    Runs several `Head` modules in parallel, concatenates their outputs, and
    projects the result back to the model embedding width.
    """

    def __init__(self, n_head, head_size, n_embed, block_size, dropout):
        """Build a bank of attention heads and output projection.

        Args:
            n_head (int): Number of parallel attention heads.
            head_size (int): Width of each head.
            n_embed (int): Model embedding width.
            block_size (int): Maximum sequence length.
            dropout (float): Dropout probability for projected output.
        """
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(head_size, n_embed, block_size, dropout) for _ in range(n_head)]
        )
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Apply all heads, concatenate outputs, and project.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, time, n_embed).

        Returns:
            torch.Tensor: Tensor of shape (batch, time, n_embed).
        """
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    """Position-wise feed-forward network.

    Implements the token-wise MLP sublayer found in transformer blocks:
    expansion -> non-linearity -> projection back to model width.
    """

    def __init__(self, n_embed, dropout):
        """Create the feed-forward MLP stack.

        Args:
            n_embed (int): Model embedding width.
            dropout (float): Dropout probability after final projection.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """Apply the MLP to each token position independently.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, time, n_embed).

        Returns:
            torch.Tensor: Output tensor of shape (batch, time, n_embed).
        """
        return self.net(x)


class Block(nn.Module):
    """Single transformer decoder block with residual connections.

    The block uses pre-layer normalization, then applies causal multi-head
    attention and a feed-forward network, each wrapped in residual additions.
    """

    def __init__(self, n_embed, n_head, block_size, dropout):
        """Initialize attention and feed-forward sublayers.

        Args:
            n_embed (int): Model embedding width.
            n_head (int): Number of attention heads.
            block_size (int): Maximum context length.
            dropout (float): Dropout probability for sublayers.
        """
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embed, block_size, dropout)
        self.ffwd = FeedForward(n_embed, dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        """Run forward pass for one transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, time, n_embed).

        Returns:
            torch.Tensor: Block output with same shape as input.
        """
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    """Minimal decoder-only GPT language model.

    The model learns next-token prediction from character token IDs using
    token/position embeddings, stacked decoder blocks, and a final linear head.
    """

    def __init__(self, vocab_size, block_size, n_embed, n_head, n_layer, dropout):
        """Construct model modules and embeddings.

        Args:
            vocab_size (int): Number of unique tokens in the vocabulary.
            block_size (int): Maximum context length the model can attend to.
            n_embed (int): Embedding width.
            n_head (int): Number of attention heads per block.
            n_layer (int): Number of transformer blocks.
            dropout (float): Dropout probability.
        """
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(
            *[Block(n_embed, n_head, block_size, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        """Compute token logits and optional training loss.

        Args:
            idx (torch.Tensor): Input token IDs with shape (batch, time).
            targets (torch.Tensor | None): Optional target IDs with the same
                shape as `idx`. If provided, cross-entropy loss is computed.

        Returns:
            tuple[torch.Tensor, torch.Tensor | None]:
                - logits: If `targets` is None, shape is (batch, time, vocab).
                  If `targets` is provided, logits are flattened to
                  (batch * time, vocab) for loss computation.
                - loss: Scalar cross-entropy loss or None.
        """
        b, t = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(t, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            b, t, c = logits.shape
            logits = logits.view(b * t, c)
            targets = targets.view(b * t)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        """Autoregressively sample continuation tokens.

        Args:
            idx (torch.Tensor): Starting token IDs of shape (batch, time).
            max_new_tokens (int): Number of tokens to append.

        Returns:
            torch.Tensor: Extended token IDs of shape
            (batch, time + max_new_tokens).
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


def main():
    """Run end-to-end training and generate a sample output.

    Workflow:
    1. Parse CLI arguments and choose device.
    2. Build a character-level vocabulary from the input corpus.
    3. Train the model with periodic train/validation loss estimates.
    4. Save checkpoint artifacts.
    5. Generate a short text sample from the trained model.
    """
    args = parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    text = args.corpus.read_text(encoding="utf-8")
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    def encode(s):
        """Map text characters to token IDs.

        Args:
            s (str): Input text string.

        Returns:
            list[int]: Token ID sequence for the input text.
        """
        return [stoi[c] for c in s]

    def decode(tokens):
        """Map token IDs back to characters.

        Args:
            tokens (list[int]): Sequence of token IDs.

        Returns:
            str: Decoded text string.
        """
        return "".join([itos[i] for i in tokens])

    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    def get_batch(split):
        """Sample a random training batch of contiguous token windows.

        Args:
            split (str): `"train"` or `"val"` indicating which data split to use.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - x: Input token IDs with shape (batch_size, block_size).
                - y: Next-token targets with same shape as x.
        """
        source = train_data if split == "train" else val_data
        ix = torch.randint(len(source) - args.block_size, (args.batch_size,))
        x = torch.stack([source[i : i + args.block_size] for i in ix])
        y = torch.stack([source[i + 1 : i + args.block_size + 1] for i in ix])
        return x.to(device), y.to(device)

    @torch.no_grad()
    def estimate_loss(model):
        """Estimate average train/validation loss without gradient updates.

        Args:
            model (GPTLanguageModel): Model to evaluate.

        Returns:
            dict[str, float]: Mean loss for `"train"` and `"val"` splits.
        """
        out = {}
        model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(args.eval_iters)
            for k in range(args.eval_iters):
                xb, yb = get_batch(split)
                _, loss = model(xb, yb)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        model.train()
        return out

    model = GPTLanguageModel(
        vocab_size=vocab_size,
        block_size=args.block_size,
        n_embed=args.n_embed,
        n_head=args.n_head,
        n_layer=args.n_layer,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"device={device} vocab_size={vocab_size} tokens={len(data):,}")
    for step in range(args.max_iters):
        if step % args.eval_interval == 0 or step == args.max_iters - 1:
            losses = estimate_loss(model)
            print(
                f"step={step:5d} train_loss={losses['train']:.4f} val_loss={losses['val']:.4f}"
            )

        xb, yb = get_batch("train")
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "stoi": stoi,
        "itos": itos,
        "args": vars(args),
    }
    torch.save(checkpoint, args.out)
    print(f"saved checkpoint to {args.out}")

    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    sample = model.generate(context, max_new_tokens=args.generate_tokens)[0].tolist()
    print("\n=== SAMPLE ===\n")
    print(decode(sample))


if __name__ == "__main__":
    main()
