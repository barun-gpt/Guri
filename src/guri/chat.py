"""
Guri Level 1 chat — talk to the brain you trained.

Loads the checkpoint + dictionary, then lets you type messages and read
what the brain dreams up next.
"""

import argparse
from pathlib import Path

import torch

from .data import SimpleTokenizer
from .model import GPT, GPTConfig


def load_brain(ckpt_path: str | Path, vocab_path: str | Path | None = None, device=None):
    """Load a trained brain and its dictionary from disk."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_cfg = GPTConfig(**blob["model_config"])
    model = GPT(model_cfg)
    model.load_state_dict(blob["model"])
    model.to(device).eval()

    if vocab_path is None:
        # Default: vocab.json sitting next to the checkpoint.
        vocab_path = Path(ckpt_path).parent / "vocab.json"
    tokenizer = SimpleTokenizer.load(vocab_path)
    print(f"🧠 Loaded brain from step {blob.get('step', '?')} "
          f"({model.num_parameters() / 1e6:.2f}M params) on {device}")
    return model, tokenizer, device


@torch.no_grad()
def reply(
    model,
    tokenizer,
    prompt: str,
    device,
    max_new_tokens: int = 150,
    temperature: float = 0.8,
    top_k: int = 50,
) -> str:
    """Feed a prompt to the brain and read what it writes back."""
    ids = tokenizer.encode(prompt)
    if not ids:
        ids = [tokenizer.token_to_id["<bos>"]]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(x, max_new_tokens=max_new_tokens,
                         temperature=temperature, top_k=top_k)
    new_ids = out[0, len(ids):].tolist()
    return tokenizer.decode(new_ids)


def chat_loop(model, tokenizer, device) -> None:
    """A simple command-line chat window (works in Colab too)."""
    print("\n💬 Chat with your brain! (type /quit to leave)")
    print("   Commands: /temp 0.8   /tokens 200   /prompt some text\n")
    temperature, max_new = 0.8, 150
    while True:
        try:
            you = input("You  > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Bye!")
            break
        if not you:
            continue
        if you == "/quit":
            print("👋 Bye!")
            break
        if you.startswith("/temp"):
            try:
                temperature = float(you.split()[1])
                print(f"   (temperature = {temperature})")
            except (IndexError, ValueError):
                print("   usage: /temp 0.8")
            continue
        if you.startswith("/tokens"):
            try:
                max_new = int(you.split()[1])
                print(f"   (max new tokens = {max_new})")
            except (IndexError, ValueError):
                print("   usage: /tokens 200")
            continue
        if you.startswith("/prompt"):
            you = you[len("/prompt"):].strip()

        text = reply(
            model, tokenizer, you, device,
            max_new_tokens=max_new, temperature=temperature,
        )
        print(f"Guri > {text}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Talk to your Guri brain")
    parser.add_argument("--ckpt", default="out/ckpt_latest.pt")
    parser.add_argument("--vocab", default=None)
    args = parser.parse_args()
    model, tokenizer, device = load_brain(args.ckpt, args.vocab)
    chat_loop(model, tokenizer, device)


if __name__ == "__main__":
    main()
