"""
Guri Level 1 trainer — teaches the brain by showing it millions of examples.

Design rules:
  * Works on free Google Colab (T4 GPU) and on plain CPU (slow, for testing).
  * Saves checkpoints often, so a kicked session never loses progress.
  * Resumes automatically: re-run and it picks up where it left off.
"""

import math
import time
from pathlib import Path

import numpy as np
import torch

from .data import (
    SimpleTokenizer,
    build_corpus,
    encode_files,
    get_batch,
    train_val_split,
)
from .model import GPT, GPTConfig

# ---------------------------------------------------------------------------
# EDIT ME: the knobs at the top of the recipe. Kids: this is the only
# section you ever need to touch.
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = {
    "data_paths": ["data/sample.txt"],   # where the brain food lives
    "out_dir": "out",                    # where checkpoints get saved
    "vocab_path": "out/vocab.json",      # where the mini-dictionary is saved
    "max_vocab": 16000,                  # mini-dictionary size
    "min_freq": 2,                       # ignore ultra-rare words
    "block_size": 512,                   # memory span (tokens)
    "batch_size": 32,                    # snacks per step
    "n_layer": 6,                        # depth of the brain
    "n_head": 4,                         # attention spotlights
    "n_embd": 256,                       # width of the brain
    "dropout": 0.1,
    "learning_rate": 3e-4,               # how big each correction is
    "min_lr_ratio": 0.1,                 # LR decays to 10% of starting value
    "warmup_steps": 200,                 # gentle start
    "max_steps": 5000,                   # total training steps
    "eval_interval": 250,                # how often to measure progress
    "eval_batches": 40,                  # practice-test batches per eval
    "save_interval": 500,                # checkpoint every N steps
    "grad_clip": 1.0,                    # stop crazy big corrections
    "seed": 42,
    "device": "auto",                    # "auto" picks GPU if available
}


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    print("⚠️  No GPU found — training on CPU. It works, but much slower.")
    print("   In Colab: Runtime -> Change runtime type -> GPU, then re-run.")
    return torch.device("cpu")


def lr_at(step: int, cfg: dict) -> float:
    """Warm up gently, then decay — like easing into a cold pool."""
    if step < cfg["warmup_steps"]:
        return cfg["learning_rate"] * (step + 1) / max(1, cfg["warmup_steps"])
    progress = (step - cfg["warmup_steps"]) / max(1, cfg["max_steps"] - cfg["warmup_steps"])
    progress = min(max(progress, 0.0), 1.0)
    min_lr = cfg["learning_rate"] * cfg["min_lr_ratio"]
    return min_lr + 0.5 * (cfg["learning_rate"] - min_lr) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def estimate_loss(model, data, cfg, device) -> float:
    """Score the brain on held-out text it never trained on (practice test)."""
    model.eval()
    losses = []
    for _ in range(cfg["eval_batches"]):
        x, y = get_batch(data, cfg["block_size"], cfg["batch_size"], device)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def save_checkpoint(path: Path, model, optimizer, scaler, step: int, cfg: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "step": step,
            "model_config": {
                "vocab_size": model.cfg.vocab_size,
                "block_size": model.cfg.block_size,
                "n_layer": model.cfg.n_layer,
                "n_head": model.cfg.n_head,
                "n_embd": model.cfg.n_embd,
                "dropout": model.cfg.dropout,
            },
            "train_config": cfg,
        },
        tmp,
    )
    tmp.replace(path)  # atomic-ish: never leave a half-written checkpoint


def train(cfg: dict | None = None) -> dict:
    """Main training loop. Returns a small report dict."""
    cfg = {**DEFAULT_CONFIG, **(cfg or {})}
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    device = pick_device(cfg["device"])
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Brain food -------------------------------------------------------
    print("🍽️  Reading brain food...")
    corpus = build_corpus(cfg["data_paths"])
    print(f"   {len(corpus):,} characters from {len(cfg['data_paths'])} file(s)")

    Path(cfg["vocab_path"]).parent.mkdir(parents=True, exist_ok=True)
    tokenizer = SimpleTokenizer.build(
        corpus, max_vocab=cfg["max_vocab"], min_freq=cfg["min_freq"]
    )
    tokenizer.save(cfg["vocab_path"])
    print(f"   Mini-dictionary: {tokenizer.vocab_size:,} tokens -> {cfg['vocab_path']}")

    tokens = encode_files(cfg["data_paths"], tokenizer)
    train_data, val_data = train_val_split(tokens)
    print(f"   Token tape: {len(tokens):,} tokens "
          f"({len(train_data):,} train / {len(val_data):,} val)")

    # Shrink the memory span if the data is tiny (keeps small demos working).
    smallest = min(len(train_data), len(val_data))
    if smallest < cfg["block_size"] + 2:
        cfg["block_size"] = max(32, smallest - 2)
        print(f"   ⚠️  Small dataset: memory span reduced to {cfg['block_size']} tokens. "
              f"Add more text for a smarter brain!")

    # ---- Build the brain --------------------------------------------------
    model_cfg = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=cfg["block_size"],
        n_layer=cfg["n_layer"],
        n_head=cfg["n_head"],
        n_embd=cfg["n_embd"],
        dropout=cfg["dropout"],
    )
    model = GPT(model_cfg).to(device)
    n_params = model.num_parameters()
    print(f"\n🧠 Brain built: {n_params / 1e6:.2f} million parameters on {device}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ---- Resume if a checkpoint exists ------------------------------------
    latest = out_dir / "ckpt_latest.pt"
    start_step = 0
    if latest.exists():
        blob = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(blob["model"])
        optimizer.load_state_dict(blob["optimizer"])
        if blob.get("scaler") and use_amp:
            scaler.load_state_dict(blob["scaler"])
        start_step = int(blob["step"])
        print(f"🔁 Resumed from step {start_step:,} (training continues!)")

    # ---- The learning loop ------------------------------------------------
    print(f"\n🚀 Training for {cfg['max_steps']:,} steps "
          f"(already done: {start_step:,}). Keep this tab open!\n")
    print(f"{'step':>7} {'train':>8} {'val':>8} {'lr':>10} {'tok/s':>9}  note")
    print("-" * 62)

    model.train()
    t0 = time.time()
    tokens_since_log = 0
    window_start = t0
    report = {"best_val": float("inf"), "final_step": start_step}

    for step in range(start_step + 1, cfg["max_steps"] + 1):
        lr = lr_at(step - 1, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = get_batch(train_data, cfg["block_size"], cfg["batch_size"], device)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        scaler.step(optimizer)
        scaler.update()
        tokens_since_log += x.numel()

        if step % cfg["eval_interval"] == 0 or step == cfg["max_steps"]:
            train_loss = estimate_loss(model, train_data, cfg, device)
            val_loss = estimate_loss(model, val_data, cfg, device)
            now = time.time()
            tok_s = tokens_since_log / max(now - window_start, 1e-6)
            tokens_since_log = 0
            window_start = now

            note = ""
            if val_loss < report["best_val"]:
                report["best_val"] = val_loss
                note = "⭐ new best!"
                save_checkpoint(
                    out_dir / "ckpt_best.pt", model, optimizer, scaler, step, cfg
                )
            if step % cfg["save_interval"] == 0 or step == cfg["max_steps"]:
                save_checkpoint(latest, model, optimizer, scaler, step, cfg)
                if not note:
                    note = "💾 saved"
            print(f"{step:>7,} {train_loss:>8.3f} {val_loss:>8.3f} "
                  f"{lr:>10.2e} {tok_s:>9,.0f}  {note}")

    elapsed = time.time() - t0
    save_checkpoint(latest, model, optimizer, scaler, cfg["max_steps"], cfg)
    print("\n🎓 Training finished!")
    print(f"   Time: {elapsed / 60:.1f} minutes")
    print(f"   Best practice-test score (val loss): {report['best_val']:.3f}")
    print(f"   Brain saved to: {latest}")
    print(f"   Dictionary saved to: {cfg['vocab_path']}")
    print("\n👉 Next: run the chat section and TALK TO YOUR BRAIN!")
    report["final_step"] = cfg["max_steps"]
    report["checkpoint"] = str(latest)
    return report


if __name__ == "__main__":
    train()
