#!/usr/bin/env python3
"""
Build colab/Guri_Level1_Train.ipynb from the src/ files.

Why: the notebook is what the student actually runs in Google Colab, but
src/ is the real code. This script copies the code into the notebook so
there is only ONE source of truth. Re-run after editing src/:

    python3 scripts/build_notebook.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "guri"
OUT = ROOT / "colab" / "Guri_Level1_Train.ipynb"
SAMPLE = ROOT / "data" / "sample.txt"


def strip_module_boilerplate(text: str) -> str:
    """Remove relative imports and the `if __name__ == ...` CLI blocks.

    In the notebook every file's code sits in cells, so package imports and
    CLI entry points would either fail or auto-run at the wrong time.
    Handles multi-line imports like `from .data import ( ... )`.
    """
    lines = text.splitlines()
    kept: list[str] = []
    skipping_import = False
    for line in lines:
        if skipping_import:
            if line.strip() == ")":
                skipping_import = False
            continue
        if re.match(r"^from\s+\.", line) or re.match(r"^import\s+\.", line):
            # Single-line relative import, or the start of a parenthesized one.
            if line.rstrip().endswith("("):
                skipping_import = True
            continue
        if line.startswith("if __name__ =="):
            break  # drop the CLI entry block (always at the end)
        kept.append(line)
    # Trim trailing blank lines
    while kept and not kept[-1].strip():
        kept.pop()
    return "\n".join(kept) + "\n"


def md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.strip("\n").splitlines(keepends=True),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip("\n").splitlines(keepends=True),
    }


def load(name: str) -> str:
    return strip_module_boilerplate((SRC / name).read_text(encoding="utf-8"))


def download_cell() -> str:
    sample_literal = json.dumps(SAMPLE.read_text(encoding="utf-8"))
    return f'''
# STEP 1 — Get the brain's food (free public-domain books from Project Gutenberg)
import urllib.request
from pathlib import Path

DATA_DIR = Path("/content/guri_data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# How much food: "small" = 1 book (~15 min of training data)
#                 "normal" = 5 books (recommended)
book_amount = "normal"  # @param ["small", "normal"]

BOOKS = {{
    "small": [11],                       # Alice in Wonderland
    "normal": [11, 74, 1342, 84, 2701],  # Alice, Sherlock Holmes, Pride and Prejudice, Frankenstein, Moby Dick
}}[book_amount]

def fetch(book_id: int) -> str | None:
    urls = [
        f"https://www.gutenberg.org/cache/epub/{{book_id}}/pg{{book_id}}.txt",
        f"https://www.gutenberg.org/files/{{book_id}}/{{book_id}}-0.txt",
        f"https://www.gutenberg.org/files/{{book_id}}/{{book_id}}.txt",
    ]
    for url in urls:
        try:
            req = urllib.request.Request(url, headers={{"User-Agent": "GuriBrain/0.1 (training)"}})
            with urllib.request.urlopen(req, timeout=30) as r:
                text = r.read().decode("utf-8", errors="ignore")
            if len(text) > 10_000 and not text.lstrip().startswith("<!"):
                return text
        except Exception as e:  # noqa: BLE001 — try the next mirror
            print(f"   (mirror failed for {{book_id}}: {{type(e).__name__}})")
    return None

paths = []
for bid in BOOKS:
    dest = DATA_DIR / f"book_{{bid}}.txt"
    if not dest.exists() or dest.stat().st_size < 10_000:
        print(f"Downloading book {{bid}} ...")
        text = fetch(bid)
        if text is None:
            print(f"   ❌ Could not download book {{bid}} — trying the next one.")
            continue
        dest.write_text(text, encoding="utf-8")
    print(f"   ✅ {{dest.name}}: {{dest.stat().st_size:,}} characters")
    paths.append(str(dest))

if not paths:
    print("⚠️  No internet books available — using the bundled mini story pack.")
    print("   (Training will work, but a bigger dinner makes a smarter brain.)")
    fallback = DATA_DIR / "sample.txt"
    fallback.write_text({sample_literal}, encoding="utf-8")
    paths = [str(fallback)]

print(f"\\n🍽️  {{len(paths)}} food file(s) ready.")
data_paths_cell = paths  # used by the training step later
'''.lstrip("\n")


def settings_cell() -> str:
    # NOTE: plain string (not an f-string) — braces are written singly.
    return '''
# STEP 2 — SETTINGS (this is the only cell you should ever edit!)
#
# The brain's report card number is called "loss":
#   loss starts HIGH (confused) and we want it to go DOWN (smarter).

SAVE_TO_GOOGLE_DRIVE = True  # @param {type:"boolean"}
HOW_LONG = "normal"  # @param ["quick", "normal", "long"]

# Training length presets (a "step" = one tiny lesson for the brain)
PRESETS = {
    "quick":  1500,   # ~10-15 minutes on the free GPU  (silliest brain)
    "normal": 5000,   # ~30-50 minutes                  (recommended!)
    "long":   15000,  # many hours, resumes across sessions (smartest)
}
MAX_STEPS = PRESETS[HOW_LONG]

if SAVE_TO_GOOGLE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    OUT_DIR = "/content/drive/MyDrive/GuriLevel1"
else:
    OUT_DIR = "/content/guri_out"

import os
os.makedirs(OUT_DIR, exist_ok=True)

CFG = {
    "data_paths": data_paths_cell,
    "out_dir": OUT_DIR,
    "vocab_path": os.path.join(OUT_DIR, "vocab.json"),
    "max_steps": MAX_STEPS,
}
print(f"✅ Settings ready: {MAX_STEPS:,} steps -> {OUT_DIR}")
'''.lstrip("\n")


def build_notebook() -> dict:
    cells = [
        md("""
# 🧠 Guri Level 1 — Train Your Own AI Brain (from scratch!)

**Welcome, builder!** In this notebook you will:

1. 🍽️ Feed free books to a brand-new brain
2. 🧱 Build the brain yourself (millions of tiny knobs, all random at first)
3. 🏋️ Train it on a **free Google GPU** (no money, no coding)
4. 💬 Chat with it!

### Before you press anything:
- **Runtime → Change runtime type → Hardware accelerator → GPU → Save**
  (this gives you a free NVIDIA GPU for a while)
- Then press **Run** on each cell **in order** (click the ▶ play button)
- A cell shows `[*]` while it works and `[1]`, `[2]`… when finished
- If a cell shows an error in red — scroll to the bottom of this guide file
  (`GUIDE.md` in the repository) and check **Troubleshooting**

> **Rule:** never close this browser tab while a cell is running!
> Google may disconnect you after ~12 hours — that's normal.
> Your brain saves checkpoints, so you can always continue later.
"""),
        code("""
# STEP 0 — Check that our free GPU is present
import torch

print("PyTorch:", torch.__version__)
if torch.cuda.is_available():
    print("✅ GPU:", torch.cuda.get_device_name(0))
    print("   VRAM:", round(torch.cuda.get_device_properties(0).total_mem / 1e9, 1), "GB")
else:
    print("❌ NO GPU FOUND")
    print("   Fix: Runtime → Change runtime type → Hardware accelerator → GPU → Save")
    print("   Then Runtime → Restart runtime, and start again from this cell.")
"""),
        md("""
## 🍽️ Step 1 — Brain food

A brain learns by **reading**. We download free, old books (public domain —
legal for anyone to use). The more food, the smarter the brain.
"""),
        code(download_cell()),
        md("""
## 🧱 Step 2 — Settings + the mini-dictionary

The brain doesn't read letters — it reads **numbers**. So we build a
mini-dictionary (called a *tokenizer*) that converts words into numbers.

Then you choose **how long** to train. Leave it on `normal` the first time.
"""),
        code(settings_cell()),
        code(load("data.py")),
        md("""
## 🔢 Step 3 — Turn the books into a number tape
"""),
        code("""
# Build the mini-dictionary from OUR books (not downloaded from anyone!)
tokenizer = SimpleTokenizer.build(
    build_corpus(data_paths_cell),
    max_vocab=CFG.get("max_vocab", 16000),
    min_freq=2,
)
tokenizer.save(CFG["vocab_path"])
print(f"📖 Mini-dictionary: {tokenizer.vocab_size:,} words/pieces")

tokens = encode_files(data_paths_cell, tokenizer)
train_data, val_data = train_val_split(tokens)
print(f"🎞️  Number tape: {len(tokens):,} tokens")
print(f"   train: {len(train_data):,}   practice-test: {len(val_data):,}")

# Show a tiny example so it feels real
example = "The robot smiled and waved."
ids = tokenizer.encode(example)
print(f'\\nExample: "{example}" -> {ids[:12]}...')
print(f"Back to words: {tokenizer.decode(ids)}")
"""),
        md("""
## 🧱 Step 4 — Build the brain

Same idea as GPT, but mini-sized: about **8–9 million knobs** (a real 3-billion
brain would need a supercomputer — we level up to that later!).
"""),
        code(load("model.py")),
        code("""
cfg_model = GPTConfig(
    vocab_size=tokenizer.vocab_size,
    block_size=512,
    n_layer=6,
    n_head=4,
    n_embd=256,
    dropout=0.1,
)
model = GPT(cfg_model)
n = model.num_parameters()
print(f"🧠 Your brain has {n:,} knobs ({n/1e6:.2f} million)")
print("   All of them are RANDOM right now. Training will teach them.")
print("   It knows nothing yet — like a baby who never heard a word.")
"""),
        md("""
## 🏋️ Step 5 — Train! (watch the loss go down)

- **loss = the brain's report card.** High = confused, low = smart.
- The first numbers will be around **10** (pure confusion). We want **under 5**.
- Checkpoints save automatically. If Colab disconnects: **Runtime → Reconnect**,
  re-run the cells above (they'll reload from Drive), and training **resumes**.
"""),
        code(load("train.py")),
        code("""
# 🚀 START TRAINING — keep this tab open and watch the numbers!
report = train(CFG)
"""),
        md("""
## 💬 Step 6 — Talk to your brain!

Type a message and press Enter. The brain continues your text —
it's like a clever parrot that read a whole library. 🦜

Commands inside chat: `/temp 0.8` (creativity), `/tokens 200` (length), `/quit`
"""),
        code(load("chat.py")),
        code("""
# One-shot example first (no typing needed):
import os
brain, vocab, dev = load_brain(
    os.path.join(OUT_DIR, "ckpt_latest.pt"),
    CFG["vocab_path"],
)
sample = reply(brain, vocab, "Once upon a time", dev, max_new_tokens=80)
print("Once upon a time" + sample)
"""),
        code("""
# Interactive chat — click in the box below, type, press Enter.
# Type /quit when you want to stop.
chat_loop(brain, vocab, dev)
"""),
        md("""
## 🎓 What's next? (Level 2 preview)

Your Level 1 brain is a **baby storyteller**. To make it smarter:

1. **More food** — set `book_amount = "normal"` or add your own `.txt` files
2. **Longer training** — set `HOW_LONG = "long"` (resumes automatically)
3. **Bigger brain** — edit `n_layer` / `n_embd` in Step 4 (Level 2 = ~50M knobs)
4. **Save forever** — upload `OUT_DIR` to Hugging Face (see `GUIDE.md`)

Every level is trained **from scratch** — your brain, your rules. 🤝
"""),
    ]
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "cells": cells,
    }


def main() -> int:
    nb = build_notebook()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    # Sanity: valid JSON + every code cell compiles
    loaded = json.loads(OUT.read_text(encoding="utf-8"))
    for i, cell in enumerate(loaded["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        try:
            compile(src, f"cell_{i}", "exec")
        except SyntaxError as e:
            print(f"SYNTAX ERROR in cell {i}: {e}", file=sys.stderr)
            return 1
    print(f"✅ Wrote {OUT.relative_to(ROOT)} ({len(loaded['cells'])} cells, all compile)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
