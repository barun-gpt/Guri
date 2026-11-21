# 🧠 Guri — Build Your Own AI Brain (from scratch)

**Guri** is a project to train your very own AI language model —
**from zero, on free computers, with no coding experience required.**

> No pre-trained models. No downloaded brains. Every parameter starts random
> and learns only from text we feed it. 🤝

## 🎯 The goal

| Level | Size | Status |
|---|---|---|
| 🟢 Level 1 | ~8 million parameters | **Ready now** — trains in one afternoon on free Colab |
| 🔵 Level 2 | ~50 million | Next |
| 🟡 Level 3 | ~350 million | Later |
| 🔴 Level 4 | **3 billion** | The final boss |

## 🚀 Quick start (5 minutes)

1. Read **[GUIDE.md](GUIDE.md)** — the full kid-friendly tutorial
2. Open the notebook in Colab:
   [**Guri_Level1_Train.ipynb**](https://colab.research.google.com/github/barun-gpt/Guri/blob/main/colab/Guri_Level1_Train.ipynb)
3. Pick GPU: **Runtime → Change runtime type → GPU**
4. Press ▶ on each cell, in order
5. Chat with the brain you trained 💬

## 🧰 Free tools we use

- **[Google Colab](https://colab.research.google.com)** — free GPU training (~15–30 h/week)
- **[Kaggle Notebooks](https://www.kaggle.com/code)** — second free GPU (~30 h/week)
- **[Google Drive](https://drive.google.com)** — checkpoint storage
- **[Hugging Face](https://huggingface.co)** — publish your brain when ready
- **[GitHub](https://github.com/barun-gpt/Guri)** — this repository
- **Arena AI** — writes and explains all the code

## 📁 What's in here

```
GUIDE.md                        ← start here! full step-by-step tutorial
colab/Guri_Level1_Train.ipynb   ← the notebook you run in Google Colab
src/guri/model.py               ← the brain (tiny GPT, built from scratch)
src/guri/data.py                ← tokenizer + data pipeline (our own dictionary)
src/guri/train.py               ← training loop, checkpoints, resume
src/guri/chat.py                ← talk to your trained brain
data/sample.txt                 ← bundled mini story pack (offline fallback)
scripts/build_notebook.py       ← regenerates the notebook from src/
```

## 🛡️ Rules of this project

1. **We never pay.** Free tiers only.
2. **We never download a pre-trained brain.** From scratch, always.
3. **Save often.** Checkpoints everywhere — a lost session never loses work.
4. **Level up honestly.** Small brain first, billions later.

## 💬 Talk to your brain

```bash
# after training (also works inside the notebook)
python -m guri.chat --ckpt out/ckpt_latest.pt
```

---

*Built with Arena AI. Teach it, train it, talk to it.* 🧠
