# 📚 GUIDE — Build Your Own AI Brain (Level 1)

**Hi! I'm your programming partner.** This guide has *every tiny step*.
You don't need to know coding. You need to know how to **click buttons**
and **read words**. That's it.

> **Your mission:** train a little AI brain **from scratch** (no downloaded
> brains!), then talk to it. Everything runs on **free** tools. We never pay.

---

## 🗺️ The big picture (2 minutes of reading)

| Tool | What it does | Cost |
|---|---|---|
| **Arena AI** (me) | Writes the code, fixes problems, teaches you | Free |
| **Google Colab** | A free computer with a GPU in your browser — training happens HERE, not on your laptop | Free |
| **Google Drive** | Saves your brain's homework so nothing is lost | Free |
| **GitHub** | Stores the project files (already yours!) | Free |
| **Hugging Face** | A garage to park your finished brain (later) | Free |

**Your weak laptop only needs to open a web page.** The heavy lifting
happens on Google's computers.

---

## ✅ STEP 0 — Make 2 free accounts (10 minutes, do once)

### 0A. Google account (for Colab + Drive)
1. Go to **accounts.google.com**
2. Click **Create account**
3. Fill in your name → pick a Gmail address → set a password
4. Finish the signup (they may ask for a phone number — use it once)

> Already have a Gmail? Skip to 0B. You're done. 🎉

### 0B. Hugging Face account (for later — 2 minutes)
1. Go to **huggingface.co**
2. Click **Sign Up**
3. Use your email (or "Sign up with Google")
4. Confirm the email they send you

*We won't use this until Level 1 finishes. But it's nice to have it ready.*

**👉 Tell a grown-up (or just say here): "STEP 0 DONE." Then continue.**

---

## 🚀 STEP 1 — Open the training notebook (2 minutes)

The notebook is a magic page where you press ▶ and the AI brain gets built.

### Option A — one-click link (easiest)
Open this link in Chrome/Edge/Firefox:

```
https://colab.research.google.com/github/barun-gpt/Guri/blob/main/colab/Guri_Level1_Train.ipynb
```

- If Google asks you to **sign in**, use your Google account.
- If it says **"Open in Colab?"** → click **Open**.

### Option B — do it by hand
1. Go to **colab.research.google.com**
2. Click **File → Open notebook**
3. Click the **GitHub** tab at the top
4. Type `barun-gpt/Guri` and press Enter
5. Click **Guri_Level1_Train.ipynb**

---

## ⚡ STEP 2 — Give yourself a free GPU (1 minute, VERY important)

A GPU is a special chip that makes training ~50× faster.

1. At the top menu click **Runtime**
2. Click **Change runtime type**
3. Where it says **Hardware accelerator**, choose **GPU**
4. Click **Save**

### How to check it worked
- The first code cell will print `✅ GPU: NVIDIA Tesla T4` (or similar)
- If you see `❌ NO GPU FOUND` → repeat steps 1–4, then click
  **Runtime → Restart runtime** and run the first cell again.

---

## ▶️ STEP 3 — Run the cells IN ORDER (the main event)

Every box on the page is a **cell**. Each has a ▶ play button on its left.
**Click ▶, wait for `[1]`, `[2]`, `[3]`… then move to the next.**

Never skip. Never run two at once. (One at a time. Like chewing food. 🍽️)

---

### Cell 1 — Check the GPU
**Press ▶.** Expect:
```
PyTorch: 2.x.x
✅ GPU: NVIDIA Tesla T4
   VRAM: 15.0 GB
```
❌ If you see NO GPU → go back to **STEP 2**.

---

### Cell 2 — Download the brain's food 🍽️
**Press ▶.** This fetches **5 free old books** from Project Gutenberg
(Alice in Wonderland, Sherlock Holmes, Pride and Prejudice, Frankenstein,
Moby Dick). Old books = free for everyone = legal food. ✅

Expect:
```
Downloading book 11 ...
   ✅ book_11.txt: 174,434 characters
...
🍽️  5 food file(s) ready.
```

**⏱ Wait.** It can take 1–3 minutes. Don't click anything else.

**If it says "No internet books"** — don't panic. Google sometimes blocks
connections. Click ▶ again (it retries), or change `"normal"` to `"small"`
in that cell and retry.

---

### Cell 3 — Settings ⚙️ (the ONLY cell you should ever edit)
**Press ▶.** You'll see two dropdowns Colab turns into little menus:

| Setting | What it means | What to pick |
|---|---|---|
| `SAVE_TO_GOOGLE_DRIVE` | Keep the brain safe after Colab kicks you out | `True` (default) ✅ |
| `HOW_LONG` | How long to train | First time: `normal` |

It will ask permission: **"Permit access to Google Drive?"** →
click **Connect to Google Drive** → allow. This is how your homework
survives. (If you pick `False`, training works but a disconnect = start over.)

Expect:
```
✅ Settings ready: 5,000 steps -> /content/drive/MyDrive/GuriLevel1
```

---

### Cell 4 — Build the mini-dictionary 🔤
**Press ▶.** The brain reads **numbers**, not letters. This cell chops
books into words and gives each word an ID number.

Expect:
```
📖 Mini-dictionary: 16,000 words/pieces
🎞️  Number tape: 1,234,567 tokens
...
Example: "The robot smiled." -> [5, 88, 74]...
Back to words: the robot smiled.
```

---

### Cell 5 — Build the brain 🧱
**Press ▶.** This is where your AI is born. All its knobs start **random**.

Expect:
```
🧠 Your brain has 8,412,672 knobs (8.41 million)
   All of them are RANDOM right now. Training will teach them.
   It knows nothing yet — like a baby who never heard a word.
```

> "But you promised 3 BILLION!" — We did! And we'll get there.
> Level 1 is the tutorial world. You wouldn't give a learner driver
> a race car. 🏎️ First: learn to steer.

---

### Cell 6 — THE TRAINING 🏋️ (the long one!)
**Press ▶ and KEEP THIS TAB OPEN.** ☕ This is the real work.

You'll see a table like:
```
   step    train      val         lr     tok/s  note
--------------------------------------------------------------
   250    6.842    6.901   1.50e-04   82,340  ⭐ new best!
   500    5.910    6.022   3.00e-04   83,101  💾 saved
   750    5.433    5.610   2.87e-04   82,889  ⭐ new best!
```

**How to read your brain's report card:**

| Column | Kid meaning |
|---|---|
| `step` | How many lessons finished (out of 5,000) |
| `train` / `val` | **The loss** — starts ~10 (clueless), we want **under 5** |
| `tok/s` | Reading speed: words per second |
| `⭐ new best` | It just got smarter than ever before! |
| `💾 saved` | Homework saved to Drive ✅ |

**Rules while training:**
- ❌ Don't close the tab
- ❌ Don't press Stop
- ✅ You CAN switch to another tab and come back
- ✅ Go eat dinner. The table keeps updating.

**⏱ How long?** `normal` ≈ **30–50 minutes** on the free GPU.

---

### When Colab disconnects (it will — that's normal!) 🔌

Google gives free GPUs for ~12 hours, then says *"session ended."*
**Your brain is SAFE** (it saved to Drive every 500 steps).

**How to continue:**
1. Click **Runtime → Reconnect** (or refresh the page)
2. Re-run cells **1 → 6** in order (they reload from Drive)
3. Training prints: `🔁 Resumed from step 2,500` — and keeps going!

This is why we chose `normal` the first time: it fits in one session.

---

### Cell 7 — Talk to your brain! 💬
**Press ▶** (the load cell), then **▶** again (the example), then **▶**
on the chat cell.

Example output:
```
Once upon a time the robot walked to the hill and the girl said ...
```
Then a box appears:
```
You  > _
```
**Type anything, press Enter.** Your brain answers!

**Chat commands:**
- `/temp 0.5` — more serious · `/temp 1.2` — more chaotic
- `/tokens 300` — longer answers
- `/quit` — leave

### What will it sound like? (honest truth 🦜)
- **After 5,000 steps:** a **clever parrot**. It copies the *style* of the
  books. Sometimes silly. Sometimes surprisingly good. That's a REAL brain
  you trained from zero — be proud! 🎉
- **It is NOT Jarvis yet.** Jarvis comes at Level 3–4 (more data, bigger
  brain, more training). We level up together.

---

## 🆘 TROUBLESHOOTING (when something goes red)

| Problem | Fix |
|---|---|
| `❌ NO GPU FOUND` | Runtime → Change runtime type → GPU → Save → Runtime → Restart runtime → re-run from cell 1 |
| `CUDA out of memory` | You probably ran two training cells. Click **Runtime → Restart runtime**, then run cells 1→6 ONCE each |
| `⚠️ Session crashed` / disconnected | Runtime → Reconnect → re-run cells in order. Training resumes! |
| `Downloading...` stuck | Wait 3 min. Still stuck? Runtime → Restart → start over (books are cached after first success) |
| `Permission denied: /content` | You're not in Colab anymore — reopen the notebook link from STEP 1 |
| Loss not going down | Wait until step 500+. Still flat at step 1000? Tell Arena: "loss is flat" |
| Red error text | Copy the **last line** of the error, paste it to Arena. I fix it. That's my job. |
| "GPU quota exceeded" | Free GPUs are busy at night in some countries. Wait 1 hour, or try Kaggle (see LEVEL UP) |
| Drive popup won't appear | Settings cell → set `SAVE_TO_GOOGLE_DRIVE = False` (you can re-run training later) |

**Stuck on anything?** Copy-paste the red error to me in Arena.
*"You break it, I fix it."* That's the deal. 🤝

---

## 💾 STEP 4 — Save your brain forever (optional, 10 min)

Your brain currently lives in Google Drive. To park it in your
**Hugging Face** garage:

1. In Colab: **Files → Refresh** (find `GuriLevel1` folder)
2. Download `ckpt_latest.pt` and `vocab.json` to your computer
3. On huggingface.co: **New model** → name it `guri-level1` → create
4. **Upload files** → drop both files in
5. Now your brain lives on the internet forever. 🌍

*(Arena can walk you through this step-by-step when you're ready.)*

---

## 📈 THE LEVEL-UP PLAN (your roadmap to 3 billion)

| Level | Knobs | What changes | When |
|---|---|---|---|
| **1** ✅ | ~8 million | You are here! Storyteller baby. | This week |
| **2** | ~50 million | More books + `HOW_LONG="long"` + slightly bigger brain (I'll edit the settings for you) | Weeks 2–4 |
| **3** | ~350 million | Multi-week training, smarter tricks, Kaggle+Colab team-up | Months 2–3 |
| **4** 🎯 | **3 billion** | Needs a compute plan — free tiers stretched to the max, and *maybe* a small saved-up cloud budget (we'll cross that bridge together, with numbers) | The final boss |

**Every level is trained FROM SCRATCH.** Your brain. Your food. Your rules.
No downloaded homework. 🤝

### Bonus free GPU: Kaggle (when Colab says "quota exceeded")
1. Make a free account at **kaggle.com**
2. **Code → New Notebook**
3. Upload this same notebook file
4. **Settings → GPU → P100/T4** → Save → Run
5. You now have **two** free computers tag-teaming your brain. 🤖🤖

---

## 📖 WORD DICTIONARY (grown-up words, kid meanings)

| Word | Kid meaning |
|---|---|
| **Parameter** | One tiny knob inside the brain that gets twisted during learning |
| **Token** | One little piece of a word the brain reads (like a puzzle piece) |
| **Tokenizer** | The machine that chops words into pieces / mini-dictionary |
| **Loss** | The brain's test score — high = confused, low = smart |
| **Epoch / Step** | One lesson. One more practice round |
| **GPU** | A fast chip that does millions of math problems at once |
| **Checkpoint** | A saved photo of the brain so nothing is ever lost |
| **Training** | Showing millions of examples and twisting the knobs, slowly |
| **Inference** | Talking to the brain AFTER training (the chat part) |
| **From scratch** | Starting from zero — no borrowed brain, no cheating |

---

## ❓ FAQ

**Q: Do I need to learn coding first?**
A: No. I write it. You press ▶. When something breaks, you paste the error
to me. Over time you'll *accidentally* start understanding the code —
that's how everyone learns. 😄

**Q: Is this really free?**
A: Yes. Colab free + Drive free + GitHub free + Hugging Face free.
Rule #1: **we never pay.** If a tool asks for money, close the tab and
tell me — I'll find a free way.

**Q: Will my friends be able to talk to my brain?**
A: After Level 1: only you (on your Drive). After Level 3: we can put it
on Hugging Face Spaces so anyone with the link can chat with YOUR brain. 🌍

**Q: Can I train on my own laptop instead?**
A: You *can* (the code runs on CPU too) but it's 50× slower — hours
instead of minutes. Colab is the way. Your laptop stays cool. 🌀

**Q: When do we start Level 2?**
A: When your loss gets under ~4.5 and you're bored of chatting with the
Level 1 brain. Come back to me and say: **"Let's do Level 2."**

---

*Now go press ▶ on cell 1. Your brain is waiting.* 🧠💪
