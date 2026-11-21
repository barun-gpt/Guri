"""
Guri Level 1 data tools — the brain's food pipeline.

1. Read plain text files.
2. Build a mini-dictionary (tokenizer) FROM our own text — not downloaded.
3. Turn text into number lists the brain can eat.
"""

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch

# Special "meta" tokens every brain needs.
PAD, UNK, BOS, EOS = "<pad>", "<unk>", "<bos>", "<eos>"
SPECIAL_TOKENS = [PAD, UNK, BOS, EOS]

# Split text into words and punctuation marks: "Hello, world!" -> [hello, ,, world, !]
_TOKEN_RE = re.compile(r"\w+|[^\w\s]")


def split_tokens(text: str) -> list[str]:
    """Lowercase and chop text into little pieces (words + punctuation)."""
    return _TOKEN_RE.findall(text.lower())


class SimpleTokenizer:
    """A from-scratch word-level tokenizer (our own mini-dictionary)."""

    def __init__(self, token_to_id: dict[str, int]):
        self.token_to_id = token_to_id
        self.id_to_token = {i: t for t, i in token_to_id.items()}

    @classmethod
    def build(cls, text: str, max_vocab: int = 16000, min_freq: int = 2) -> "SimpleTokenizer":
        """Grow the dictionary from the words that actually appear in our food."""
        counts = Counter(split_tokens(text))
        token_to_id = {t: i for i, t in enumerate(SPECIAL_TOKENS)}
        for token, freq in counts.most_common():
            if freq < min_freq or len(token_to_id) >= max_vocab:
                break
            token_to_id[token] = len(token_to_id)
        return cls(token_to_id)

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def encode(self, text: str, add_eos: bool = False) -> list[int]:
        ids = [self.token_to_id.get(t, self.token_to_id[UNK]) for t in split_tokens(text)]
        if add_eos:
            ids.append(self.token_to_id[EOS])
        return ids

    def decode(self, ids) -> str:
        words = []
        for i in ids:
            tok = self.id_to_token.get(int(i), UNK)
            if tok in (PAD, BOS, EOS):
                if tok == EOS:
                    words.append("\n")
                continue
            words.append(tok)
        # Glue words back together with spaces; keep punctuation snug.
        out = ""
        for w in words:
            if not out:
                out = w
            elif re.fullmatch(r"[^\w]", w):
                out += w
            else:
                out += " " + w
        return out

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.token_to_id, ensure_ascii=False, indent=0))

    @classmethod
    def load(cls, path: str | Path) -> "SimpleTokenizer":
        return cls(json.loads(Path(path).read_text()))


def build_corpus(paths: list[str | Path], max_files: int | None = None) -> str:
    """Read and stitch text files into one big scroll of brain food."""
    chunks = []
    for p in (paths[:max_files] if max_files else paths):
        text = Path(p).read_text(encoding="utf-8", errors="ignore")
        chunks.append(text.strip())
    return "\n\n".join(chunks)


def encode_files(paths: list[str | Path], tokenizer: SimpleTokenizer) -> np.ndarray:
    """Turn every file into token numbers, one after another on a long tape.

    Each file ends with a real EOS token so the brain learns where stories end.
    """
    ids: list[int] = []
    for p in paths:
        text = Path(p).read_text(encoding="utf-8", errors="ignore")
        ids.extend(tokenizer.encode(text, add_eos=True))
    return np.asarray(ids, dtype=np.int32)


def train_val_split(tokens: np.ndarray, val_fraction: float = 0.03):
    """Keep a small slice of tokens aside as a 'practice test' set."""
    n_val = max(1, int(len(tokens) * val_fraction))
    if len(tokens) - n_val <= 1:
        raise ValueError("Not enough data to train — add more text files.")
    return tokens[:-n_val], tokens[-n_val:]


def get_batch(
    data: np.ndarray, block_size: int, batch_size: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Grab random windows of text — the brain's snack for one step."""
    starts = np.random.randint(0, len(data) - block_size - 1, size=batch_size)
    x = np.stack([data[s : s + block_size] for s in starts])
    y = np.stack([data[s + 1 : s + 1 + block_size] for s in starts])
    return (
        torch.from_numpy(x.astype(np.int64)).to(device),
        torch.from_numpy(y.astype(np.int64)).to(device),
    )
