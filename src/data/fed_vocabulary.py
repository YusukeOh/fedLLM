"""
Fed vocabulary extractor for Time-LLM Reprogramming.

Phase 1 task 1-4: Build a vocabulary dictionary from FOMC statements and
minutes.  The extracted tokens are later used to constrain the Reprogramming
layer (Phase 3, task 3-1) so that time-series patches are projected onto
a subspace of the LLM's embedding matrix spanned by Fed-relevant words.

Design choices
--------------
1. **Tokenizer-aligned extraction**: We extract *LLM token IDs*, not raw
   words.  A phrase like "labor market tightening" may map to multiple
   sub-word tokens depending on the tokenizer — all of them are included.
2. **Frequency filtering**: Tokens that appear in fewer than `min_doc_freq`
   documents are dropped to avoid idiosyncratic noise.
3. **Stopword handling**: Generic English stopwords are excluded by default
   since they carry no economic semantics.
4. **Multi-document support**: Vocabulary can be built from statements only,
   minutes only, or both.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# Common English stopwords (a compact set for filtering)
_STOPWORDS = frozenset(
    "a an the and or but in on at to for of is are was were be been being "
    "have has had do does did will would shall should may might can could "
    "i you he she it we they this that these those my your his her its our "
    "their what which who whom where when how if not no nor so as by with "
    "from up about into through during before after above below between "
    "out off over under again further then once here there all each every "
    "both few more most other some such only own same than too very just "
    "also now new because however well still even much back also while "
    "since let make like time".split()
)


def load_fomc_texts(
    fomc_dir: str | Path,
    doc_types: tuple[str, ...] = ("statements", "minutes"),
) -> list[str]:
    """Load all FOMC document texts from disk.

    Returns one string per document.
    """
    fomc_dir = Path(fomc_dir)
    texts = []
    for sub in doc_types:
        sub_dir = fomc_dir / sub
        if not sub_dir.exists():
            logger.warning("Directory not found: %s", sub_dir)
            continue
        for f in sorted(sub_dir.glob("*.txt")):
            text = f.read_text().strip()
            if text:
                texts.append(text)
    logger.info("Loaded %d FOMC documents from %s", len(texts), fomc_dir)
    return texts


def build_fed_vocabulary(
    texts: list[str],
    tokenizer,
    min_doc_freq: int = 3,
    max_vocab_size: Optional[int] = None,
    exclude_stopwords: bool = True,
) -> dict:
    """Build a Fed-specific vocabulary from FOMC document texts.

    Parameters
    ----------
    texts : list[str]
        Raw text of FOMC documents (one per document).
    tokenizer : PreTrainedTokenizer
        The LLM's tokenizer (e.g. LLaMA or GPT-2).
    min_doc_freq : int
        Minimum number of documents a token must appear in.
    max_vocab_size : int, optional
        If set, keep only the top-N most frequent tokens.
    exclude_stopwords : bool
        Whether to remove tokens whose decoded form is a stopword.

    Returns
    -------
    dict with keys:
        token_ids : list[int] — LLM vocabulary IDs that form the Fed vocabulary
        token_strings : list[str] — decoded token strings
        doc_freq : dict[int, int] — per-token document frequency
        total_tokens_scanned : int
        total_documents : int
    """
    doc_freq: Counter = Counter()
    token_total_freq: Counter = Counter()

    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        unique_ids = set(ids)
        for tid in unique_ids:
            doc_freq[tid] += 1
        for tid in ids:
            token_total_freq[tid] += 1

    # Filter by document frequency
    candidates = {
        tid for tid, df in doc_freq.items() if df >= min_doc_freq
    }

    # Optionally remove stopwords
    if exclude_stopwords:
        filtered = set()
        for tid in candidates:
            decoded = tokenizer.decode([tid]).strip().lower()
            # Remove pure punctuation and single characters
            if len(decoded) <= 1 or decoded in _STOPWORDS:
                continue
            if re.match(r"^[\W\d]+$", decoded):
                continue
            filtered.add(tid)
        candidates = filtered

    # Rank by total frequency
    ranked = sorted(candidates, key=lambda t: token_total_freq[t], reverse=True)

    if max_vocab_size is not None:
        ranked = ranked[:max_vocab_size]

    token_strings = [tokenizer.decode([tid]).strip() for tid in ranked]

    result = {
        "token_ids": ranked,
        "token_strings": token_strings,
        "doc_freq": {tid: doc_freq[tid] for tid in ranked},
        "total_tokens_scanned": sum(token_total_freq.values()),
        "total_documents": len(texts),
    }

    logger.info(
        "Fed vocabulary: %d tokens (from %d candidates, %d documents, min_doc_freq=%d)",
        len(ranked), len(doc_freq), len(texts), min_doc_freq,
    )
    return result


def save_vocabulary(vocab: dict, path: str | Path) -> None:
    """Save vocabulary to JSON for reproducibility."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    serializable = {
        "token_ids": vocab["token_ids"],
        "token_strings": vocab["token_strings"],
        "doc_freq": {str(k): v for k, v in vocab["doc_freq"].items()},
        "total_tokens_scanned": vocab["total_tokens_scanned"],
        "total_documents": vocab["total_documents"],
        "vocab_size": len(vocab["token_ids"]),
    }
    path.write_text(json.dumps(serializable, indent=2))
    logger.info("Vocabulary saved to %s (%d tokens)", path, len(vocab["token_ids"]))


def load_vocabulary(path: str | Path) -> dict:
    """Load a previously saved Fed vocabulary."""
    raw = json.loads(Path(path).read_text())
    raw["token_ids"] = [int(x) for x in raw["token_ids"]]
    raw["doc_freq"] = {int(k): v for k, v in raw["doc_freq"].items()}
    return raw


def get_fed_embedding_mask(vocab: dict, full_vocab_size: int) -> np.ndarray:
    """Return a boolean mask over the full LLM vocabulary.

    True at indices corresponding to Fed vocabulary tokens — used to
    constrain the Reprogramming layer's source embedding matrix.
    """
    mask = np.zeros(full_vocab_size, dtype=bool)
    for tid in vocab["token_ids"]:
        if tid < full_vocab_size:
            mask[tid] = True
    return mask
