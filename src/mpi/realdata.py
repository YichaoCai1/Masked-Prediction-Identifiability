"""E3 -- Residual mode MMSE on real corpora.

Implements ``documents/E3-real-corpora-spec.md``.

Both structural assumptions of the paper are statements about
``mmse_p(Z | X_V)`` -- a property of the **data law**, not of any trained model.
E1/E2 verify them on a synthetic law where every constant is predicted.  E3's
job is narrower: show that the *shape* both assumptions require actually occurs
in real data.  One curve, ``log mmse_hat(|V|)`` vs ``|V|``, exhibiting

* large ``|V|``: exponential decay (pinning; the slope estimates ``kappa``),
* small ``|V|``: bounded below, not collapsing (low-visibility; ``u_0``, ``L``),
* a nonempty gap between the two regimes.

The direction problem (spec section 1)
--------------------------------------
For any predictor, ``E[(Z - beta_hat)^2] >= mmse_p(Z | X_V)``, with equality iff
``beta_hat`` is the true posterior.  So **any classifier measures an upper
bound**, never a lower one, and the two assumptions need opposite sides:

======================  =========================  ==========================
assumption              needs                      valid estimator
======================  =========================  ==========================
pinning, large ``|V|``  upper bound that decays    any calibrated classifier
low-vis, small ``|V|``  lower bound that stays up  near-Bayes-optimal posterior
======================  =========================  ==========================

Hence two estimators on two disjoint ranges, mirroring the theory's own
structure: the two assumptions are proved by different arguments on disjoint
ranges of ``|V|``.

Division of labour between the corpora (spec section 2.2)
---------------------------------------------------------
Code vs prose separates *too* well: at moderate ``|V|`` the classifier is
near-perfect, ``mmse_hat`` hits the estimator's floor, and the fitted slope is
truncated rather than measured.  Two languages are harder to separate, so the
decay is gentler and the slope is resolvable before the floor.  Therefore
**code-vs-prose carries the "nonempty window" claim; the bilingual corpus
carries the numerical ``kappa_hat``.**
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from .config import CACHE_DIR, RunConfig
from .fits import ols
from .tableio import write_table

__all__ = ["Corpus", "run_e3", "V_SMALL", "V_BRIDGE", "V_LARGE", "V_GRID", "CORPORA"]

# -- |V| grid (spec section 2.4) -------------------------------------------
#: near-Bayes-optimal posterior arm
V_SMALL: tuple[int, ...] = (1, 2, 4, 8)
#: bridge sizes -- they catch the large-arm slope before it reaches the floor
V_BRIDGE: tuple[int, ...] = (12, 16, 24)
#: calibrated-classifier arm
V_LARGE: tuple[int, ...] = (32, 64, 128, 256, 512, 1024)
V_GRID: tuple[int, ...] = V_SMALL + V_BRIDGE + V_LARGE

#: counts below this are not evidence; they cap the realised backoff order
BACKOFF_COUNT_THRESHOLD = 20
#: fraction of n-gram occurrences that must clear the threshold for an order
BACKOFF_COVERAGE = 0.5
#: stupid-backoff discount
BACKOFF_GAMMA = 0.4

#: A calibrated predictor has ``E[beta(1-beta)] == E[(Z-beta)^2]``.  When the
#: Brier score exceeds ``mmse_hat`` by more than this factor, calibration has
#: failed for that cell and the upper-bound guarantee is void (spec section
#: 3.2), so the cell is censored out of the ``kappa_hat`` fit.
BRIER_RATIO_MAX = 2.0


# ==========================================================================
# Tokenisers -- one shared, *neutral* tokenizer per corpus (spec section 2.3)
# ==========================================================================


class HFTokenizer:
    """A general-purpose BPE tokenizer, GPT-2 by default.

    Neutrality is a correctness requirement, not a detail: a code-specialised
    tokenizer carries regime information in the tokenisation itself (dedicated
    indentation and bracket tokens), which leaks ``Z`` through structure rather
    than through the token distribution.  With a neutral tokenizer all regime
    information lives in the token *distribution*, which is exactly what
    ``X_V`` is in the theory.
    """

    def __init__(self, name: str = "gpt2"):
        from transformers import AutoTokenizer

        self.name = name
        self._tok = AutoTokenizer.from_pretrained(name)
        self.vocab_size = int(self._tok.vocab_size)

    def encode_batch(self, texts: list[str]) -> list[np.ndarray]:
        out = self._tok(texts, add_special_tokens=False)["input_ids"]
        return [np.asarray(ids, dtype=np.int32) for ids in out]


class ByteTokenizer:
    """UTF-8 bytes: 256 symbols, no download.  Offline fallback only."""

    name = "bytes"
    vocab_size = 256

    def encode_batch(self, texts: list[str]) -> list[np.ndarray]:
        return [
            np.frombuffer(t.encode("utf-8", errors="ignore"), dtype=np.uint8).astype(np.int32)
            for t in texts
        ]


def make_tokenizer(spec: str):
    if spec == "bytes":
        return ByteTokenizer()
    if spec.startswith("hf:"):
        return HFTokenizer(spec[3:])
    raise ValueError(f"unknown tokenizer {spec!r} (use 'bytes' or 'hf:<name>')")


# ==========================================================================
# Corpus registry (spec section 2.1-2.2)
# ==========================================================================


@dataclass(frozen=True)
class Side:
    """One regime of a corpus: where to stream it from and how it is licensed."""

    path: str
    config: str | None
    split: str
    field: str                       # dotted path into the row dict
    label: str
    licence: str
    kwargs: dict = field(default_factory=dict)


CORPORA: dict[str, tuple[Side, Side]] = {
    # Primary.  Regime label = source dataset, so ground truth needs no model.
    # Single language on the code side is deliberate: mixing languages blurs
    # the regime boundary and E3 wants a clean binary Z.
    "code_prose": (
        Side("codeparrot/github-code-clean", None, "train", "code",
             "Python code (permissive licences)",
             "per-file: MIT / Apache-2.0 / BSD-2-Clause / BSD-3-Clause",
             {"languages": ["Python"],
              "licenses": ["mit", "apache-2.0", "bsd-3-clause", "bsd-2-clause"],
              "trust_remote_code": True}),
        Side("allenai/c4", "en", "train", "text",
             "web prose (C4 en)", "ODC-BY"),
    ),
    # Secondary.  Connects to the short-segment language-identification
    # literature the paper cites; carries the numerical kappa_hat.
    "bilingual": (
        Side("Helsinki-NLP/opus-100", "de-en", "train", "translation.de",
             "German (OPUS-100)", "CC-BY-4.0"),
        Side("Helsinki-NLP/opus-100", "de-en", "train", "translation.en",
             "English (OPUS-100)", "CC-BY-4.0"),
    ),
}


@dataclass
class Corpus:
    """Two labelled sources, packed into equal-length documents.

    Documents are a fixed number of tokens on *both* sides, so document length
    carries no information about ``Z`` and cannot leak into the measurement
    (spec section 2.4).
    """

    name: str
    docs_plus: np.ndarray            # (n_plus, doc_tokens) int32
    docs_minus: np.ndarray           # (n_minus, doc_tokens) int32
    vocab_size: int
    tokenizer: str
    label_plus: str
    label_minus: str
    licence_plus: str
    licence_minus: str
    note: str = ""

    @property
    def doc_tokens(self) -> int:
        return int(self.docs_plus.shape[1])

    def stats(self) -> dict:
        return {
            "corpus": self.name,
            "label_plus": self.label_plus,
            "label_minus": self.label_minus,
            "licence_plus": self.licence_plus,
            "licence_minus": self.licence_minus,
            "tokenizer": self.tokenizer,
            "vocab_size": self.vocab_size,
            "n_docs_plus": int(self.docs_plus.shape[0]),
            "n_docs_minus": int(self.docs_minus.shape[0]),
            "doc_tokens": self.doc_tokens,
            "tokens_plus": int(self.docs_plus.size),
            "tokens_minus": int(self.docs_minus.size),
            "note": self.note,
        }


# -- streaming + packing ----------------------------------------------------


def _hf_stream(side: Side, limit_rows: int):
    from datasets import load_dataset

    ds = load_dataset(side.path, side.config, split=side.split, streaming=True,
                      **side.kwargs)
    keys = side.field.split(".")
    for i, row in enumerate(ds):
        if i >= limit_rows:
            break
        val = row
        for k in keys:
            val = val[k]
        if val:
            yield val


def _pack(texts, tokenizer, n_docs: int, doc_tokens: int,
          batch: int = 256, min_fraction: float = 0.25) -> np.ndarray:
    """Tokenise a text stream and pack it into up to ``n_docs`` fixed-length docs.

    Returns fewer documents if the source runs out -- OPUS-100 has exactly 1M
    sentence pairs, and its English side is shorter than its German side, so
    the two regimes exhaust at different points.  The caller re-balances.
    """
    out = np.empty((n_docs, doc_tokens), dtype=np.int32)
    buf: list[np.ndarray] = []
    have = filled = 0
    pending: list[str] = []

    def flush():
        nonlocal buf, have, filled
        for ids in tokenizer.encode_batch(pending):
            if ids.size == 0:
                continue
            buf.append(ids)
            have += ids.size
            while have >= doc_tokens and filled < n_docs:
                flat = np.concatenate(buf)
                out[filled] = flat[:doc_tokens]
                rest = flat[doc_tokens:]
                buf = [rest] if rest.size else []
                have = rest.size
                filled += 1
        pending.clear()

    for text in texts:
        pending.append(text)
        if len(pending) >= batch:
            flush()
            if filled >= n_docs:
                break
    if filled < n_docs and pending:
        flush()
    if filled < max(1, int(min_fraction * n_docs)):
        raise RuntimeError(
            f"source exhausted after only {filled}/{n_docs} documents of "
            f"{doc_tokens} tokens; lower --n-docs or --doc-tokens"
        )
    return out[:filled]


def _cache_path(corpus: str, side: str, tok: str, n_docs: int, doc_tokens: int) -> Path:
    key = f"{corpus}|{side}|{tok}|{n_docs}|{doc_tokens}"
    h = hashlib.sha256(key.encode()).hexdigest()[:16]
    return CACHE_DIR / "e3" / f"{corpus}_{side}_{h}.npy"


def build_corpus(name: str, tokenizer_spec: str, n_docs: int, doc_tokens: int,
                 limit_rows: int = 4_000_000, use_cache: bool = True,
                 verbose: bool = True) -> Corpus:
    if name == "markov":
        return build_markov_corpus(n_docs=n_docs, doc_tokens=doc_tokens)

    plus_side, minus_side = CORPORA[name]
    tok = make_tokenizer(tokenizer_spec)

    packed = {}
    for tag, side in (("plus", plus_side), ("minus", minus_side)):
        cache = _cache_path(name, tag, tok.name, n_docs, doc_tokens)
        if use_cache and cache.exists():
            packed[tag] = np.load(cache)
            if verbose:
                print(f"    {tag:5s}: {cache.name} (cached)")
            continue
        t0 = time.time()
        packed[tag] = _pack(_hf_stream(side, limit_rows), tok, n_docs, doc_tokens)
        if verbose:
            print(f"    {tag:5s}: {side.path} -> {len(packed[tag])} x {doc_tokens} "
                  f"tokens [{time.time()-t0:.0f}s]")
        if use_cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache, packed[tag])

    # Keep the two regimes balanced, so w = 1/2 by construction and the
    # estimators' implicit uniform prior is correct.
    keep = min(len(packed["plus"]), len(packed["minus"]))
    if verbose and keep < n_docs:
        print(f"    balanced to {keep} docs/side (source exhausted)")
    packed = {k: v[:keep] for k, v in packed.items()}

    return Corpus(
        name=name, docs_plus=packed["plus"], docs_minus=packed["minus"],
        vocab_size=tok.vocab_size, tokenizer=tok.name,
        label_plus=plus_side.label, label_minus=minus_side.label,
        licence_plus=plus_side.licence, licence_minus=minus_side.licence,
    )


def build_markov_corpus(n_docs: int = 8000, doc_tokens: int = 1152,
                        vocab: int = 96, seed: int = 20260725,
                        sharpness: float = 0.55) -> Corpus:
    """Two order-1 Markov sources -- an offline fixture, **not** a result.

    Kept so the whole E3 pipeline stays runnable and testable without network
    access.  Never reported.
    """
    rng = np.random.default_rng(seed)
    background = rng.dirichlet(np.full(vocab, 1.0), size=vocab)

    def transition() -> np.ndarray:
        M = sharpness * rng.dirichlet(np.full(vocab, 0.35), size=vocab) \
            + (1 - sharpness) * background
        return M / M.sum(axis=1, keepdims=True)

    def sample(M: np.ndarray, n: int) -> np.ndarray:
        cdf = np.cumsum(M, axis=1)
        docs = np.empty((n, doc_tokens), dtype=np.int32)
        state = rng.integers(0, vocab, size=n)
        for t in range(doc_tokens):
            u = rng.random(n)
            state = (cdf[state] < u[:, None]).sum(axis=1).clip(0, vocab - 1)
            docs[:, t] = state
        return docs

    return Corpus(
        name="markov", docs_plus=sample(transition(), n_docs),
        docs_minus=sample(transition(), n_docs), vocab_size=vocab,
        tokenizer="synthetic", label_plus="source A", label_minus="source B",
        licence_plus="n/a", licence_minus="n/a",
        note="offline validation fixture, not natural data",
    )


# ==========================================================================
# Splits and window sampling (spec section 2.4, section 4 step 1)
# ==========================================================================


@dataclass
class Splits:
    fit: np.ndarray
    calib: np.ndarray
    evalu: np.ndarray


def make_splits(n: int, seed: int = 0) -> Splits:
    """80/10/10 at the **document** level, so no document spans two splits."""
    idx = np.random.default_rng(seed).permutation(n)
    a, b = int(0.8 * n), int(0.9 * n)
    return Splits(idx[:a], idx[a:b], idx[b:])


def sample_windows(docs: np.ndarray, doc_ids: np.ndarray, V: int, n_windows: int,
                   rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """``n_windows`` windows of ``V`` contiguous tokens.

    A document is drawn uniformly, then a uniformly random offset inside it.
    Returns ``(windows, owning document id)``; the second array is what makes
    the bootstrap document-level.
    """
    T = docs.shape[1]
    if V > T:
        raise ValueError(f"V={V} exceeds document length {T}")
    pick = rng.integers(0, len(doc_ids), size=n_windows)
    off = rng.integers(0, T - V + 1, size=n_windows)
    rows = doc_ids[pick]
    windows = docs[rows[:, None], off[:, None] + np.arange(V)[None, :]]
    return windows, rows


# ==========================================================================
# Small arm: stupid-backoff n-gram posterior (spec section 3.1)
# ==========================================================================

_HASH_MULT = np.uint64(0x100000001B3)          # FNV-1a prime


def _rolling_codes(seqs: np.ndarray, order: int) -> list[np.ndarray]:
    """``codes[k][:, i]`` = rolling hash of ``seqs[:, i:i+k]``, ``k = 1..order``.

    A 64-bit rolling hash rather than an exact positional code: an exact code
    needs ``vocab^k`` values, which overflows 64 bits at ``k = 4`` already for a
    50k-token vocabulary.  With ~1e7 distinct keys the expected number of
    collisions is ~1e-5, far below every other error source here.
    """
    codes: list[np.ndarray] = [None]
    cur = seqs.astype(np.uint64)
    codes.append(cur)
    with np.errstate(over="ignore"):
        for k in range(2, order + 1):
            if cur.shape[1] < 2:
                cur = np.empty((seqs.shape[0], 0), dtype=np.uint64)
            else:
                cur = cur[:, :-1] * _HASH_MULT + seqs[:, k - 1:].astype(np.uint64)
            codes.append(cur)
    return codes


class _CountTable:
    """Sorted-key count table with vectorised lookup."""

    __slots__ = ("keys", "counts", "total")

    def __init__(self, raw: np.ndarray):
        self.keys, self.counts = np.unique(raw.ravel(), return_counts=True)
        self.total = int(self.counts.sum())

    def get(self, query: np.ndarray) -> np.ndarray:
        if self.keys.size == 0:
            return np.zeros(query.shape, dtype=np.float64)
        idx = np.clip(np.searchsorted(self.keys, query), 0, self.keys.size - 1)
        hit = self.keys[idx] == query
        return np.where(hit, self.counts[idx], 0).astype(np.float64)

    def coverage(self, threshold: int) -> float:
        """Fraction of occurrences that fall in types seen ``>= threshold`` times."""
        if self.total == 0:
            return 0.0
        keep = self.counts >= threshold
        return float(self.counts[keep].sum()) / self.total


class StupidBackoffLM:
    r"""Per-side stupid-backoff n-gram model.

    ``S(x_i | h) = C(h, x_i)/C(h)`` at the highest order with a non-zero
    context, otherwise ``gamma * S(x_i | h')`` at the next order down, with
    ``S`` at order 1 the unigram relative frequency.  No normalisation is
    needed because only the *ratio* between the two sides is used.

    Raw high-order n-grams estimate nothing here -- with a 50k-token vocabulary
    the 8-gram space is ``(5e4)^8`` and essentially every high-order n-gram is
    seen once or never.  The realised order is therefore capped at the highest
    ``n`` whose counts clear :data:`BACKOFF_COUNT_THRESHOLD` on at least
    :data:`BACKOFF_COVERAGE` of occurrences, and reported per ``|V|``.
    """

    def __init__(self, max_order: int, vocab_size: int, gamma: float = BACKOFF_GAMMA):
        self.max_order = max_order
        self.vocab_size = vocab_size
        self.gamma = gamma
        self.ngram: list[_CountTable | None] = [None] * (max_order + 1)
        self.context: list[_CountTable | None] = [None] * (max_order + 1)
        self.eff_order = 1

    def fit(self, docs: np.ndarray) -> "StupidBackoffLM":
        codes = _rolling_codes(docs, self.max_order)
        for k in range(1, self.max_order + 1):
            width = codes[k].shape[1]
            if width == 0:
                self.ngram[k] = _CountTable(np.empty(0, dtype=np.uint64))
                self.context[k] = _CountTable(np.empty(0, dtype=np.uint64))
                continue
            self.ngram[k] = _CountTable(codes[k])
            ctx = (np.zeros_like(codes[k]) if k == 1
                   else codes[k - 1][:, :width])
            self.context[k] = _CountTable(ctx)

        self.eff_order = 1
        for k in range(1, self.max_order + 1):
            if self.ngram[k].coverage(BACKOFF_COUNT_THRESHOLD) >= BACKOFF_COVERAGE:
                self.eff_order = k
            else:
                break
        return self

    def log_score(self, windows: np.ndarray, order_cap: int | None = None) -> np.ndarray:
        """``log S(x_1..x_V)`` by the chain rule, one value per window."""
        n, V = windows.shape
        order = min(self.eff_order, self.max_order, V)
        if order_cap is not None:
            order = min(order, order_cap)
        codes = _rolling_codes(windows, order)
        total = np.zeros(n)
        for i in range(V):
            # walk down from the highest available order until a context is seen
            score = np.zeros(n)
            done = np.zeros(n, dtype=bool)
            for k in range(min(order, i + 1), 0, -1):
                start = i - k + 1
                ctx_key = (np.zeros(n, dtype=np.uint64) if k == 1
                           else codes[k - 1][:, start])
                c_ctx = self.context[k].get(ctx_key)
                c_ng = self.ngram[k].get(codes[k][:, start])
                ok = (~done) & (c_ctx > 0) & (c_ng > 0)
                if ok.any():
                    depth = min(order, i + 1) - k          # how far we backed off
                    score[ok] = (self.gamma ** depth) * (c_ng[ok] / c_ctx[ok])
                    done |= ok
                if done.all():
                    break
            score = np.where(done, score, self.gamma ** order / self.vocab_size)
            total += np.log(np.maximum(score, 1e-300))
        return total


def _small_arm(corpus: Corpus, sp_plus: Splits, sp_minus: Splits,
               n_windows: int, seed: int):
    """Returns, per ``V``, ``(V, beta on eval, Z on eval, doc ids, eff_order)``."""
    max_order = max(V_SMALL)
    lm_plus = StupidBackoffLM(max_order, corpus.vocab_size).fit(
        corpus.docs_plus[sp_plus.fit])
    lm_minus = StupidBackoffLM(max_order, corpus.vocab_size).fit(
        corpus.docs_minus[sp_minus.fit])
    eff = min(lm_plus.eff_order, lm_minus.eff_order)

    results = []
    for V in V_SMALL:
        rng = np.random.default_rng(seed + V)
        wp, dp = sample_windows(corpus.docs_plus, sp_plus.evalu, V, n_windows, rng)
        wm, dm = sample_windows(corpus.docs_minus, sp_minus.evalu, V, n_windows, rng)
        llr = np.concatenate([
            lm_plus.log_score(wp, eff) - lm_minus.log_score(wp, eff),
            lm_plus.log_score(wm, eff) - lm_minus.log_score(wm, eff),
        ])
        z = np.concatenate([np.ones(len(wp)), np.zeros(len(wm))])
        # sides are balanced by construction, so the prior term vanishes
        beta = 1.0 / (1.0 + np.exp(-np.clip(llr, -700, 700)))
        docs = np.concatenate([dp, dm + corpus.docs_plus.shape[0]])
        results.append((V, beta, z, docs, min(eff, V)))
    return results


# ==========================================================================
# Large arm: calibrated bag-of-tokens logistic regression (spec section 3.2)
# ==========================================================================


#: hashed-bigram block width for the order-2 feature set
BIGRAM_BUCKETS = 1 << 20

#: the classifier feature sets, weakest first.  ``unigram_bigram`` contains
#: ``unigram`` as a sub-model (it keeps the exact unigram block and can zero the
#: bigram weights), so it is a strictly richer class -- which is what makes the
#: comparison between them a clean test of model saturation.
FEATURE_SETS: tuple[str, ...] = ("unigram", "unigram_bigram")


def _bag_of_tokens(windows: np.ndarray, vocab_size: int) -> sparse.csr_matrix:
    n, V = windows.shape
    rows = np.repeat(np.arange(n), V)
    data = np.ones(n * V, dtype=np.float32)
    M = sparse.csr_matrix((data, (rows, windows.ravel())), shape=(n, vocab_size))
    M.sum_duplicates()
    return M


def _features(windows: np.ndarray, vocab_size: int, feature_set: str
              ) -> sparse.csr_matrix:
    """Bag of tokens, optionally plus a hashed bag of bigrams.

    Bag-of-tokens discards word order, so it has a ceiling: past some ``|V|``
    extra tokens stop tightening the bound *for that model class*, which is
    indistinguishable from the data law itself flattening.  Adding bigrams is
    the cheapest strictly-richer class that restores order information, so
    comparing the two separates "the corpus decays slowly" from "the model
    saturated".  Exact bigram indices would need ``vocab^2`` columns, so the
    bigram block is hashed; the unigram block stays exact.
    """
    if feature_set == "unigram":
        return _bag_of_tokens(windows, vocab_size)
    if feature_set != "unigram_bigram":
        raise ValueError(f"unknown feature set {feature_set!r}")

    n, V = windows.shape
    blocks = [_bag_of_tokens(windows, vocab_size)]
    if V >= 2:
        bg = (_rolling_codes(windows, 2)[2] % np.uint64(BIGRAM_BUCKETS)).astype(np.int64)
        rows = np.repeat(np.arange(n), bg.shape[1])
        blocks.append(sparse.csr_matrix(
            (np.ones(bg.size, dtype=np.float32), (rows, bg.ravel())),
            shape=(n, BIGRAM_BUCKETS)))
    else:
        blocks.append(sparse.csr_matrix((n, BIGRAM_BUCKETS), dtype=np.float32))
    M = sparse.hstack(blocks, format="csr")
    M.sum_duplicates()
    return M


def calibration_floor(n_calib: int) -> float:
    """Smallest ``beta(1-beta)`` the calibrated estimator can resolve.

    A finite calibration sample cannot certify probability exactly 0 or 1, so
    probabilities are clipped to ``[f, 1-f]`` with ``f = 1/(2 n_calib)``.  A
    ``mmse_hat`` sitting at ``f(1-f)`` is therefore **censored** -- it records
    the resolution of the estimator, not the value of the data law.  Such cells
    are marked ``floored`` and excluded from the ``kappa_hat`` fit.
    """
    f = 1.0 / (2.0 * max(n_calib, 2))
    return f * (1.0 - f)


def _calibrate(scores_cal: np.ndarray, z_cal: np.ndarray, method: str):
    """Monotone score -> probability map fitted on the calibration split."""
    if method == "platt":
        from sklearn.linear_model import LogisticRegression

        lr = LogisticRegression(C=1e6, max_iter=1000)
        lr.fit(scores_cal.reshape(-1, 1), z_cal)
        return lambda s: lr.predict_proba(s.reshape(-1, 1))[:, 1]

    from sklearn.isotonic import IsotonicRegression

    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(scores_cal, z_cal)
    floor = 1.0 / (2.0 * max(len(z_cal), 2))
    return lambda s: np.clip(iso.predict(s), floor, 1.0 - floor)


def _large_arm(corpus: Corpus, sp_plus: Splits, sp_minus: Splits,
               V_list, n_windows: int, C: float, calibration: str, seed: int,
               feature_sets=FEATURE_SETS, verbose: bool = True):
    from sklearn.linear_model import LogisticRegression

    results, calib_curves = [], []
    for V in V_list:
        if V > corpus.doc_tokens:
            continue
        # identical windows for every feature set, so the comparison between
        # them isolates the model class and nothing else
        rng = np.random.default_rng(seed + 1000 + V)
        raw = {}
        for split_attr in ("fit", "calib", "evalu"):
            wp, dp = sample_windows(corpus.docs_plus, getattr(sp_plus, split_attr),
                                    V, n_windows, rng)
            wm, dm = sample_windows(corpus.docs_minus, getattr(sp_minus, split_attr),
                                    V, n_windows, rng)
            raw[split_attr] = (
                np.vstack([wp, wm]),
                np.concatenate([np.ones(len(wp)), np.zeros(len(wm))]),
                np.concatenate([dp, dm + corpus.docs_plus.shape[0]]),
            )

        for fs in feature_sets:
            t0 = time.time()
            X_fit, z_fit, _ = (_features(raw["fit"][0], corpus.vocab_size, fs),
                               raw["fit"][1], raw["fit"][2])
            X_cal, z_cal = _features(raw["calib"][0], corpus.vocab_size, fs), raw["calib"][1]
            X_ev, z_ev, d_ev = (_features(raw["evalu"][0], corpus.vocab_size, fs),
                                raw["evalu"][1], raw["evalu"][2])

            # one classifier per |V| *and* per feature set: the optimal
            # posterior differs across |V|, so a shared classifier is invalid.
            # C is held fixed across feature sets so the comparison varies the
            # model class alone; the richer class winning on Brier at every |V|
            # is the check that it is not merely over-regularised.
            clf = LogisticRegression(penalty="l2", C=C, solver="liblinear",
                                     dual=True, max_iter=20000)
            clf.fit(X_fit, z_fit)

            cal_fn = _calibrate(clf.decision_function(X_cal), z_cal, calibration)
            beta = cal_fn(clf.decision_function(X_ev))

            bins = np.linspace(0.0, 1.0, 11)
            which = np.clip(np.digitize(beta, bins) - 1, 0, 9)
            for b in range(10):
                sel = which == b
                if sel.sum() >= 20:
                    calib_curves.append(
                        {"corpus": corpus.name, "features": fs, "V": V,
                         "bin_lo": bins[b], "bin_hi": bins[b + 1],
                         "pred_mean": float(beta[sel].mean()),
                         "emp_freq": float(z_ev[sel].mean()), "n": int(sel.sum())})
            results.append((V, fs, beta, z_ev, d_ev, int(z_cal.size)))
            if verbose:
                print(f"    |V|={V:<5d} {fs:<14s} nnz={X_fit.nnz/1e6:5.1f}M "
                      f"[{time.time()-t0:5.1f}s]")
    return results, calib_curves


# ==========================================================================
# Estimation + document-level bootstrap
# ==========================================================================


def _bootstrap(beta: np.ndarray, z: np.ndarray, doc_ids: np.ndarray,
               B: int, seed: int):
    """``(mmse_hat, ci_lo, ci_hi, brier)`` with a **document-level** bootstrap.

    Windows from one document are correlated, so resampling windows would give
    a CI that is too narrow.  Documents are the resampling unit.
    """
    v = beta * (1.0 - beta)
    mmse = float(v.mean())
    brier = float(((z - beta) ** 2).mean())

    uniq, inv = np.unique(doc_ids, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    starts = np.searchsorted(inv[order], np.arange(uniq.size))
    ends = np.searchsorted(inv[order], np.arange(uniq.size), side="right")
    per_doc_sum = np.add.reduceat(v[order], starts) if uniq.size else np.array([])
    per_doc_n = (ends - starts).astype(float)

    rng = np.random.default_rng(seed)
    draws = np.empty(B)
    for b in range(B):
        pick = rng.integers(0, uniq.size, size=uniq.size)
        draws[b] = per_doc_sum[pick].sum() / per_doc_n[pick].sum()
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return mmse, float(lo), float(hi), brier


# ==========================================================================
# Driver
# ==========================================================================


def run_e3_corpus(cfg: RunConfig, corpus_name: str, tokenizer: str, n_docs: int,
                  doc_tokens: int, n_windows: int, n_bootstrap: int, C: float,
                  calibration: str, seed: int, verbose: bool = True):
    if verbose:
        print(f"  corpus {corpus_name!r}: {n_docs} docs/side x {doc_tokens} tokens")
    corpus = build_corpus(corpus_name, tokenizer, n_docs, doc_tokens, verbose=verbose)

    sp_plus = make_splits(corpus.docs_plus.shape[0], seed)
    sp_minus = make_splits(corpus.docs_minus.shape[0], seed + 1)

    rows: list[dict] = []

    if verbose:
        print("    small arm (stupid-backoff n-gram posterior)")
    for V, beta, z, docs, eff in _small_arm(corpus, sp_plus, sp_minus,
                                            n_windows, seed):
        mmse, lo, hi, brier = _bootstrap(beta, z, docs, n_bootstrap, seed + V)
        # The small arm is deliberately *uncalibrated* (spec section 3.1): the
        # plug-in backoff posterior is over-confident, so mmse_hat sits below
        # the truth.  That is the conservative direction for a lower bound --
        # if the underestimate stays up, the truth does too -- so these cells
        # are never censored.  The Brier score is recorded as the matching
        # rigorous upper bound.
        rows.append({"corpus": corpus.name, "arm": "small",
                     "features": "backoff-ngram", "V": V,
                     "mmse_hat": mmse, "ci_lo": lo, "ci_hi": hi, "brier": brier,
                     "brier_ratio": brier / mmse if mmse > 0 else np.nan,
                     "eff_order": eff, "floored": False, "floor": np.nan,
                     "n_windows": int(beta.size),
                     "n_docs": int(np.unique(docs).size)})

    if verbose:
        print("    large arm (calibrated bag-of-tokens logistic regression)")
    large, calib_curves = _large_arm(corpus, sp_plus, sp_minus,
                                     V_BRIDGE + V_LARGE, n_windows, C,
                                     calibration, seed, verbose=verbose)
    for V, fs, beta, z, docs, n_cal in large:
        mmse, lo, hi, brier = _bootstrap(beta, z, docs, n_bootstrap, seed + V)
        floor = calibration_floor(n_cal)
        ratio = brier / mmse if mmse > 0 else np.inf
        # Two ways a large-arm cell stops being a measurement of the data law
        # and becomes a measurement of the estimator: it reaches the clip floor,
        # or the classifier is confidently wrong often enough that no monotone
        # recalibration can restore E[beta(1-beta)] = Brier.  Both are censored.
        rows.append({"corpus": corpus.name, "arm": "large", "features": fs,
                     "V": V,
                     "mmse_hat": mmse, "ci_lo": lo, "ci_hi": hi, "brier": brier,
                     "brier_ratio": ratio, "eff_order": np.nan,
                     "floored": bool(mmse <= 1.05 * floor
                                     or ratio > BRIER_RATIO_MAX),
                     "floor": floor, "n_windows": int(beta.size),
                     "n_docs": int(np.unique(docs).size)})

    realdata = pd.DataFrame(rows)

    fit_rows, form_rows = [], []
    for (arm, fs), sub_all in realdata.groupby(["arm", "features"], sort=False):
        sub = sub_all[(sub_all.mmse_hat > 0) & (~sub_all.floored)].sort_values("V")
        if len(sub) < 3:
            continue
        V = sub.V.to_numpy(float)
        y = np.log(sub.mmse_hat.to_numpy())
        f = ols(V, y)
        fit_rows.append({
            "corpus": corpus.name, "arm": arm, "features": fs,
            "slope": f.slope, "intercept": f.intercept, "R2": f.r2,
            "n_points": f.n, "V_lo": f.x_lo, "V_hi": f.x_hi,
            "L_hat": -f.slope if arm == "small" else np.nan,
            "u0_hat": math.exp(f.intercept) if arm == "small" else np.nan,
            "kappa_hat": -f.slope if arm == "large" else np.nan,
            "n_floored_excluded": int(sub_all.floored.sum()),
        })
    # Asm. 4.1 assumes exponential decay in |V|.  Nothing forces the data to
    # obey that form, so both candidates are fitted and stored; which one wins
    # is a measurement, not an assumption.  Fitted twice: over each feature
    # set's own uncensored cells, and over the cells uncensored for *every*
    # feature set, so that comparing model classes compares like with like.
    large = realdata[realdata.arm == "large"]
    common = (set.intersection(*[set(g[~g.floored].V)
                                 for _, g in large.groupby("features")])
              if len(large) else set())
    for (arm, fs), sub_all in realdata.groupby(["arm", "features"], sort=False):
        for rng_name in ("uncensored", "common"):
            sub = sub_all[(sub_all.mmse_hat > 0) & (~sub_all.floored)]
            if rng_name == "common":
                if arm != "large":
                    continue
                sub = sub[sub.V.isin(common)]
            sub = sub.sort_values("V")
            if len(sub) < 3:
                continue
            V = sub.V.to_numpy(float)
            y = np.log(sub.mmse_hat.to_numpy())
            ex, pw = ols(V, y), ols(np.log(V), y)
            for model, fit, param in (("exponential", ex, -ex.slope),
                                      ("power_law", pw, pw.slope)):
                form_rows.append({
                    "corpus": corpus.name, "arm": arm, "features": fs,
                    "fit_range": rng_name, "model": model,
                    "slope": fit.slope, "intercept": fit.intercept,
                    "R2": fit.r2, "n_points": fit.n,
                    "V_lo": float(V.min()), "V_hi": float(V.max()),
                    "rate_or_exponent": param,
                })

    stats = pd.DataFrame([{**corpus.stats(), "n_windows": n_windows,
                           "n_bootstrap": n_bootstrap, "C": C,
                           "calibration": calibration,
                           "backoff_gamma": BACKOFF_GAMMA,
                           "backoff_threshold": BACKOFF_COUNT_THRESHOLD,
                           "brier_ratio_max": BRIER_RATIO_MAX,
                           "seed": seed}])
    return (realdata, pd.DataFrame(fit_rows), stats,
            pd.DataFrame(calib_curves), pd.DataFrame(form_rows))


def run_e3(
    cfg: RunConfig,
    corpora: tuple[str, ...] = ("code_prose", "bilingual"),
    tokenizer: str = "hf:gpt2",
    n_docs: int = 20000,
    doc_tokens: int = 1152,
    n_windows: int = 20000,
    n_bootstrap: int = 1000,
    C: float = 1.0,
    calibration: str = "isotonic",
    seed: int = 20260725,
    results_dir: Path | None = None,
    verbose: bool = True,
):
    t0 = time.time()
    all_rd, all_fits, all_stats, all_cal, all_form = [], [], [], [], []
    for name in corpora:
        rd, fits, stats, cal, form = run_e3_corpus(
            cfg, name, tokenizer, n_docs, doc_tokens, n_windows, n_bootstrap,
            C, calibration, seed, verbose)
        all_rd.append(rd)
        all_fits.append(fits)
        all_stats.append(stats)
        if len(cal):
            all_cal.append(cal)
        if len(form):
            all_form.append(form)

    realdata = pd.concat(all_rd, ignore_index=True)
    fits = pd.concat(all_fits, ignore_index=True) if all_fits else pd.DataFrame()
    stats = pd.concat(all_stats, ignore_index=True)
    calib = pd.concat(all_cal, ignore_index=True) if all_cal else pd.DataFrame()
    form = pd.concat(all_form, ignore_index=True) if all_form else pd.DataFrame()

    write_table(realdata, "realdata", cfg, results_dir)
    write_table(fits, "realdata_fits", cfg, results_dir)
    write_table(stats, "realdata_stats", cfg, results_dir)
    write_table(calib, "realdata_calib", cfg, results_dir)
    write_table(form, "realdata_form", cfg, results_dir)

    if verbose:
        print(f"  E3 done in {time.time()-t0:.1f}s")
        print(realdata.drop(columns=["floor"]).to_string(index=False))
        print()
        print(fits.to_string(index=False))
        print()
        print("functional form (exponential is what Asm. 4.1 assumes); the "
              "'common' range compares model classes like with like:")
        print(form[form.fit_range == "common"]
              .pivot_table(index=["corpus", "features"], columns="model",
                           values=["R2", "rate_or_exponent"]).to_string())
    return {"realdata": realdata, "fits": fits, "stats": stats,
            "calib": calib, "form": form}
