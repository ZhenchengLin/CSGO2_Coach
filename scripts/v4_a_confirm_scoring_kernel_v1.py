"""
V4-A Confirmation scoring arithmetic.

This module operates only on in-memory arrays.

It does not:
    read Confirmation files;
    select matches or target rows;
    load models;
    generate predictions;
    write scoring results;
    modify frozen research artifacts.

A future Scoring Runner must verify the final Manifest,
model identities, feature ordering, and target-row alignment
before calling these functions on Confirmation predictions.
"""

from __future__ import annotations

import numpy as np


CLASSES = 15
MATCHES = 20
RESAMPLES = 20_000
SEED = 20260919


def require(condition, message):
    if not condition:
        raise ValueError(message)


def row_log_loss(labels, probabilities):
    """
    Calculate per-row multiclass Log Loss.

    labels:
        One-dimensional integer class indices, 0 through 14.

    probabilities:
        Matrix with shape (number_of_rows, 15).

    The true-class probability must be strictly positive.
    No epsilon clipping or probability renormalization occurs.
    """

    y = np.asarray(labels)
    p = np.asarray(probabilities)

    require(
        y.ndim == 1 and y.size > 0,
        "Labels must be a nonempty 1D array.",
    )

    require(
        y.dtype.kind in "iu"
        and np.all((0 <= y) & (y < CLASSES)),
        "Labels must be integer class indices 0..14.",
    )

    require(
        p.ndim == 2
        and p.shape == (len(y), CLASSES),
        "Expected one 15-class probability row per label.",
    )

    require(
        np.issubdtype(p.dtype, np.number),
        "Non-numeric model probabilities.",
    )

    p = p.astype(np.float64)

    require(
        np.isfinite(p).all()
        and np.all(p >= 0),
        "Probabilities must be finite and nonnegative.",
    )

    require(
        np.all(
            np.isclose(
                p.sum(axis=1),
                1.0,
                rtol=0.0,
                atol=1e-6,
            )
        ),
        "Probability row sum differs from one.",
    )

    true_p = p[
        np.arange(len(y)),
        y,
    ]

    require(
        np.all(true_p > 0),
        "Zero true-class probability: "
        "stop; no epsilon clipping.",
    )

    return -np.log(true_p)


def summarize_paired_differences(
    match_ranks,
    control_loss,
    candidate_loss,
):
    """
    Calculate the two frozen paired comparison statistics.

    Each input contains rows from ONE prediction horizon only.

    match_ranks:
        Frozen candidate ranks associated with the rows.

    control_loss:
        Validated per-row Control Log Loss.

    candidate_loss:
        Validated per-row Candidate Log Loss.

    Both model-loss arrays must correspond to exactly the
    same eligible target rows in exactly the same order.

    This function requires 20 distinct frozen candidate ranks.
    It does not select or remove matches or rows.
    """

    ranks = np.asarray(match_ranks)

    control = np.asarray(
        control_loss,
        dtype=np.float64,
    )

    candidate = np.asarray(
        candidate_loss,
        dtype=np.float64,
    )

    require(
        ranks.ndim == 1
        and ranks.size > 0
        and ranks.dtype.kind in "iu",
        "Match ranks must be a nonempty 1D integer array.",
    )

    require(
        control.shape
        == candidate.shape
        == ranks.shape,
        "Control, Candidate and Match-rank rows "
        "must align exactly.",
    )

    require(
        np.isfinite(control).all()
        and np.isfinite(candidate).all(),
        "Loss arrays contain nonfinite values.",
    )

    require(
        np.all(control >= 0)
        and np.all(candidate >= 0),
        "Per-row Log Loss cannot be negative.",
    )

    unique = np.unique(ranks)

    require(
        len(unique) == MATCHES
        and np.all(
            (unique >= 1) & (unique <= 25)
        ),
        "Expected 20 distinct frozen candidate ranks "
        "between 1 and 25.",
    )

    difference = candidate - control

    counts = np.asarray(
        [
            (ranks == rank).sum()
            for rank in unique
        ],
        dtype=np.int64,
    )

    totals = np.asarray(
        [
            difference[ranks == rank].sum(
                dtype=np.float64
            )
            for rank in unique
        ],
        dtype=np.float64,
    )

    means = totals / counts

    pooled = float(
        totals.sum(dtype=np.float64)
        / counts.sum()
    )

    equal_match = float(
        means.mean(dtype=np.float64)
    )

    # Matches are ordered by frozen candidate_rank.
    #
    # Initialize the RNG independently for each horizon.
    # Reuse these exact sampled match identities for both
    # the pooled and equal-match statistics.

    rng = np.random.default_rng(SEED)

    draw = rng.integers(
        0,
        MATCHES,
        size=(RESAMPLES, MATCHES),
    )

    pooled_draw = (
        totals[draw].sum(axis=1)
        / counts[draw].sum(axis=1)
    )

    equal_draw = means[draw].mean(axis=1)

    pooled_ci = np.quantile(
        pooled_draw,
        [0.025, 0.975],
        method="linear",
    )

    equal_ci = np.quantile(
        equal_draw,
        [0.025, 0.975],
        method="linear",
    )

    return {
        "rows": int(ranks.size),

        "matches": MATCHES,

        "pooled_difference": pooled,

        "pooled_match_bootstrap_95ci": (
            pooled_ci.tolist()
        ),

        "equal_match_difference": equal_match,

        "equal_match_bootstrap_95ci": (
            equal_ci.tolist()
        ),

        "bootstrap_resamples": RESAMPLES,

        "bootstrap_seed": SEED,
    }
