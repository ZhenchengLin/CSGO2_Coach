"""
V4-A Confirmation secondary scoring arithmetic.

Accepts in-memory arrays only. Does not read Confirmation
data, load models, generate predictions, or write results.

The future Scoring Runner must verify the frozen Manifest,
row alignment, class order, and model identities before
passing arrays to this module.
"""

from __future__ import annotations

import numpy as np

from v4_a_confirm_scoring_kernel_v1 import (
    CLASSES,
    MATCHES,
    require,
    row_log_loss,
)


def summarize_secondary_metrics(
    labels,
    probabilities,
    match_ranks,
    class_order,
):
    """
    Calculate one model's secondary metrics for ONE horizon.

    labels:       Integer target-class indices.
    probabilities: Corresponding (n_rows, 15) probabilities.
    match_ranks:  Frozen candidate rank for each target row.
    class_order:  Exact frozen sequence of 15 class names.
    """

    y = np.asarray(labels)
    p = np.asarray(probabilities)
    ranks = np.asarray(match_ranks)

    # Reuse the primary kernel's frozen probability, label,
    # row-sum, and zero-true-class validation rules.
    losses = row_log_loss(y, p)
    p64 = p.astype(np.float64)

    require(
        ranks.ndim == 1
        and ranks.shape == y.shape
        and ranks.dtype.kind in "iu",
        "Match-rank rows must align with labels.",
    )

    unique_ranks = np.unique(ranks)

    require(
        len(unique_ranks) == MATCHES
        and np.all((unique_ranks >= 1) & (unique_ranks <= 25)),
        "Expected 20 distinct frozen candidate ranks.",
    )

    require(
        isinstance(class_order, (list, tuple))
        and len(class_order) == CLASSES
        and all(
            isinstance(name, str) and name
            for name in class_order
        )
        and len(set(class_order)) == CLASSES,
        "Expected 15 distinct, nonempty class names.",
    )

    # Stable sorting preserves lower class-index priority
    # when two classes have exactly equal probabilities.
    ordering = np.argsort(
        -p64,
        axis=1,
        kind="stable",
    )

    predicted = ordering[:, 0]
    top2 = ordering[:, :2]

    correct = predicted == y

    top2_correct = np.any(
        top2 == y[:, None],
        axis=1,
    )

    confidence = p64.max(axis=1)

    true_probability = p64[
        np.arange(y.size),
        y,
    ]

    # Frozen Brier definition: sum across all 15 classes.
    # There is no division by the number of classes.
    brier_per_row = (
        np.sum(p64**2, axis=1)
        - 2.0 * true_probability
        + 1.0
    )

    # Floating-point roundoff can make a mathematically
    # zero Brier score slightly negative. Do not clip:
    # report the value computed by the frozen formula.
    brier = float(
        np.mean(brier_per_row, dtype=np.float64)
    )

    per_zone = []

    f1_values = []

    for index, name in enumerate(class_order):
        actual_mask = y == index
        predicted_mask = predicted == index

        support = int(actual_mask.sum())

        true_positive = int(
            np.count_nonzero(
                actual_mask & predicted_mask
            )
        )

        false_positive = int(
            np.count_nonzero(
                ~actual_mask & predicted_mask
            )
        )

        false_negative = support - true_positive

        denominator = (
            2 * true_positive
            + false_positive
            + false_negative
        )

        # Equivalent to macro-F1 with all 15 labels
        # included and zero_division=0.
        f1_values.append(
            2.0 * true_positive / denominator
            if denominator > 0
            else 0.0
        )

        per_zone.append({
            "class_index": index,
            "zone": name,
            "support": support,
            "recall": (
                true_positive / support
                if support > 0
                else None
            ),
        })

    # Equal-match Log Loss: average within each match,
    # then average the 20 match means equally.
    per_match = []

    for rank in unique_ranks:
        mask = ranks == rank

        per_match.append({
            "candidate_rank": int(rank),
            "rows": int(mask.sum()),
            "mean_log_loss": float(
                losses[mask].mean(dtype=np.float64)
            ),
        })

    equal_match_log_loss = float(
        np.mean(
            [record["mean_log_loss"] for record in per_match],
            dtype=np.float64,
        )
    )

    # ECE: ten fixed bins. Bins 0..8 are [lo, hi);
    # bin 9 is [0.9, 1.0].
    edges = np.linspace(0.0, 1.0, 11)

    ece = 0.0
    calibration_bins = []

    for index in range(10):
        lower = float(edges[index])
        upper = float(edges[index + 1])

        if index == 9:
            mask = (
                (confidence >= lower)
                & (confidence <= upper)
            )
        else:
            mask = (
                (confidence >= lower)
                & (confidence < upper)
            )

        count = int(mask.sum())

        if count == 0:
            calibration_bins.append({
                "bin": index,
                "rows": 0,
                "mean_confidence": None,
                "accuracy": None,
            })
            continue

        mean_confidence = float(
            confidence[mask].mean(dtype=np.float64)
        )

        bin_accuracy = float(
            correct[mask].mean(dtype=np.float64)
        )

        ece += (
            count / len(y)
            * abs(mean_confidence - bin_accuracy)
        )

        calibration_bins.append({
            "bin": index,
            "rows": count,
            "mean_confidence": mean_confidence,
            "accuracy": bin_accuracy,
        })

    require(
        sum(item["rows"] for item in calibration_bins)
        == len(y),
        "ECE binning did not cover every eligible row.",
    )

    return {
        "rows": int(len(y)),
        "matches": MATCHES,
        "log_loss": float(
            losses.mean(dtype=np.float64)
        ),
        "equal_match_log_loss": equal_match_log_loss,
        "multiclass_brier": brier,
        "accuracy": float(
            correct.mean(dtype=np.float64)
        ),
        "macro_f1": float(
            np.mean(f1_values, dtype=np.float64)
        ),
        "top2_accuracy": float(
            top2_correct.mean(dtype=np.float64)
        ),
        "per_zone": per_zone,
        "calibration": {
            "top1_ece": float(ece),
            "bins": calibration_bins,
        },
        "per_match": per_match,
    }
