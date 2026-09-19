"""V3 frozen-model independent confirmation evaluator.

Run modes:
  uv run python scripts/v3_confirm_evaluator.py --mode dev-test
  uv run python scripts/v3_confirm_evaluator.py --mode score
  uv run python scripts/v3_confirm_evaluator.py --mode close

Do not alter this source or any frozen research artifact after --score.
"""
from __future__ import annotations

import argparse
import ast
import csv
import gc
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import polars as pl
from scipy.spatial import cKDTree
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

from cs2_tactical_intelligence.v0.features import V0_PLAYER_PROPS, has_c4
from cs2_tactical_intelligence.v0.timing import open_v0_demo


MANIFEST = Path("docs/v3_confirm_manifest.csv")
IDENTITY = Path("docs/v3_confirm_manifest_identity.json")
PROTOCOL = Path("docs/v3_confirm_scoring_protocol_v1.json")
FREEZE = Path("docs/v3_confirm_model_freeze.json")
TARGET_BUILDER = Path("scripts/build_v3_confirm_candidate_targets.py")
MAPPING = Path("docs/v3_macro_zone_mapping_v1_frozen.json")

SPATIAL = Path(
    "data/interim/v3_b1_player_semantic_samples_gate1_exact_v1.parquet"
)
DEV_MOTION = Path("data/interim/v3_b1_motion_inputs_v1.parquet")
DEV_MANIFEST = Path("docs/v3_dev_manifest.csv")

MOTION_OUT = Path("data/interim/v3_confirm_motion_v1.parquet")
PRED_OUT = Path("data/interim/v3_confirm_predictions_v1.parquet")
RESULTS = Path("docs/v3_confirm_results_v1.json")
CLOSURE = Path("docs/v3_closure_v1.md")

N_CLASSES = 15
MIN_FREE_BYTES = 12 * 1024**3


def require(condition, message):
    if not condition:
        raise RuntimeError("STOP: " + message)


def sha256(path):
    digest = hashlib.sha256()

    with Path(path).open("rb") as file:
        for block in iter(
            lambda: file.read(4 * 1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text())


def storage_guard():
    free = shutil.disk_usage(".").free

    require(
        free >= MIN_FREE_BYTES,
        f"Available disk below 12 GiB: {free / 1024**3:.2f} GiB",
    )


def verify_sources():
    required = [
        MANIFEST,
        IDENTITY,
        PROTOCOL,
        FREEZE,
        TARGET_BUILDER,
        MAPPING,
        SPATIAL,
        DEV_MOTION,
        DEV_MANIFEST,
    ]

    for path in required:
        require(path.is_file(), f"Missing required file: {path}")

    identity = load_json(IDENTITY)
    protocol = load_json(PROTOCOL)
    freeze = load_json(FREEZE)

    manifest_sha = sha256(MANIFEST)

    require(
        identity["manifest_sha256"] == manifest_sha,
        "Confirmation manifest SHA changed.",
    )

    require(
        protocol["manifest_sha256"] == manifest_sha,
        "Scoring protocol references a different manifest.",
    )

    require(
        protocol["model_freeze_sha256"] == sha256(FREEZE),
        "Scoring protocol model-freeze SHA mismatch.",
    )

    require(
        protocol["status"] == "FROZEN_BEFORE_CONFIRMATION_SCORING",
        "Scoring protocol is not frozen.",
    )

    require(
        protocol["candidate"] == "B3_TABULAR_MAP_AWARE_V1",
        "Unexpected confirmation candidate.",
    )

    require(
        protocol["primary_comparator"] == "B2_ZONE_MARKOV_V1",
        "Unexpected primary comparator.",
    )

    require(
        protocol["horizons_sec"] == [5, 10],
        "Unexpected confirmation horizons.",
    )

    require(
        protocol["primary_metric"] == "multiclass_log_loss",
        "Unexpected primary metric.",
    )

    bootstrap = protocol["uncertainty"]

    require(
        bootstrap["method"] == "paired nonparametric match bootstrap",
        "Unexpected bootstrap method.",
    )

    require(
        bootstrap["resamples"] == 10000
        and bootstrap["seed"] == 42,
        "Unexpected bootstrap configuration.",
    )

    for artifact in freeze["artifacts"].values():
        path = Path(artifact["path"])

        require(
            path.is_file() and sha256(path) == artifact["sha256"],
            f"Frozen model artifact missing or changed: {path}",
        )

    require(
        sha256(SPATIAL) == freeze["b1"]["spatial_cache_sha256"],
        "Frozen B1 spatial cache changed.",
    )

    require(
        sha256(DEV_MOTION) == freeze["b1"]["motion_cache_sha256"],
        "Frozen development motion cache changed.",
    )

    with MANIFEST.open(newline="") as file:
        rows = list(csv.DictReader(file))

    require(
        len(rows) == 20,
        "Confirmation manifest must contain exactly 20 matches.",
    )

    ranks = [int(row["candidate_rank"]) for row in rows]

    require(
        ranks == list(range(1, 21)),
        "Frozen confirmation candidate order changed.",
    )

    require(
        len({row["demo_filename"] for row in rows}) == 20,
        "Duplicate confirmation demo filename.",
    )

    require(
        len({row["sha256"] for row in rows}) == 20,
        "Duplicate confirmation demo SHA.",
    )

    return rows, protocol, freeze


def load_frozen_bomb_helpers():
    """
    Reuse function definitions from the existing frozen target builder.

    Its top-level confirmation intake pipeline is NOT executed here.
    """

    source = ast.parse(
        TARGET_BUILDER.read_text(),
        filename=str(TARGET_BUILDER),
    )

    names = {
        "build_events_by_round",
        "latest_event_at_or_before",
        "materialize_snapshots",
        "map_place",
        "build_drop_lookup",
        "resolve_unplanted_bomb",
    }

    functions = [
        node
        for node in source.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]

    require(
        {node.name for node in functions} == names,
        "Required frozen bomb helper is missing.",
    )

    from bisect import bisect_right

    mapping = load_json(MAPPING)

    class_order = list(mapping["zones"])

    require(
        len(class_order) == N_CLASSES,
        "Frozen macro-zone count changed.",
    )

    place_to_zone = {}

    for zone, places in mapping["zones"].items():
        for place in places:
            require(
                place not in place_to_zone,
                f"Duplicate fine place in frozen mapping: {place}",
            )

            place_to_zone[place] = zone

    environment = {
        "pl": pl,
        "require": require,
        "has_c4": has_c4,
        "bisect_right": bisect_right,
        "place_to_zone": place_to_zone,
    }

    module = ast.fix_missing_locations(
        ast.Module(body=functions, type_ignores=[])
    )

    exec(
        compile(module, str(TARGET_BUILDER), "exec"),
        environment,
    )

    return environment, class_order, place_to_zone


def compute_motion(
    demo_path,
    source_rows,
    environment,
    *,
    dev_expected=None,
):
    """
    Reconstruct causal motion from the exact snapshot 64 ticks earlier.

    Only current and past information enters the velocity calculation.
    Future target labels are not consulted.
    """

    demo_path = Path(demo_path)

    demo = open_v0_demo(
        demo_path,
        verbose=False,
    )

    require(
        demo.header.get("map_name") == "de_mirage",
        f"Unexpected map: {demo_path.name}",
    )

    demo.parse(
        player_props=[
            *V0_PLAYER_PROPS,
            "last_place_name",
        ]
    )

    keys = {}

    for row in source_rows.iter_rows(named=True):
        key = (
            int(row["round_num"]),
            int(row["current_nominal_tick"]),
            int(row["current_tick"]),
        )

        keys.setdefault(key, row)

    prior_keys = {
        (round_num, tick - 64)
        for round_num, _, tick in keys
    }

    snapshots = environment["materialize_snapshots"](
        demo,
        prior_keys,
    )

    events = environment["build_events_by_round"](demo)

    drop_lookup, _ = environment["build_drop_lookup"](
        demo,
        demo_path.name,
    )

    result = []

    for (round_num, nominal_tick, current_tick), row in sorted(
        keys.items()
    ):
        prior_tick = current_tick - 64

        snapshot = snapshots.get(
            (round_num, prior_tick)
        )

        require(
            snapshot is not None,
            (
                f"Missing exact prior snapshot: {demo_path.name}, "
                f"round={round_num}, current_tick={current_tick}"
            ),
        )

        prior = environment["resolve_unplanted_bomb"](
            round_num=round_num,
            tick=prior_tick,
            snapshot=snapshot,
            events=events.get(round_num, []),
            drop_lookup=drop_lookup,
            state_error="PRIOR_BOMB_STATE_UNRESOLVED",
            place_error="PRIOR_BOMB_PLACE_UNRESOLVED",
        )

        require(
            prior["ok"],
            (
                f"Unresolved prior bomb state: {demo_path.name}, "
                f"round={round_num}, tick={prior_tick}, "
                f"reason={prior.get('error')}"
            ),
        )

        values = {
            "demo_filename": demo_path.name,
            "round_num": round_num,
            "current_nominal_tick": nominal_tick,
            "current_tick": current_tick,
            "prior_nominal_tick": nominal_tick - 64,
            "prior_tick": prior_tick,
            "prior_source": prior["source"],
            "elapsed_ticks": 64,
            "elapsed_sec": 1.0,
            "status": "RESOLVED",
        }

        for axis in "XYZ":
            values[f"velocity_{axis}"] = (
                float(row[f"current_bomb_{axis}"])
                - float(prior[axis])
            )

        if dev_expected is not None:
            expected = dev_expected.get(
                (round_num, nominal_tick, current_tick)
            )

            require(
                expected is not None,
                "Development motion key missing.",
            )

            require(
                int(expected["prior_tick"]) == prior_tick
                and expected["prior_source"] == prior["source"],
                (
                    f"Development prior-state mismatch: "
                    f"round={round_num}, nominal={nominal_tick}"
                ),
            )

            for axis in "XYZ":
                require(
                    np.isclose(
                        values[f"velocity_{axis}"],
                        float(expected[f"velocity_{axis}"]),
                        atol=1e-7,
                        rtol=1e-9,
                    ),
                    (
                        f"Development motion parity failed: "
                        f"round={round_num}, nominal={nominal_tick}, "
                        f"axis={axis}"
                    ),
                )

        result.append(values)

    del demo
    del snapshots
    del events
    del drop_lookup

    gc.collect()

    return pl.DataFrame(result)


def dev_test():
    """
    Pre-score motion parity test using the frozen development corpus.

    Does not run confirmation model inference or confirmation metrics.
    """

    verify_sources()

    environment, _, _ = load_frozen_bomb_helpers()

    manifest = pl.read_csv(
        DEV_MANIFEST,
        infer_schema_length=None,
    )

    motion = pl.read_parquet(DEV_MOTION)

    require(
        motion.height == 14869,
        "Development motion row count changed.",
    )

    require(
        motion.filter(pl.col("status") != "RESOLVED").height == 0,
        "Development motion contains unresolved rows.",
    )

    demo_name = None
    demo_entry = None

    for candidate in motion["demo_filename"].unique().to_list():
        entry = manifest.filter(
            pl.col("demo_filename") == candidate
        )

        if (
            entry.height == 1
            and Path(entry["path"][0]).is_file()
        ):
            demo_name = candidate
            demo_entry = entry
            break

    require(
        demo_name is not None,
        "No available development demo for motion parity test.",
    )

    expected_rows = motion.filter(
        pl.col("demo_filename") == demo_name
    )

    expected = {
        (
            int(row["round_num"]),
            int(row["current_nominal_tick"]),
            int(row["current_tick"]),
        ): row
        for row in expected_rows.iter_rows(named=True)
    }

    require(
        len(expected) == expected_rows.height,
        "Development motion cache contains duplicate keys.",
    )

    recreated = compute_motion(
        demo_entry["path"][0],
        expected_rows,
        environment,
        dev_expected=expected,
    )

    require(
        recreated.height == expected_rows.height,
        "Development motion row-count parity failed.",
    )

    print(
        f"DEVELOPMENT MOTION PARITY PASS: {demo_name}, "
        f"{recreated.height} observations"
    )

    print(
        "NO confirmation model inference or metrics performed."
    )


def assemble_confirmation_data(rows, environment):
    frames = []

    total_plus5 = 0
    total_plus10 = 0

    for record in rows:
        storage_guard()

        rank = int(record["candidate_rank"])

        demo_path = Path(record["path"])

        require(
            demo_path.is_file(),
            f"Rank {rank}: raw demo missing.",
        )

        require(
            sha256(demo_path) == record["sha256"],
            f"Rank {rank}: raw demo SHA mismatch.",
        )

        target_path = Path(
            f"data/interim/v3_confirm_intake/"
            f"rank_{rank:02d}/targets.parquet"
        )

        require(
            target_path.is_file(),
            f"Rank {rank}: target file missing.",
        )

        targets = pl.read_parquet(target_path)

        require(
            targets["demo_filename"].n_unique() == 1
            and targets["demo_filename"][0] == record["demo_filename"],
            f"Rank {rank}: target demo identity mismatch.",
        )

        plus5 = targets.filter(
            pl.col("horizon_sec") == 5
        ).height

        plus10 = targets.filter(
            pl.col("horizon_sec") == 10
        ).height

        require(
            plus5 == int(record["valid_plus5"])
            and plus10 == int(record["valid_plus10"]),
            f"Rank {rank}: target row counts differ from intake audit.",
        )

        key = [
            "demo_filename",
            "round_num",
            "current_nominal_tick",
            "horizon_sec",
        ]

        require(
            targets.unique(subset=key).height == targets.height,
            f"Rank {rank}: duplicate target rows.",
        )

        require(
            set(targets["horizon_sec"].unique().to_list()) == {5, 10},
            f"Rank {rank}: unexpected target horizons.",
        )

        unique_current = targets.unique(
            subset=[
                "round_num",
                "current_nominal_tick",
                "current_tick",
            ]
        )

        motion = compute_motion(
            demo_path,
            unique_current,
            environment,
        )

        require(
            motion.height == unique_current.height,
            f"Rank {rank}: motion row count mismatch.",
        )

        joined = targets.join(
            motion,
            on=[
                "demo_filename",
                "round_num",
                "current_nominal_tick",
                "current_tick",
            ],
            how="left",
        )

        require(
            joined.height == targets.height,
            f"Rank {rank}: motion join changed target row count.",
        )

        require(
            joined["status"].null_count() == 0,
            f"Rank {rank}: missing motion history.",
        )

        require(
            joined.filter(
                pl.col("status") != "RESOLVED"
            ).height == 0,
            f"Rank {rank}: unresolved motion history.",
        )

        frames.append(joined)

        total_plus5 += plus5
        total_plus10 += plus10

        free_gib = shutil.disk_usage(".").free / 1024**3

        print(
            f"Prepared rank {rank:02d}/20, "
            f"+5={plus5}, +10={plus10}, "
            f"free={free_gib:.2f} GiB",
            flush=True,
        )

    data = pl.concat(frames)

    require(
        data["demo_filename"].n_unique() == 20,
        "Confirmation data does not contain exactly 20 demos.",
    )

    return data, total_plus5, total_plus10


def calculate_metrics(
    y,
    probabilities,
    match_names,
    class_order,
):
    n = len(y)

    require(
        n > 0
        and probabilities.shape == (n, N_CLASSES),
        "Probability matrix has invalid shape.",
    )

    require(
        np.isfinite(probabilities).all(),
        "Nonfinite class probabilities.",
    )

    require(
        (probabilities >= 0).all()
        and (probabilities <= 1).all(),
        "Class probability outside [0,1].",
    )

    require(
        np.allclose(
            probabilities.sum(axis=1),
            1.0,
            atol=1e-6,
            rtol=0,
        ),
        "Class probabilities do not sum to one.",
    )

    hard_predictions = np.argmax(
        probabilities,
        axis=1,
    )

    true_probabilities = probabilities[
        np.arange(n),
        y,
    ]

    require(
        (true_probabilities > 0).all(),
        "Zero probability assigned to a true target.",
    )

    row_log_loss = -np.log(true_probabilities)

    row_brier = (
        (probabilities**2).sum(axis=1)
        - 2 * true_probabilities
        + 1
    )

    by_match = {
        name: float(
            row_log_loss[match_names == name].mean()
        )
        for name in np.unique(match_names)
    }

    per_zone = []

    for index, zone in enumerate(class_order):
        support = int((y == index).sum())

        recall = (
            float(
                ((hard_predictions == index) & (y == index)).sum()
                / support
            )
            if support
            else None
        )

        per_zone.append({
            "zone": zone,
            "support": support,
            "recall": recall,
        })

    summary = {
        "rows": n,
        "matches": len(by_match),
        "log_loss": float(row_log_loss.mean()),
        "equal_match_mean_log_loss": float(
            np.mean(list(by_match.values()))
        ),
        "multiclass_brier_score": float(row_brier.mean()),
        "accuracy": float(
            np.mean(hard_predictions == y)
        ),
        "macro_f1": float(
            f1_score(
                y,
                hard_predictions,
                labels=np.arange(N_CLASSES),
                average="macro",
                zero_division=0,
            )
        ),
        "per_zone": per_zone,
    }

    return summary, row_log_loss, hard_predictions


def score():
    """
    Execute the single formal confirmation scoring pass.

    The evaluator source must already be committed in Git.
    """

    script_path = Path(__file__).resolve()

    relative_path = str(
        script_path.relative_to(Path.cwd().resolve())
    )

    require(
        relative_path == "scripts/v3_confirm_evaluator.py",
        "Evaluator must be in the project's scripts directory.",
    )

    committed = subprocess.run(
        [
            "git",
            "show",
            f"HEAD:{relative_path}",
        ],
        capture_output=True,
    )

    require(
        committed.returncode == 0,
        "Evaluator source is not committed in Git.",
    )

    require(
        hashlib.sha256(committed.stdout).hexdigest()
        == sha256(script_path),
        "Evaluator source differs from committed pre-score version.",
    )

    rows, protocol, freeze = verify_sources()

    for output in (MOTION_OUT, PRED_OUT, RESULTS):
        require(
            not output.exists(),
            f"Confirmation output already exists: {output}",
        )

    storage_guard()

    environment, class_order, place_to_zone = (
        load_frozen_bomb_helpers()
    )

    require(
        class_order == freeze["class_order"],
        "Frozen model class order mismatch.",
    )

    print(
        "Building causal confirmation motion; "
        "no inference until all 20 matches are validated.",
        flush=True,
    )

    data, total_plus5, total_plus10 = (
        assemble_confirmation_data(
            rows,
            environment,
        )
    )

    frozen_counts = {
        int(horizon): int(count)
        for horizon, count in load_json(IDENTITY)[
            "valid_rows_by_horizon"
        ].items()
    }

    require(
        {
            5: total_plus5,
            10: total_plus10,
        } == frozen_counts,
        "Confirmation horizon counts disagree with manifest identity.",
    )

    storage_guard()

    spatial = pl.read_parquet(SPATIAL)

    require(
        spatial.height == freeze["b1"]["spatial_training_rows"],
        "Frozen B1 spatial cache row count changed.",
    )

    training_xyz = np.column_stack([
        spatial[axis].to_numpy()
        for axis in "XYZ"
    ]).astype(np.float64)

    training_places = np.asarray(
        spatial["place"].to_list(),
        dtype=object,
    )

    tree = cKDTree(training_xyz)

    b0 = load_json(
        freeze["artifacts"]["b0"]["path"]
    )["horizons"]

    b1 = load_json(
        freeze["artifacts"]["b1"]["path"]
    )["horizons"]

    b2 = load_json(
        freeze["artifacts"]["b2"]["path"]
    )["horizons"]

    zone_to_index = {
        zone: index
        for index, zone in enumerate(class_order)
    }

    results_by_horizon = []
    scoring = {}

    for horizon in (5, 10):
        frame = data.filter(
            pl.col("horizon_sec") == horizon
        )

        true_index = frame[
            "target_class_index"
        ].to_numpy().astype(np.int64)

        current_index = frame[
            "current_class_index"
        ].to_numpy().astype(np.int64)

        n = len(true_index)

        require(
            np.all((true_index >= 0) & (true_index < N_CLASSES)),
            "Invalid frozen target class index.",
        )

        require(
            np.all((current_index >= 0) & (current_index < N_CLASSES)),
            "Invalid frozen current class index.",
        )

        match_names = np.asarray(
            frame["demo_filename"].to_list()
        )

        probabilities = {}

        # --------------------------------------------------
        # B0: frozen persistence adapter
        # --------------------------------------------------

        q0 = float(b0[str(horizon)]["q"])

        p0 = np.full(
            (n, N_CLASSES),
            float(b0[str(horizon)]["other_probability"]),
        )

        p0[np.arange(n), current_index] = q0

        probabilities["B0"] = p0

        # --------------------------------------------------
        # Shared causal motion
        # --------------------------------------------------

        xyz = np.column_stack([
            frame[f"current_bomb_{axis}"].to_numpy()
            for axis in "XYZ"
        ]).astype(np.float64)

        velocity = np.column_stack([
            frame[f"velocity_{axis}"].to_numpy()
            for axis in "XYZ"
        ]).astype(np.float64)

        # --------------------------------------------------
        # B1: frozen constant-motion model
        # --------------------------------------------------

        projected = xyz + horizon * velocity

        require(
            np.isfinite(projected).all(),
            "Nonfinite B1 projected position.",
        )

        _, nearest = tree.query(
            projected,
            k=1,
            workers=-1,
        )

        predicted_b1 = np.asarray([
            zone_to_index[
                place_to_zone[str(training_places[index])]
            ]
            for index in nearest
        ])

        p1 = np.full(
            (n, N_CLASSES),
            float(b1[str(horizon)]["other_probability"]),
        )

        p1[np.arange(n), predicted_b1] = float(
            b1[str(horizon)]["q"]
        )

        probabilities["B1"] = p1

        # --------------------------------------------------
        # B2: frozen Markov transition matrix
        # --------------------------------------------------

        matrix = np.asarray(
            b2[str(horizon)]["probabilities"],
            dtype=np.float64,
        )

        require(
            matrix.shape == (N_CLASSES, N_CLASSES),
            "Frozen B2 matrix shape changed.",
        )

        probabilities["B2"] = matrix[current_index]

        # --------------------------------------------------
        # B3: frozen 24-dimensional tabular model
        # --------------------------------------------------

        X = np.zeros(
            (n, 24),
            dtype=np.float32,
        )

        X[np.arange(n), current_index] = 1.0

        for row_index, source in enumerate(
            frame["current_source"].to_list()
        ):
            if source in {
                "CARRIED_INVENTORY",
                "CARRIED_PICKUP_FALLBACK",
            }:
                X[row_index, 15] = 1.0

            elif source == "DROPPED":
                X[row_index, 16] = 1.0

            else:
                raise RuntimeError(
                    f"Unexpected current bomb source: {source}"
                )

        X[:, 17:20] = xyz.astype(np.float32)

        X[:, 20:23] = velocity.astype(np.float32)

        X[:, 23] = np.linalg.norm(
            velocity.astype(np.float32).astype(np.float64),
            axis=1,
        ).astype(np.float32)

        require(
            np.isfinite(X).all(),
            "Nonfinite B3 feature values.",
        )

        feature_protocol = load_json(
            "docs/v3_b3_feature_protocol_v1.json"
        )

        require(
            freeze["b3"]["feature_order"]
            == feature_protocol["feature_order"],
            "Frozen B3 feature order mismatch.",
        )

        model = XGBClassifier()

        model.load_model(
            freeze["artifacts"][
                f"b3_plus{horizon}"
            ]["path"]
        )

        probabilities["B3"] = np.asarray(
            model.predict_proba(X),
            dtype=np.float64,
        )

        # --------------------------------------------------
        # Evaluate all frozen models on identical rows
        # --------------------------------------------------

        scoring[str(horizon)] = {}

        row_log_losses = {}

        for label, matrix_probabilities in probabilities.items():
            summary, row_ll, _ = calculate_metrics(
                true_index,
                matrix_probabilities,
                match_names,
                class_order,
            )

            scoring[str(horizon)][label] = summary
            row_log_losses[label] = row_ll

            print(
                f"+{horizon}s {label}: "
                f"Log Loss={summary['log_loss']:.6f}, "
                f"n={n}",
                flush=True,
            )

        # --------------------------------------------------
        # Paired match bootstrap: B3 minus B2
        # --------------------------------------------------

        differences = (
            row_log_losses["B3"]
            - row_log_losses["B2"]
        )

        match_totals = np.asarray([
            differences[
                match_names == row["demo_filename"]
            ].sum()
            for row in rows
        ], dtype=np.float64)

        match_counts = np.asarray([
            (
                match_names == row["demo_filename"]
            ).sum()
            for row in rows
        ], dtype=np.int64)

        require(
            (match_counts > 0).all(),
            "A confirmation match has zero rows at a horizon.",
        )

        rng = np.random.default_rng(42)

        indices = rng.integers(
            0,
            20,
            size=(10000, 20),
        )

        bootstrap = (
            match_totals[indices].sum(axis=1)
            / match_counts[indices].sum(axis=1)
        )

        scoring[str(horizon)]["B3_minus_B2"] = {
            "pooled_log_loss_delta": float(
                differences.mean()
            ),
            "equal_match_log_loss_delta": float(
                np.mean(match_totals / match_counts)
            ),
            "match_bootstrap_95ci": [
                float(value)
                for value in np.quantile(
                    bootstrap,
                    [0.025, 0.975],
                )
            ],
            "bootstrap_resamples": 10000,
            "seed": 42,
        }

        output = {
            "demo_filename": frame[
                "demo_filename"
            ].to_list(),
            "round_num": frame[
                "round_num"
            ].to_list(),
            "current_nominal_tick": frame[
                "current_nominal_tick"
            ].to_list(),
            "horizon_sec": frame[
                "horizon_sec"
            ].to_list(),
            "target_class_index": true_index.tolist(),
        }

        for label, p in probabilities.items():
            for class_index in range(N_CLASSES):
                output[
                    f"{label}_p_{class_index:02d}"
                ] = p[:, class_index].tolist()

            output[
                f"{label}_row_log_loss"
            ] = row_log_losses[label].tolist()

        results_by_horizon.append(
            pl.DataFrame(output)
        )

    # ------------------------------------------------------
    # Creation-only outputs
    # ------------------------------------------------------

    for path in (MOTION_OUT, PRED_OUT, RESULTS):
        require(
            not path.exists(),
            f"Scoring output appeared during processing: {path}",
        )

    MOTION_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pl.concat(results_by_horizon).write_parquet(
        PRED_OUT,
        compression="zstd",
    )

    data.select([
        "demo_filename",
        "round_num",
        "current_nominal_tick",
        "current_tick",
        "prior_tick",
        "prior_source",
        "elapsed_ticks",
        "elapsed_sec",
        "velocity_X",
        "velocity_Y",
        "velocity_Z",
        "status",
    ]).unique().write_parquet(
        MOTION_OUT,
        compression="zstd",
    )

    final_results = {
        "version": "V3_CONFIRM_RESULTS_V1",
        "status": "FORMAL_CONFIRMATION_SCORING_COMPLETE",
        "manifest_sha256": sha256(MANIFEST),
        "scoring_protocol_sha256": sha256(PROTOCOL),
        "model_freeze_sha256": sha256(FREEZE),
        "evaluator_source_sha256": sha256(Path(__file__)),
        "frozen_target_builder_sha256": sha256(TARGET_BUILDER),
        "matches": 20,
        "eligible_rows_by_horizon": {
            "5": total_plus5,
            "10": total_plus10,
        },
        "results_by_horizon": scoring,
        "interpretation_note": (
            "Independent 20-match replication. "
            "No confirmation-based tuning, model regeneration "
            "or data substitution."
        ),
    }

    final_results["predictions_sha256"] = sha256(
        PRED_OUT
    )

    final_results["confirmation_motion_sha256"] = sha256(
        MOTION_OUT
    )

    with RESULTS.open("x") as file:
        json.dump(final_results, file, indent=2)
        file.write("\n")

    print(
        "\nFORMAL_CONFIRMATION_SCORING_COMPLETE:",
        RESULTS,
    )

    print(
        f"Free disk: "
        f"{shutil.disk_usage('.').free / 1024**3:.2f} GiB"
    )

    print(
        "NEXT: Generate closure from immutable results. "
        "Do not run formal scoring a second time."
    )


def close():
    """Create final report from stored results without model inference."""

    require(
        RESULTS.is_file()
        and PRED_OUT.is_file()
        and MOTION_OUT.is_file(),
        "Formal scoring outputs are missing.",
    )

    require(
        not CLOSURE.exists(),
        "V3 closure report already exists.",
    )

    result = load_json(RESULTS)

    require(
        result["status"] == "FORMAL_CONFIRMATION_SCORING_COMPLETE",
        "Formal scoring has not completed.",
    )

    require(
        result["manifest_sha256"] == sha256(MANIFEST)
        and result["scoring_protocol_sha256"] == sha256(PROTOCOL)
        and result["model_freeze_sha256"] == sha256(FREEZE),
        "Frozen research identity changed since scoring.",
    )

    require(
        result["predictions_sha256"] == sha256(PRED_OUT)
        and result["confirmation_motion_sha256"] == sha256(MOTION_OUT),
        "Scoring outputs changed since creation.",
    )

    lines = [
        "# V3 independent confirmation — closure",
        "",
        "**Status:** Formal independent confirmation completed; "
        "V3 research version closed.",
        "",
        "## Fixed experiment",
        "",
        "- First 20 technically eligible Mirage matches in the frozen queue.",
        "- Development: 53 separate matches. Confirmation: 20 later-period matches.",
        "- Frozen B0/B1/B2/B3 and 15-zone target; +5s and +10s separately.",
        "- Primary comparison: B3 minus B2 pooled multiclass Log Loss.",
        "- 10,000 paired match-bootstrap replicates per horizon, seed 42.",
        "",
        "## Formal confirmation measurements",
        "",
        "| Horizon | Rows | B0 LL | B1 LL | B2 LL | B3 LL | B3 − B2 | 95% match-bootstrap CI |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]

    for horizon in ("5", "10"):
        values = result["results_by_horizon"][horizon]
        difference = values["B3_minus_B2"]

        lower, upper = difference["match_bootstrap_95ci"]

        lines.append(
            f"| +{horizon}s "
            f"| {values['B3']['rows']} "
            f"| {values['B0']['log_loss']:.6f} "
            f"| {values['B1']['log_loss']:.6f} "
            f"| {values['B2']['log_loss']:.6f} "
            f"| {values['B3']['log_loss']:.6f} "
            f"| {difference['pooled_log_loss_delta']:+.6f} "
            f"| [{lower:+.6f}, {upper:+.6f}] |"
        )

    lines.extend([
        "",
        "## Interpretation",
        "",
    ])

    for horizon in ("5", "10"):
        difference = result[
            "results_by_horizon"
        ][horizon]["B3_minus_B2"]

        lower, upper = difference["match_bootstrap_95ci"]

        if upper < 0:
            statement = (
                "B3 had lower observed Log Loss, "
                "with the percentile interval below zero."
            )

        elif lower > 0:
            statement = (
                "B3 had higher observed Log Loss, "
                "with the percentile interval above zero."
            )

        else:
            statement = (
                "The percentile interval includes zero; "
                "the confirmation sample does not clearly "
                "separate the two Log Loss values."
            )

        lines.append(
            f"- +{horizon}s: {statement}"
        )

    lines.extend([
        "",
        "## Boundaries and limitations",
        "",
        "- Repeated observations within a match are correlated; "
        "uncertainty resamples matches, not individual rows.",
        "- Findings are limited to the frozen Mirage task and "
        "this later-period 20-match sample.",
        "- No confirmation-based tuning, refitting or candidate replacement.",
        "- Methodological changes belong to a new research version.",
        "",
        "## Immutable provenance",
        "",
        f"- Manifest SHA256: `{result['manifest_sha256']}`",
        f"- Model freeze SHA256: `{result['model_freeze_sha256']}`",
        f"- Scoring protocol SHA256: `{result['scoring_protocol_sha256']}`",
        f"- Evaluator SHA256: `{result['evaluator_source_sha256']}`",
        f"- Predictions SHA256: `{result['predictions_sha256']}`",
        f"- Causal motion SHA256: `{result['confirmation_motion_sha256']}`",
        "",
        "Detailed secondary metrics and per-zone support/recall are "
        "stored in `docs/v3_confirm_results_v1.json`.",
        "",
    ])

    with CLOSURE.open("x") as file:
        file.write("\n".join(lines))

    print(
        "V3_CLOSURE_REPORT_COMPLETE:",
        CLOSURE,
    )

    print(
        "The frozen V3 confirmation experiment is finished. "
        "Do not rescore or retune V3."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        required=True,
        choices=[
            "dev-test",
            "score",
            "close",
        ],
    )

    args = parser.parse_args()

    if args.mode == "dev-test":
        dev_test()

    elif args.mode == "score":
        score()

    elif args.mode == "close":
        close()
