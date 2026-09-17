from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


FREEZE = Path(
    "docs/v3_confirm_model_freeze.json"
)

VERIFICATION = Path(
    "docs/v3_confirm_model_verification.json"
)

FITTER = Path(
    "scripts/fit_v3_confirm_models.py"
)

VERIFIER = Path(
    "scripts/verify_v3_confirm_model_freeze.py"
)

OUT = Path(
    "docs/v3_confirm_fit_provenance_repair.json"
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_json(path):
    return json.loads(
        Path(path).read_text()
    )


def sha256(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def git_output(args):
    return subprocess.check_output(
        ["git", *args],
        text=True,
    ).strip()


def committed_sha(commit, path):
    result = subprocess.run(
        [
            "git",
            "show",
            f"{commit}:{path}",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    if result.returncode != 0:
        return None

    return hashlib.sha256(
        result.stdout
    ).hexdigest()


for path in [
    FREEZE,
    VERIFICATION,
    FITTER,
    VERIFIER,
]:
    require(
        path.exists(),
        f"Missing required evidence: {path}",
    )


require(
    not OUT.exists(),
    "Provenance repair record already exists.",
)


freeze = read_json(FREEZE)
verification = read_json(
    VERIFICATION
)


require(
    verification["overall_status"]
    == "PASS",
    "Independent verification did not pass.",
)

require(
    freeze["confirmation_data_accessed"]
    is False,
    "Freeze record indicates confirmation access.",
)

require(
    freeze[
        "confirmation_performance_calculated"
    ]
    is False,
    "Freeze record indicates confirmation scoring.",
)

require(
    verification["confirmation_data_accessed"]
    is False,
    "Verifier indicates confirmation access.",
)

require(
    verification[
        "confirmation_performance_calculated"
    ]
    is False,
    "Verifier indicates confirmation scoring.",
)


fit_time_commit = freeze[
    "fit_git_commit"
]

fit_time_sha = freeze[
    "fit_script_sha256"
]

current_fitter_sha = sha256(
    FITTER
)


require(
    current_fitter_sha == fit_time_sha,
    (
        "Current fitter bytes differ from "
        "fit-time recorded SHA."
    ),
)


current_head = git_output([
    "rev-parse",
    "HEAD",
])


current_committed_sha = committed_sha(
    current_head,
    str(FITTER),
)


require(
    current_committed_sha
    == fit_time_sha,
    (
        "Currently committed fitter does not "
        "match fit-time SHA."
    ),
)


fit_time_committed_sha = committed_sha(
    fit_time_commit,
    str(FITTER),
)


require(
    fit_time_committed_sha is None,
    (
        "Expected fitter to be absent from the "
        "fit-time commit; provenance situation changed."
    ),
)


fitter_commit = git_output([
    "log",
    "-1",
    "--format=%H",
    "--",
    str(FITTER),
])


verifier_commit = git_output([
    "log",
    "-1",
    "--format=%H",
    "--",
    str(VERIFIER),
])


artifact_records = freeze[
    "artifacts"
]


for name, metadata in artifact_records.items():

    path = Path(
        metadata["path"]
    )

    require(
        path.exists(),
        f"Frozen artifact missing: {name}",
    )

    require(
        sha256(path)
        == metadata["sha256"],
        f"Frozen artifact SHA mismatch: {name}",
    )


v2_tracked = subprocess.run(
    [
        "git",
        "ls-files",
        "--error-unmatch",
        "artifacts/v2_confirm_frozen/"
        "v0_xgb_a5.json",
    ],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    check=False,
).returncode == 0


v3_tracking = {}

for name, metadata in artifact_records.items():

    path = metadata["path"]

    tracked = subprocess.run(
        [
            "git",
            "ls-files",
            "--error-unmatch",
            path,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    ).returncode == 0

    ignored = subprocess.run(
        [
            "git",
            "check-ignore",
            "-q",
            path,
        ],
        check=False,
    ).returncode == 0

    v3_tracking[name] = {
        "path":
            path,

        "sha256":
            metadata["sha256"],

        "git_tracked":
            tracked,

        "git_ignored":
            ignored,
    }

    require(
        tracked is False,
        f"Unexpected tracked V3 artifact: {path}",
    )

    require(
        ignored is True,
        f"Expected ignored V3 artifact: {path}",
    )


record = {
    "record":
        "V3 confirmation fitter provenance repair",

    "status":
        "RESOLVED_BEFORE_CONFIRMATION_ACCESS",

    "created_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "issue": {
        "description":
            (
                "The one-time full-development fit was "
                "executed while scripts/"
                "fit_v3_confirm_models.py existed in the "
                "working tree but had not yet been committed."
            ),

        "scientific_effect":
            (
                "No confirmation data were accessed and no "
                "confirmation performance was calculated. "
                "The fit-time freeze record captured the "
                "exact fitter SHA256."
            ),
    },

    "fit_time": {
        "git_commit":
            fit_time_commit,

        "fitter_path":
            str(FITTER),

        "fitter_sha256":
            fit_time_sha,

        "fitter_present_in_fit_time_commit":
            False,
    },

    "repair": {
        "action":
            (
                "Commit the exact byte-identical fitter "
                "source without regenerating model artifacts."
            ),

        "fitter_commit":
            fitter_commit,

        "committed_fitter_sha256":
            current_committed_sha,

        "byte_identity":
            "PASS",

        "models_refit":
            False,

        "model_artifacts_regenerated":
            False,
    },

    "independent_verification": {
        "verifier_commit":
            verifier_commit,

        "verification_record":
            str(VERIFICATION),

        "verification_record_sha256":
            sha256(VERIFICATION),

        "overall_status":
            verification["overall_status"],

        "artifact_sha256":
            "PASS",

        "B0_recomputation":
            "PASS",

        "B1_recomputation":
            "PASS",

        "B2_recomputation":
            "PASS",

        "B3_load_and_inference":
            "PASS",
    },

    "artifact_storage_policy": {
        "gitignore_rule":
            "artifacts/*",

        "v2_frozen_artifacts_git_tracked":
            v2_tracked,

        "v3_policy":
            (
                "Follow existing repository precedent: "
                "binary/model artifacts remain outside Git "
                "and are frozen by immutable path plus "
                "SHA256 metadata."
            ),

        "v3_artifacts":
            v3_tracking,
    },

    "confirmation_boundary": {
        "D_V3_CONFIRM_accessed":
            False,

        "confirmation_performance_calculated":
            False,

        "confirmation_outcome_used_for_repair":
            False,
    },

    "conclusion":
        (
            "The provenance gap is resolved without refitting. "
            "The committed fitter is byte-identical to the "
            "fit-time source recorded by SHA256, all frozen "
            "artifacts independently verify, and the repair "
            "occurred before any V3 confirmation access."
        ),
}


OUT.write_text(
    json.dumps(
        record,
        indent=2,
    )
    + "\n"
)


print("V3 CONFIRM FIT PROVENANCE REPAIR")
print()
print("fit-time fitter tracked: NO")
print("fit-time fitter SHA recorded: YES")
print("committed fitter byte-identical: PASS")
print("models refit: NO")
print("independent model verification: PASS")
print("V2 model artifacts tracked in Git:", v2_tracked)
print("V3 model artifacts tracked in Git: NO")
print("V3 model artifacts SHA-frozen: YES")
print("confirmation accessed: NO")
print("confirmation performance calculated: NO")
print()
print("V3_CONFIRM_FIT_PROVENANCE_RESOLVED")
