from __future__ import annotations

import argparse
import csv
import hashlib
import re
import shutil
import subprocess
import time
import zipfile

from pathlib import Path

from curl_cffi import requests

from cs2_tactical_intelligence.v0.timing import (
    open_v0_demo,
)


# ============================================================
# Paths
# ============================================================

QUEUE = Path(
    "docs/v3_confirm_acquisition_queue_v1.csv"
)

DEST = Path(
    "data/raw/v3_confirm_incoming"
)

STAGING = Path(
    "data/raw/v3_confirm_download_staging"
)

RAW_ROOT = Path(
    "data/raw"
)


# Construct instead of hard-coding full external URLs.
SCHEME = "https"
HOST = "www.hltv.org"

BASE = (
    SCHEME
    + "://"
    + HOST
)


DEST.mkdir(
    parents=True,
    exist_ok=True,
)

STAGING.mkdir(
    parents=True,
    exist_ok=True,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def existing_demo_hashes():
    result = {}

    for path in RAW_ROOT.rglob("*.dem"):

        if STAGING in path.parents:
            continue

        try:
            digest = sha256_file(path)
        except Exception:
            continue

        result.setdefault(
            digest,
            [],
        ).append(
            str(path)
        )

    return result


def find_demo_download_path(html: str):
    matches = re.findall(
        r'''href=["'](/download/demo/[^"'?#]+)["']''',
        html,
        flags=re.IGNORECASE,
    )

    unique = []

    for value in matches:
        if value not in unique:
            unique.append(value)

    if not unique:
        return None

    return unique[0]


def archive_extension(path: Path):
    head = path.read_bytes()[:8]

    if head.startswith(
        b"PK\x03\x04"
    ):
        return ".zip"

    if head.startswith(
        b"Rar!\x1a\x07"
    ):
        return ".rar"

    if head.startswith(
        b"7z\xbc\xaf\x27\x1c"
    ):
        return ".7z"

    return ".archive"


def extract_archive(
    archive: Path,
    output_dir: Path,
):
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if zipfile.is_zipfile(
        archive
    ):

        with zipfile.ZipFile(
            archive
        ) as zf:

            zf.extractall(
                output_dir
            )

        return


    if shutil.which(
        "unar"
    ):

        subprocess.run(
            [
                "unar",
                "-q",
                "-o",
                str(
                    output_dir
                ),
                str(
                    archive
                ),
            ],
            check=True,
        )

        return


    if shutil.which(
        "7z"
    ):

        subprocess.run(
            [
                "7z",
                "x",
                "-y",
                f"-o{output_dir}",
                str(
                    archive
                ),
            ],
            check=True,
        )

        return


    raise RuntimeError(
        "RAR/7z extractor unavailable. "
        "Install 'unar' and run again."
    )


def locate_mirage_demo(
    extracted_dir: Path,
):
    demos = sorted(
        extracted_dir.rglob(
            "*.dem"
        )
    )

    if not demos:
        raise RuntimeError(
            "Archive contained no .dem files."
        )

    mirage = []

    print(
        "    demos in archive:",
        len(demos),
    )

    for demo_path in demos:

        try:

            demo = open_v0_demo(
                demo_path,
                verbose=False,
            )

            map_name = (
                demo.header.get(
                    "map_name"
                )
            )

            print(
                "      ",
                demo_path.name,
                "->",
                map_name,
            )


            if map_name == "de_mirage":

                mirage.append(
                    demo_path
                )

        except Exception as exc:

            print(
                "      ⚠️ could not inspect",
                demo_path.name,
                ":",
                exc,
            )


    if len(mirage) != 1:

        raise RuntimeError(
            "Expected exactly one Mirage demo, "
            f"found {len(mirage)}."
        )


    return mirage[0]


def download_archive(
    session,
    demo_path: str,
    match_id: str,
):
    demo_url = (
        BASE
        + demo_path
    )

    temp_part = (
        STAGING
        / f"{match_id}.part"
    )

    for path in STAGING.glob(
        f"{match_id}.*"
    ):
        if path.is_file():
            path.unlink()

    print(
        "    downloading archive..."
    )


    response = session.get(
        demo_url,
        stream=True,
        allow_redirects=True,
        timeout=300,
    )

    response.raise_for_status()


    downloaded = 0

    try:

        with temp_part.open(
            "wb"
        ) as f:

            for chunk in (
                response.iter_content(
                    chunk_size=
                        1024 * 1024
                )
            ):

                if not chunk:
                    continue

                f.write(
                    chunk
                )

                downloaded += (
                    len(
                        chunk
                    )
                )

    finally:

        response.close()


    extension = (
        archive_extension(
            temp_part
        )
    )

    archive = (
        STAGING
        / (
            str(
                match_id
            )
            + extension
        )
    )

    temp_part.replace(
        archive
    )


    print(
        "    downloaded:",
        f"{downloaded / 1024 / 1024:.1f} MB",
        extension,
    )


    return archive


# ============================================================
# Queue
# ============================================================

with QUEUE.open(
    newline="",
) as f:

    queue = list(
        csv.DictReader(
            f
        )
    )


if len(queue) != 25:

    raise RuntimeError(
        "Expected exactly 25 frozen confirmation "
        f"candidates, found {len(queue)}."
    )


ranks = [
    int(row["candidate_rank"])
    for row in queue
]

if ranks != list(range(1, 26)):

    raise RuntimeError(
        "Confirmation candidate ranks must be "
        "exactly 1..25 in frozen order."
    )


parser = argparse.ArgumentParser(
    description=(
        "Acquire exactly one frozen V3 confirmation "
        "candidate by candidate_rank."
    )
)

parser.add_argument(
    "--candidate-rank",
    type=int,
    required=True,
    help="Frozen confirmation candidate rank 1..25.",
)

args = parser.parse_args()


if not 1 <= args.candidate_rank <= 25:

    raise RuntimeError(
        "--candidate-rank must be between 1 and 25."
    )


selected = [
    row
    for row in queue
    if int(row["candidate_rank"])
    == args.candidate_rank
]


if len(selected) != 1:

    raise RuntimeError(
        "Candidate rank did not resolve to exactly "
        "one frozen queue row."
    )


queue = selected


# ============================================================
# Session
# ============================================================

session = requests.Session(
    impersonate="chrome"
)


# ============================================================
# Run
# ============================================================

success = []
failed = []
already_present = []


print("=" * 100)
print("V3 CONFIRM DEMO ACQUISITION")
print("=" * 100)

print(
    "Queue:",
    len(queue),
)

print(
    "Destination:",
    DEST,
)

print(
    "Mode:",
    "sequential / storage-conservative",
)


for index, row in enumerate(
    queue,
    start=1,
):

    match_id = str(
        row[
            "match_id"
        ]
    )

    date = row[
        "date"
    ]

    team1 = row[
        "team1"
    ]

    team2 = row[
        "team2"
    ]

    slug = row[
        "slug"
    ]


    dest_name = (
        f"{date}_"
        f"{team1}_vs_"
        f"{team2}_mirage.dem"
    )

    dest_path = (
        DEST
        / dest_name
    )


    print()
    print(
        f"[rank {int(row['candidate_rank']):02d}/25]",
        dest_name,
    )


    # --------------------------------------------------------
    # Resume-safe
    # --------------------------------------------------------

    if dest_path.exists():

        try:

            demo = open_v0_demo(
                dest_path,
                verbose=False,
            )

            if (
                demo.header.get(
                    "map_name"
                )
                == "de_mirage"
            ):

                print(
                    "    ✅ already present"
                )

                already_present.append(
                    dest_name
                )

                continue

        except Exception:
            pass


        raise RuntimeError(
            "Destination already exists but "
            "did not verify as Mirage: "
            f"{dest_path}"
        )


    extract_dir = (
        STAGING
        / (
            match_id
            + "_extract"
        )
    )


    if extract_dir.exists():

        shutil.rmtree(
            extract_dir
        )


    try:

        # ----------------------------------------------------
        # Match page
        # ----------------------------------------------------

        match_path = (
            "/matches/"
            + match_id
            + "/"
            + slug
        )

        match_url = (
            BASE
            + match_path
        )


        page = session.get(
            match_url,
            timeout=60,
        )

        page.raise_for_status()


        if (
            "Mirage"
            not in page.text
        ):

            raise RuntimeError(
                "Match page does not appear "
                "to contain Mirage."
            )


        demo_path = (
            find_demo_download_path(
                page.text
            )
        )


        if demo_path is None:

            raise RuntimeError(
                "No Demo download link found "
                "on match page."
            )


        print(
            "    ✅ demo link found"
        )


        # Be polite to HLTV.
        time.sleep(
            1.2
        )


        # ----------------------------------------------------
        # Download one archive
        # ----------------------------------------------------

        archive = download_archive(
            session,
            demo_path,
            match_id,
        )


        # ----------------------------------------------------
        # Extract
        # ----------------------------------------------------

        print(
            "    extracting..."
        )

        extract_archive(
            archive,
            extract_dir,
        )


        # ----------------------------------------------------
        # Identify Mirage by actual parsed map header
        # ----------------------------------------------------

        mirage_demo = (
            locate_mirage_demo(
                extract_dir
            )
        )


        source_sha = (
            sha256_file(
                mirage_demo
            )
        )


        # ----------------------------------------------------
        # Historical duplicate protection
        # ----------------------------------------------------

        known = (
            existing_demo_hashes()
        )


        if source_sha in known:

            raise RuntimeError(
                "SHA256 duplicate of existing demo: "
                + " | ".join(
                    known[
                        source_sha
                    ]
                )
            )


        # ----------------------------------------------------
        # Move only Mirage to final V3 location
        # ----------------------------------------------------

        shutil.move(
            str(
                mirage_demo
            ),
            str(
                dest_path
            ),
        )


        # Verify after move.
        final_demo = open_v0_demo(
            dest_path,
            verbose=False,
        )


        if (
            final_demo.header.get(
                "map_name"
            )
            != "de_mirage"
        ):

            raise RuntimeError(
                "Post-move map verification failed."
            )


        final_sha = (
            sha256_file(
                dest_path
            )
        )


        if final_sha != source_sha:

            raise RuntimeError(
                "SHA changed during move."
            )


        print(
            "    ✅",
            dest_name,
        )

        print(
            "    SHA256:",
            final_sha,
        )


        success.append(
            dest_name
        )


        # ----------------------------------------------------
        # STORAGE POLICY:
        #
        # After successful final verification:
        # delete archive + all other maps.
        # ----------------------------------------------------

        if archive.exists():

            archive.unlink()


        if extract_dir.exists():

            shutil.rmtree(
                extract_dir
            )


        time.sleep(
            1.0
        )


    except Exception as exc:

        failed.append({
            "match_id":
                match_id,

            "filename":
                dest_name,

            "error":
                str(
                    exc
                ),
        })


        print(
            "    ❌",
            exc,
        )


# ============================================================
# Final report
# ============================================================

final_demos = sorted(
    DEST.glob(
        "*.dem"
    )
)


print()
print("=" * 100)
print("V3 CONFIRM ACQUISITION SUMMARY")
print("=" * 100)

print(
    "Downloaded this run:",
    len(
        success
    ),
)

print(
    "Already present:",
    len(
        already_present
    ),
)

print(
    "Failed:",
    len(
        failed
    ),
)

print(
    "Total .dem now in v3_confirm_incoming:",
    len(
        final_demos
    ),
)


if failed:

    print()
    print(
        "FAILED MATCHES"
    )

    for item in failed:

        print(
            " ",
            item[
                "match_id"
            ],
            "|",
            item[
                "filename"
            ],
        )

        print(
            "    ",
            item[
                "error"
            ],
        )


print()
print(
    "Current V3 incoming files:"
)

for path in final_demos:

    print(
        " ",
        path.name,
    )


print()
print("=" * 100)


if len(
    final_demos
) >= 20:

    print(
        "✅ FRESH V3 TARGET REACHED: 20 DEMOS"
    )

    print(
        "NEXT_ACTION="
        "BUILD_D_V3_DEV_53_MATCH_MANIFEST"
    )

else:

    print(
        "⚠️ FRESH V3 TARGET NOT YET REACHED"
    )

    print(
        "Need:",
        20 - len(
            final_demos
        ),
        "more successful demos",
    )

    print(
        "Send the FAILED MATCHES section to ChatGPT."
    )


print("=" * 100)
