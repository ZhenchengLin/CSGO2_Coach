"""Publish the frozen V3 research record to the existing static website."""

from __future__ import annotations

import csv
import hashlib
import html
import json
import re
import shutil
import subprocess
from pathlib import Path


ROOT = Path.cwd()
SITE = ROOT / "site"
PAGES = SITE / "pages"
SOURCES = SITE / "assets" / "sources"

OUTPUT = PAGES / "v3.html"

RESULTS = ROOT / "docs/v3_confirm_results_v1.json"
IDENTITY = ROOT / "docs/v3_confirm_manifest_identity.json"
MANIFEST = ROOT / "docs/v3_confirm_manifest.csv"
SCORING = ROOT / "docs/v3_confirm_scoring_protocol_v1.json"
FREEZE = ROOT / "docs/v3_confirm_model_freeze.json"
CLOSURE = ROOT / "docs/v3_closure_v1.md"
DESIGN = ROOT / "docs/v3_research_design.md"
DEVELOPMENT = ROOT / "docs/v3_development_closure_v1.json"
FEATURES = ROOT / "docs/v3_b3_feature_protocol_v1.json"
MAPPING = ROOT / "docs/v3_macro_zone_mapping_v1_frozen.json"

SOURCE_NAMES = [
    "v3_confirm_results_v1.json",
    "v3_closure_v1.md",
    "v3_research_design.md",
    "v3_development_closure_v1.json",
    "v3_confirm_scoring_protocol_v1.json",
    "v3_confirm_manifest_identity.json",
    "v3_confirm_model_freeze.json",
    "v3_b3_feature_protocol_v1.json",
    "v3_macro_zone_mapping_v1_frozen.json",
]


def check(condition, message):
    if not condition:
        raise RuntimeError("STOP: " + message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def esc(value):
    return html.escape(str(value), quote=True)


def fmt(number):
    return f"{number:.6f}"


def source_link(name, label):
    return (
        f'<a href="../assets/sources/{esc(name)}">'
        f"{esc(label)} ↗</a>"
    )


# ============================================================
# 1. Preflight: never overwrite an existing V3 site release
# ============================================================

check(
    (SITE / "index.html").is_file(),
    "Run this script from the CSGO2 repository root.",
)

check(
    not OUTPUT.exists(),
    "The V3 website page already exists. Do not overwrite it.",
)

site_status = subprocess.run(
    ["git", "status", "--porcelain", "--", "site"],
    capture_output=True,
    text=True,
    check=True,
).stdout.strip()

check(
    not site_status,
    "The website has uncommitted changes. Review them first.",
)

required = [
    RESULTS,
    IDENTITY,
    MANIFEST,
    SCORING,
    FREEZE,
    CLOSURE,
    DESIGN,
    DEVELOPMENT,
    FEATURES,
    MAPPING,
    PAGES / "v2.html",
    SITE / "site-manifest.json",
]

for path in required:
    check(path.is_file(), f"Missing required source: {path}")

for name in SOURCE_NAMES:
    check(
        not (SOURCES / name).exists(),
        f"Published V3 evidence already exists: {name}",
    )


# ============================================================
# 2. Read and verify the frozen V3 research record
# ============================================================

results = json.loads(RESULTS.read_text())
identity = json.loads(IDENTITY.read_text())
protocol = json.loads(SCORING.read_text())
freeze = json.loads(FREEZE.read_text())
dev = json.loads(DEVELOPMENT.read_text())
features = json.loads(FEATURES.read_text())
mapping = json.loads(MAPPING.read_text())

check(
    results["status"] == "FORMAL_CONFIRMATION_SCORING_COMPLETE",
    "V3 formal confirmation results are not complete.",
)

check(
    dev["status"] == "DEVELOPMENT_CLOSED",
    "V3 development is not closed.",
)

check(
    results["matches"] == 20
    and dev["development_corpus"]["matches"] == 53,
    "Frozen match counts changed.",
)

for key, path in [
    ("manifest_sha256", MANIFEST),
    ("scoring_protocol_sha256", SCORING),
    ("model_freeze_sha256", FREEZE),
]:
    check(
        results[key] == digest(path),
        f"V3 results reference a different {path.name}.",
    )

check(
    identity["manifest_sha256"] == digest(MANIFEST),
    "Confirmation manifest identity mismatch.",
)

check(
    protocol["manifest_sha256"] == digest(MANIFEST),
    "Scoring protocol manifest mismatch.",
)

check(
    protocol["model_freeze_sha256"] == digest(FREEZE),
    "Scoring protocol model-freeze mismatch.",
)

check(
    protocol["candidate"] == "B3_TABULAR_MAP_AWARE_V1",
    "Unexpected frozen V3 candidate.",
)

check(
    len(mapping["zones"]) == 15
    and features["total_dimensions"] == 24,
    "Frozen V3 feature or macro-zone definitions changed.",
)

with MANIFEST.open(newline="") as handle:
    matches = list(csv.DictReader(handle))

check(
    len(matches) == 20
    and [
        int(match["candidate_rank"])
        for match in matches
    ] == list(range(1, 21)),
    "Confirmation manifest ranks changed.",
)

check(
    set(results["results_by_horizon"]) == {"5", "10"},
    "Unexpected V3 prediction horizons.",
)

for horizon in ("5", "10"):
    record = results["results_by_horizon"][horizon]

    check(
        all(
            model in record
            for model in ("B0", "B1", "B2", "B3", "B3_minus_B2")
        ),
        f"Missing model result at +{horizon}s.",
    )

    check(
        record["B3"]["rows"]
        == results["eligible_rows_by_horizon"][horizon],
        f"Confirmation row count mismatch at +{horizon}s.",
    )

    delta = record["B3_minus_B2"]

    check(
        abs(
            delta["pooled_log_loss_delta"]
            - (
                record["B3"]["log_loss"]
                - record["B2"]["log_loss"]
            )
        ) < 1e-10,
        f"Paired model comparison mismatch at +{horizon}s.",
    )


# ============================================================
# 3. Update the existing website navigation
# ============================================================

plans = {}

nav_re = re.compile(
    r'(<nav class="nav">)(.*?)(</nav>)',
    re.S,
)

link_re = re.compile(
    r"<a\b[^>]*>.*?</a>",
    re.S,
)


def update_nav(document, *, home=False, page="home"):

    def replace(match):
        links = link_re.findall(match.group(2))

        check(
            len(links) == 12,
            "Expected the original 12 navigation entries.",
        )

        check(
            sum(
                'data-page="v2"' in link
                for link in links
            ) == 1,
            "Existing V2 navigation entry is missing.",
        )

        href = (
            "pages/v3.html"
            if home
            else "../pages/v3.html"
        )

        new_link = (
            f'<a data-page="v3" href="{href}">'
            "<span>04</span>V3 forecasting</a>"
        )

        index = next(
            index
            for index, link in enumerate(links)
            if 'data-page="v2"' in link
        )

        links.insert(index + 1, new_link)

        for index, link in enumerate(links, 1):

            link = re.sub(
                r'\sclass="active"|\saria-current="page"',
                "",
                link,
            )

            link, changed = re.subn(
                r"<span>\d{2}</span>",
                f"<span>{index:02d}</span>",
                link,
                count=1,
            )

            check(
                changed == 1,
                "Could not renumber an existing navigation entry.",
            )

            if f'data-page="{page}"' in link:
                link = link.replace(
                    "<a ",
                    '<a class="active" aria-current="page" ',
                    1,
                )

            links[index - 1] = link

        return (
            match.group(1)
            + "".join(links)
            + match.group(3)
        )

    updated, count = nav_re.subn(
        replace,
        document,
        count=1,
    )

    check(
        count == 1,
        "Website navigation block was not found.",
    )

    return updated


def update_shell(document, *, home=False, page="home"):

    updated = update_nav(
        document,
        home=home,
        page=page,
    )

    updated, count = re.subn(
        r'<div class="status">.*?</div>',
        '<div class="status">'
        "V3 · Independent confirmation complete"
        "</div>",
        updated,
        count=1,
        flags=re.S,
    )

    check(
        count == 1,
        "Sidebar research-status block is missing.",
    )

    # Preserve the version-specific heading on historical V0–V2 pages.
    if page not in {"v0", "v1", "v2"}:

        updated, count = re.subn(
            r'(<span class="top-meta">'
            r'MIRAGE <span>·</span> )[^<]*(</span>)',
            r"\g<1>V3 RESEARCH CLOSED\2",
            updated,
            count=1,
        )

        check(
            count == 1,
            "Top website status label format changed.",
        )

    return updated


# ============================================================
# 4. Add context-specific V3 links to the existing pages
# ============================================================

def release_card(title, body, *, home=False):

    href = (
        "pages/v3.html"
        if home
        else "v3.html"
    )

    return (
        '\n<section id="v3-release" '
        'class="section" aria-label="V3 research release">'
        '<div class="section-head">'
        '<div class="kicker">'
        "V3 · September 2026 · frozen independent confirmation"
        "</div>"
        f"<h2>{esc(title)}</h2>"
        f"<p>{esc(body)}</p>"
        "</div>"
        '<div class="badges">'
        '<span class="badge ok">53 development matches</span>'
        '<span class="badge ok">'
        "20 untouched confirmation matches"
        "</span>"
        '<span class="badge">'
        "+5s / +10s · 15 macro-zones"
        "</span>"
        "</div>"
        f'<p><a class="button" href="{href}">'
        "Read the complete V3 research record ↗"
        "</a></p>"
        "</section>\n"
    )


summaries = {
    "home": (
        "V3: a separate, completed spatial forecasting task",
        "Beyond the V0/V2 plant-outcome task, V3 forecasts "
        "the future bomb macro-zone in Mirage. Read the frozen "
        "design, independent results, uncertainty and limitations.",
    ),
    "v0": (
        "V3 is a new task, not a replacement for V0",
        "V0 predicts round-level plant outcome; V3 forecasts "
        "bomb macro-zone at +5s and +10s. Their metrics should "
        "not be compared as if they were the same target.",
    ),
    "v1": (
        "After the V1 hierarchy investigation",
        "V3 addresses a distinct spatial and temporal target "
        "with separate development and confirmation evidence.",
    ),
    "v2": (
        "V3 follows V2 without reusing its confirmation set",
        "V2 retains its frozen plant-outcome conclusion. "
        "V3 uses a different target and a new 20-match "
        "confirmation corpus.",
    ),
    "evidence": (
        "V3 independent confirmation: B3 versus B2",
        "See the complete four-model results, paired "
        "match-bootstrap intervals, per-zone support and "
        "evidence boundaries on the V3 research page.",
    ),
    "models": (
        "V3 model ladder: B0 → B1 → B2 → B3",
        "Persistence, constant motion, Markov transitions "
        "and a 24-feature map-aware XGBoost model were compared "
        "on the same frozen confirmation observations.",
    ),
    "data": (
        "V3: 53 development / 20 confirmation matches",
        "The new task uses 15 frozen Mirage macro-zones, "
        "+5s and +10s horizons, and an independently frozen "
        "20-match confirmation manifest.",
    ),
    "roadmap": (
        "V3 closed; V4 remains a research question",
        "V3 development, model freeze, independent scoring "
        "and research closure are complete. The V4 scope "
        "and evaluation protocol have not yet been frozen.",
    ),
    "decisions": (
        "V3: separate tasks and preserve frozen evidence",
        "V2 D_CONFIRM was not a V3 tuning set. The V3 candidate, "
        "model artifacts, match manifest and scoring protocol "
        "were fixed before formal confirmation.",
    ),
    "research": (
        "What V3 establishes, and what remains",
        "The completed Mirage bomb-zone experiment motivates "
        "further error analysis and tactical questions. "
        "No V4 improvement is claimed.",
    ),
    "build-log": (
        "V3 implementation and verification checkpoint",
        "Follow the data-semantic audits, 53-match development "
        "freeze, 20-match independent intake, model verification "
        "and one formal scoring pass.",
    ),
    "journey": (
        "From round outcome to spatial forecasting",
        "The research progressed from the V0/V2 plant-outcome "
        "problem to V3 bomb-zone forecasting while keeping "
        "independent evaluation boundaries explicit.",
    ),
}


existing_pages = [
    SITE / "index.html",
    *sorted(PAGES.glob("*.html")),
]

check(
    len(existing_pages) == 12,
    "Expected the homepage and 11 existing research pages.",
)

for path in existing_pages:

    page = (
        "home"
        if path == SITE / "index.html"
        else path.stem
    )

    check(
        page in summaries,
        f"Unexpected existing website page: {page}",
    )

    current = path.read_text()

    check(
        'id="v3-release"' not in current,
        f"V3 release section already exists in {path}.",
    )

    current = update_shell(
        current,
        home=(page == "home"),
        page=page,
    )

    check(
        "</header>" in current
        and '<div class="content">' in current,
        f"Existing page hero not found in {path}.",
    )

    title, body = summaries[page]

    current = current.replace(
        "</header>",
        "</header>" + release_card(
            title,
            body,
            home=(page == "home"),
        ),
        1,
    )

    plans[path] = current


# ============================================================
# 5. Build a complete V3 page using the existing site shell
# ============================================================

v2_html = (PAGES / "v2.html").read_text()

check(
    v2_html.count('<div class="content">') == 1
    and v2_html.count('<footer class="footer">') == 1,
    "Existing V2 page shell changed.",
)

prefix = (
    v2_html.split('<div class="content">', 1)[0]
    + '<div class="content">'
)

footer_end = v2_html.rfind("</footer>")

check(
    footer_end >= 0,
    "Existing V2 footer is malformed.",
)

suffix = v2_html[
    footer_end + len("</footer>"):
]

prefix = prefix.replace(
    'data-page="v2"',
    'data-page="v3"',
    1,
)

prefix = re.sub(
    r"<title>.*?</title>",
    "<title>V3 independent confirmation"
    " — CS2 Tactical Intelligence Lab</title>",
    prefix,
    count=1,
)

prefix = prefix.replace(
    "The notebook <span>/</span> V2 confirmation",
    "The notebook <span>/</span> V3 independent confirmation",
)

prefix = update_shell(
    prefix,
    page="v3",
)


# ============================================================
# 6. Generate tables directly from the frozen results
# ============================================================

metric_rows = []

for horizon in ("5", "10"):

    horizon_results = results[
        "results_by_horizon"
    ][horizon]

    model_names = [
        ("B0", "Persistence"),
        ("B1", "Constant motion"),
        ("B2", "Zone Markov"),
        ("B3", "Tabular map-aware"),
    ]

    for model, label in model_names:

        metrics = horizon_results[model]

        metric_rows.append(
            "<tr>"
            f"<td>+{horizon}s</td>"
            f"<td>{model} · {esc(label)}</td>"
            f'<td>{fmt(metrics["log_loss"])}</td>'
            f'<td>{100 * metrics["accuracy"]:.2f}%</td>'
            f'<td>{fmt(metrics["macro_f1"])}</td>'
            f'<td>{fmt(metrics["multiclass_brier_score"])}</td>'
            f'<td>{fmt(metrics["equal_match_mean_log_loss"])}</td>'
            "</tr>"
        )


zones_5 = {
    zone["zone"]: zone
    for zone in results[
        "results_by_horizon"
    ]["5"]["B3"]["per_zone"]
}

zones_10 = {
    zone["zone"]: zone
    for zone in results[
        "results_by_horizon"
    ]["10"]["B3"]["per_zone"]
}


def recall_text(record):
    if record["recall"] is None:
        return "N/A"

    return f'{100 * record["recall"]:.1f}%'


zone_rows = []

for zone in mapping["zones"]:

    first = zones_5[zone]
    second = zones_10[zone]

    zone_rows.append(
        "<tr>"
        f"<td><code>{esc(zone)}</code></td>"
        f'<td>{first["support"]}</td>'
        f"<td>{recall_text(first)}</td>"
        f'<td>{second["support"]}</td>'
        f"<td>{recall_text(second)}</td>"
        "</tr>"
    )


comparison_rows = []

for horizon in ("5", "10"):

    horizon_results = results[
        "results_by_horizon"
    ][horizon]

    delta = horizon_results["B3_minus_B2"]

    lower, upper = delta[
        "match_bootstrap_95ci"
    ]

    reduction = (
        -100
        * delta["pooled_log_loss_delta"]
        / horizon_results["B2"]["log_loss"]
    )

    comparison_rows.append(
        "<tr>"
        f"<td>+{horizon}s</td>"
        f'<td>{horizon_results["B3"]["rows"]:,}</td>'
        f'<td>{fmt(horizon_results["B2"]["log_loss"])}</td>'
        f'<td>{fmt(horizon_results["B3"]["log_loss"])}</td>'
        f'<td>{delta["pooled_log_loss_delta"]:+.6f}</td>'
        f"<td>[{lower:+.6f}, {upper:+.6f}]</td>"
        f"<td>{reduction:.1f}%</td>"
        "</tr>"
    )


min_date = min(
    match["match_date"]
    for match in matches
)

max_date = max(
    match["match_date"]
    for match in matches
)


# ============================================================
# 7. V3 research-page content
# ============================================================

section = f"""
<header class="hero">
    <div class="kicker">
        Version 3 / Independent research record
    </div>

    <h1>Forecasting the bomb's next macro-zone.</h1>

    <p class="lead">
        At an observation time in a Mirage round, estimate the
        bomb's macro-zone 5 or 10 seconds later using only
        information available at or before the observation.

        This is a different prediction target from the
        V0/V2 round-level plant outcome.
    </p>

    <div class="badges">
        <span class="badge ok">
            V3 closed · independently evaluated
        </span>
        <span class="badge">
            53 development matches
        </span>
        <span class="badge">
            20 independent confirmation matches
        </span>
        <span class="badge">
            15 frozen zones · +5s / +10s
        </span>
    </div>
</header>

<section class="section">
    <div class="section-head">
        <h2>At a glance</h2>

        <p>
            One frozen B3 model per horizon was evaluated
            alongside three fixed baselines on identical
            confirmation observations.
        </p>
    </div>

    <div class="grid">
        <article class="card">
            <div class="stat-label">Confirmation matches</div>
            <div class="stat">20</div>
            <p>
                Later-period Mirage demos
                ({esc(min_date)} to {esc(max_date)}).
                The first 20 technically eligible candidates
                in the frozen queue.
            </p>
        </article>

        <article class="card">
            <div class="stat-label">Valid +5s observations</div>
            <div class="stat">
                {results["eligible_rows_by_horizon"]["5"]:,}
            </div>
            <p>
                Repeated within-match observations,
                not independent matches.
            </p>
        </article>

        <article class="card">
            <div class="stat-label">Valid +10s observations</div>
            <div class="stat">
                {results["eligible_rows_by_horizon"]["10"]:,}
            </div>
            <p>
                Evaluated separately from the +5s horizon.
            </p>
        </article>
    </div>
</section>

<section class="section">
    <div class="section-head">
        <h2>The frozen experiment</h2>

        <p>
            Development contains
            {dev["development_corpus"]["matches"]} matches:
            {dev["development_corpus"]["roles"]["V0_DEVELOPMENT"]}
            original development,
            {dev["development_corpus"]["roles"]["V2_RESERVE"]}
            V2 reserve, and
            {dev["development_corpus"]["roles"]["V3_FRESH"]}
            V3 fresh development matches.

            The V2 <code>D_CONFIRM</code> set was not reused
            as V3 development or confirmation.
        </p>
    </div>

    <div class="flow">
        <div class="node">
            <b>Current state at t</b><br>
            Zone · C4 source · XYZ · causal 1s motion
        </div>

        <span class="arrow">→</span>

        <div class="node">
            <b>Frozen B0–B3</b><br>
            Separate +5s / +10s probability vectors
        </div>

        <span class="arrow">→</span>

        <div class="node">
            <b>Future label</b><br>
            One of 15 Mirage macro-zones
        </div>
    </div>

    <p>
        {source_link("v3_research_design.md", "Research design")}
        ·
        {source_link("v3_b3_feature_protocol_v1.json", "Frozen B3 feature protocol")}
        ·
        {source_link("v3_macro_zone_mapping_v1_frozen.json", "Macro-zone definitions")}
    </p>
</section>

<section class="section">
    <div class="section-head">
        <h2>What each baseline asks</h2>

        <p>
            These four models address the same V3 target.
            They are not ranked against the earlier V0/V2
            plant-outcome models.
        </p>
    </div>

    <div class="grid">
        <article class="card half">
            <h3>B0 · Persistence</h3>
            <p>
                Predict that the bomb stays in its current
                macro-zone, with a frozen development-fitted
                probability adapter.
            </p>
        </article>

        <article class="card half">
            <h3>B1 · Constant motion</h3>
            <p>
                Project current XYZ using one second of
                causal velocity. Resolve the projected XYZ
                against frozen development-only spatial samples
                and apply the frozen probability adapter.
            </p>
        </article>

        <article class="card half">
            <h3>B2 · Zone Markov</h3>
            <p>
                Use the development-fitted conditional
                probability of future macro-zone given
                current macro-zone, separately for each horizon.
            </p>
        </article>

        <article class="card half">
            <h3>B3 · Tabular map-aware</h3>
            <p>
                Use a frozen 24-feature XGBoost model combining
                one-hot current zone, C4 state, XYZ, velocity
                and speed. Separate models predict +5s and +10s.
            </p>
        </article>
    </div>
</section>

<section class="section">
    <div class="section-head">
        <h2>Formal independent confirmation</h2>

        <p>
            Primary comparison: B3 minus B2 pooled multiclass
            Log Loss, reported separately for each horizon.

            Negative differences mean lower B3 Log Loss
            on the frozen confirmation sample.
        </p>
    </div>

    <div
        class="table-wrap"
        role="region"
        aria-label="V3 paired confirmation results"
        tabindex="0"
    >
        <table>
            <thead>
                <tr>
                    <th>Horizon</th>
                    <th>Rows</th>
                    <th>B2 Log Loss</th>
                    <th>B3 Log Loss</th>
                    <th>B3 − B2</th>
                    <th>95% match-bootstrap CI</th>
                    <th>Observed relative reduction</th>
                </tr>
            </thead>

            <tbody>
                {"".join(comparison_rows)}
            </tbody>
        </table>
    </div>

    <p>
        The 95% percentile intervals use 10,000 paired,
        match-level bootstrap resamples (seed 42).

        Both intervals are below zero for this frozen
        20-match sample.

        This does not establish performance on other maps,
        populations or real-time player-view inputs.
    </p>
</section>

<section class="section">
    <div class="section-head">
        <h2>All four frozen models</h2>

        <p>
            Log Loss and Brier: lower is better.
            Accuracy and macro F1: higher is better.

            Equal-match Log Loss gives each match the
            same weight rather than weighting by its
            number of observations.
        </p>
    </div>

    <div
        class="table-wrap"
        role="region"
        aria-label="V3 complete model comparison"
        tabindex="0"
    >
        <table>
            <thead>
                <tr>
                    <th>Horizon</th>
                    <th>Model</th>
                    <th>Log Loss</th>
                    <th>Accuracy</th>
                    <th>Macro F1</th>
                    <th>Multiclass Brier</th>
                    <th>Equal-match Log Loss</th>
                </tr>
            </thead>

            <tbody>
                {"".join(metric_rows)}
            </tbody>
        </table>
    </div>
</section>

<section class="section">
    <div class="section-head">
        <h2>Per-zone audit · B3</h2>

        <p>
            These are hard-label recall and support,
            not confidence intervals.

            Small-support zones are particularly uncertain
            and should not be generalized from a handful
            of observations.
        </p>
    </div>

    <div
        class="table-wrap"
        role="region"
        aria-label="B3 per-zone recall and support"
        tabindex="0"
    >
        <table>
            <thead>
                <tr>
                    <th>Target zone</th>
                    <th>+5s support</th>
                    <th>+5s recall</th>
                    <th>+10s support</th>
                    <th>+10s recall</th>
                </tr>
            </thead>

            <tbody>
                {"".join(zone_rows)}
            </tbody>
        </table>
    </div>
</section>

<section class="section">
    <div class="section-head">
        <h2>Research boundary and next questions</h2>

        <p>
            V3 is an observer/replay offline forecasting
            experiment on Mirage.

            It does not validate live machine capture,
            player-view partial observation, other maps,
            tactical prescriptions or a deployed assistant.
        </p>
    </div>

    <div class="callout warn">
        <strong>Do not retune on D_V3_CONFIRM.</strong>

        All 20 matches are now exposed by formal scoring.

        V4 must define a new problem or experiment and obtain
        suitable new evaluation evidence.
    </div>

    <p>
        Potential next investigations include descriptive
        error analysis by zone and movement regime, the value
        of causally observable team context, and how to present
        forecasts as observer/replay research signals.

        None is a frozen V4 experiment yet.
    </p>
</section>

<section class="section">
    <div class="section-head">
        <h2>Reproducible record</h2>

        <p>
            The website republishes frozen research files.
            Its HTML contains no new model inference,
            training or rescoring.
        </p>
    </div>

    <p>
        {source_link("v3_closure_v1.md", "V3 closure report")}
        ·
        {source_link("v3_confirm_results_v1.json", "Full formal metrics and per-zone evidence")}
        ·
        {source_link("v3_confirm_manifest_identity.json", "Manifest identity")}
        ·
        {source_link("v3_confirm_scoring_protocol_v1.json", "Frozen scoring protocol")}
        ·
        {source_link("v3_confirm_model_freeze.json", "Frozen model identities")}
        ·
        {source_link("v3_development_closure_v1.json", "Development closure")}
    </p>

    <p>
        <small>
            Manifest SHA256:
            <code>{esc(results["manifest_sha256"])}</code>
            <br>

            Scoring protocol SHA256:
            <code>{esc(results["scoring_protocol_sha256"])}</code>
            <br>

            Frozen model record SHA256:
            <code>{esc(results["model_freeze_sha256"])}</code>
        </small>
    </p>
</section>
"""


footer = (
    '<footer class="footer">'
    '<a href="../index.html">CS2 Tactical Intelligence Lab</a>'
    "<span>V3 independent confirmation complete"
    " · September 2026</span>"
    '<a href="../pages/roadmap.html#v3-release">'
    "Research roadmap ↗</a>"
    "</footer>"
)

plans[OUTPUT] = (
    prefix
    + section
    + footer
    + suffix
)


# ============================================================
# 8. Update the site manifest without erasing V0–V2 history
# ============================================================

manifest_path = SITE / "site-manifest.json"

site_manifest = json.loads(
    manifest_path.read_text()
)

check(
    "pages/v3.html" not in site_manifest["pages"],
    "Site manifest already contains V3.",
)

index = site_manifest["pages"].index(
    "pages/v2.html"
)

site_manifest["pages"].insert(
    index + 1,
    "pages/v3.html",
)

site_manifest["stage"] = (
    "V3 independent confirmation complete"
)

site_manifest["current_stage"] = (
    "V3 independent confirmation complete; "
    "V4 not frozen"
)

site_manifest["v2_status"] = (
    "INDEPENDENT CONFIRMATION COMPLETE "
    "· V0 REMAINS PRIMARY FOR V0/V2 TASK"
)

site_manifest["baseline_status"] = (
    "Corrected V0/H2 rebuilt and evaluated; "
    "V0 retained for the V0/V2 plant-outcome task"
)

site_manifest["next_gate"] = (
    "Publish V3 research record; define separate V4 "
    "research question before new data or models"
)

site_manifest["v3_status"] = (
    "FORMAL INDEPENDENT CONFIRMATION COMPLETE "
    "· VERSION CLOSED"
)

site_manifest["v3_selected_model"] = (
    "B3_TABULAR_MAP_AWARE_V1 "
    "(V3 bomb-zone task only)"
)

site_manifest["v3_confirmation"] = {
    "status": "complete",
    "task": "Mirage bomb macro-zone at +5s / +10s",
    "development_matches": 53,
    "confirmation_matches": 20,
    "eligible_rows_by_horizon": results[
        "eligible_rows_by_horizon"
    ],
    "primary_comparison": (
        "B3 minus B2 pooled multiclass log loss"
    ),
    "results_by_horizon": {
        horizon: {
            "b2_log_loss": results[
                "results_by_horizon"
            ][horizon]["B2"]["log_loss"],

            "b3_log_loss": results[
                "results_by_horizon"
            ][horizon]["B3"]["log_loss"],

            "b3_minus_b2": results[
                "results_by_horizon"
            ][horizon]["B3_minus_B2"][
                "pooled_log_loss_delta"
            ],

            "match_bootstrap_95ci": results[
                "results_by_horizon"
            ][horizon]["B3_minus_B2"][
                "match_bootstrap_95ci"
            ],
        }
        for horizon in ("5", "10")
    },
    "manifest_sha256": results[
        "manifest_sha256"
    ],
    "source": (
        "assets/sources/v3_confirm_results_v1.json"
    ),
}

plans[manifest_path] = (
    json.dumps(
        site_manifest,
        ensure_ascii=False,
        indent=2,
    )
    + "\n"
)


# ============================================================
# 9. Validate the planned website before writing any changes
# ============================================================

expected_pages = [
    SITE / "index.html",
    *PAGES.glob("*.html"),
    OUTPUT,
]

check(
    len(plans) == len(expected_pages) + 1,
    "Not every website page and manifest is included.",
)

for path, content in plans.items():

    if path.suffix != ".html":
        continue

    check(
        content.count('<nav class="nav">') == 1,
        f"Navigation missing or duplicated: {path}",
    )

    check(
        content.count('data-page="v3"') >= 1,
        f"V3 navigation missing: {path}",
    )

    expected_release_sections = (
        0 if path == OUTPUT else 1
    )

    check(
        content.count('id="v3-release"')
        == expected_release_sections,
        f"V3 release section invalid: {path}",
    )

    check(
        content.count('<footer class="footer">') == 1,
        f"Footer missing or duplicated: {path}",
    )

    v3_href = (
        "pages/v3.html"
        if path == SITE / "index.html"
        else "../pages/v3.html"
    )

    check(
        f'href="{v3_href}"' in content
        or path == OUTPUT,
        f"V3 navigation link missing: {path}",
    )


# ============================================================
# 10. Write the website release
# ============================================================

for path, content in plans.items():
    path.write_text(content)

for name in SOURCE_NAMES:
    shutil.copy2(
        ROOT / "docs" / name,
        SOURCES / name,
    )


print()
print("=" * 64)
print("V3_SITE_RELEASE_READY")
print("=" * 64)

print("Site pages:", len(expected_pages))
print("Published frozen evidence files:", len(SOURCE_NAMES))

print(
    "Primary +5s B3-B2:",
    fmt(
        results["results_by_horizon"]["5"][
            "B3_minus_B2"
        ]["pooled_log_loss_delta"]
    ),
)

print(
    "Primary +10s B3-B2:",
    fmt(
        results["results_by_horizon"]["10"][
            "B3_minus_B2"
        ]["pooled_log_loss_delta"]
    ),
)

print()
print("New research page: site/pages/v3.html")
print("No models, raw demos, or scoring outputs were modified.")
print("Next: preview the website before committing.")
