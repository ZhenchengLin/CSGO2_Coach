from __future__ import annotations

import importlib.metadata
import sys
from pathlib import Path


def vec3(v):
    if all(
        hasattr(v, name)
        for name in ("x", "y", "z")
    ):
        return (
            float(v.x),
            float(v.y),
            float(v.z),
        )

    try:
        return (
            float(v[0]),
            float(v[1]),
            float(v[2]),
        )
    except Exception:
        return None


print("=" * 88)
print("V3 GATE 0 — MIRAGE NAV ENVIRONMENT AUDIT")
print("=" * 88)

print()
print("Python:", sys.version.split()[0])

try:
    awpy_version = importlib.metadata.version("awpy")
except Exception:
    awpy_version = "UNKNOWN"

print("Awpy:", awpy_version)

if awpy_version != "2.0.2":
    print(
        "⚠️  Expected frozen parser contract: awpy 2.0.2"
    )
else:
    print("✅ Awpy version contract matches V2")

print()

try:
    from awpy.nav import Nav
except Exception as exc:
    print("❌ Cannot import awpy.nav.Nav")
    print(repr(exc))
    raise SystemExit(1)

print("Nav API")
print("  from_json:", hasattr(Nav, "from_json"))
print("  from_path:", hasattr(Nav, "from_path"))
print("  find_path:", hasattr(Nav, "find_path"))

print()

root = Path.home() / ".awpy"

print("Awpy data root:")
print(" ", root)

if not root.exists():
    print("❌ ~/.awpy does not exist")
    print()
    print("NEXT_ACTION=DOWNLOAD_AWPY_MAP_DATA")
    raise SystemExit(0)

candidates = sorted(
    p
    for p in root.rglob("*")
    if (
        p.is_file()
        and "mirage" in p.name.lower()
        and p.suffix.lower()
        in {".json", ".nav", ".tri", ".mesh", ".png"}
    )
)

print()
print("Mirage assets:")

if not candidates:
    print("  NONE")
else:
    for p in candidates:
        try:
            rel = p.relative_to(root)
        except ValueError:
            rel = p

        print(
            f"  {rel}"
            f" | {p.stat().st_size:,} bytes"
        )

nav_candidates = [
    p
    for p in candidates
    if p.suffix.lower()
    in {".json", ".nav"}
]

if not nav_candidates:
    print()
    print("⚠️  No Mirage nav file found.")
    print("NEXT_ACTION=DOWNLOAD_NAVS")
    raise SystemExit(0)

print()
print("-" * 88)
print("NAV LOAD TEST")
print("-" * 88)

loaded = None
loaded_path = None

for path in nav_candidates:

    try:
        if path.suffix.lower() == ".json":
            nav = Nav.from_json(path)
        else:
            nav = Nav.from_path(path)

    except Exception as exc:
        print()
        print("Could not load:")
        print(" ", path)
        print(" ", type(exc).__name__, str(exc))
        continue

    loaded = nav
    loaded_path = path

    print()
    print("✅ Loaded:")
    print(" ", path)
    break

if loaded is None:
    print()
    print("❌ No Mirage nav candidate could be loaded.")
    print("NEXT_ACTION=INVESTIGATE_NAV_FORMAT")
    raise SystemExit(1)

areas = loaded.areas

print()
print("Nav metadata")
print("  version:", getattr(loaded, "version", None))
print(
    "  sub_version:",
    getattr(loaded, "sub_version", None),
)
print(
    "  is_analyzed:",
    getattr(loaded, "is_analyzed", None),
)
print("  areas:", len(areas))

if not areas:
    print("❌ Nav contains zero areas")
    raise SystemExit(1)

centroids = []
sizes = []
connection_counts = []

for area in areas.values():

    c = vec3(area.centroid)

    if c is not None:
        centroids.append(c)

    try:
        sizes.append(float(area.size))
    except Exception:
        pass

    try:
        connection_counts.append(
            len(area.connected_areas)
        )
    except Exception:
        pass

print()
print("Geometry audit")
print("  valid centroids:", len(centroids))
print("  valid sizes:", len(sizes))

if centroids:
    xs = [x for x, _, _ in centroids]
    ys = [y for _, y, _ in centroids]
    zs = [z for _, _, z in centroids]

    print(
        "  X range:",
        f"{min(xs):.2f} .. {max(xs):.2f}",
    )

    print(
        "  Y range:",
        f"{min(ys):.2f} .. {max(ys):.2f}",
    )

    print(
        "  Z range:",
        f"{min(zs):.2f} .. {max(zs):.2f}",
    )

if sizes:
    print(
        "  area size range:",
        f"{min(sizes):.2f} .. {max(sizes):.2f}",
    )

if connection_counts:
    isolated = sum(
        n == 0
        for n in connection_counts
    )

    print(
        "  mean connections:",
        f"{sum(connection_counts) / len(connection_counts):.2f}",
    )

    print(
        "  isolated areas:",
        isolated,
    )

print()
print("-" * 88)
print("PATHFINDING SMOKE TEST")
print("-" * 88)

path_tested = False

for area_id, area in areas.items():

    try:
        neighbors = sorted(
            area.connected_areas
        )
    except Exception:
        continue

    if not neighbors:
        continue

    neighbor = neighbors[0]

    if neighbor not in areas:
        continue

    try:
        path = loaded.find_path(
            area_id,
            neighbor,
            weight="dist",
        )
    except Exception as exc:
        print(
            "❌ find_path failed:",
            type(exc).__name__,
            str(exc),
        )
        raise SystemExit(1)

    print("start area:", area_id)
    print("neighbor:", neighbor)
    print("path length:", len(path))

    if path:
        print(
            "path ids:",
            [
                a.area_id
                for a in path
            ],
        )

    path_tested = True
    break

if not path_tested:
    print(
        "⚠️  Could not find a connected "
        "area pair for smoke test."
    )

print()
print("-" * 88)

if (
    len(centroids) == len(areas)
    and len(sizes) == len(areas)
    and path_tested
):
    print("✅ V3 NAV ENVIRONMENT AUDIT PASS")
    print("NEXT_ACTION=BUILD_MIRAGE_NAV_INDEX")
else:
    print("⚠️  V3 NAV ENVIRONMENT AUDIT INCOMPLETE")
    print("NEXT_ACTION=INVESTIGATE_NAV_DATA")

print("-" * 88)
