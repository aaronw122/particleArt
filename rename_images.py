"""
Rename existing images so v0 and v1 of the same scene share the same number.

Run AFTER the batch finishes:
    uv run python rename_images.py
"""

import json
from pathlib import Path
from collections import defaultdict

raw_dir = Path("images/raw")
manifest_path = raw_dir / "manifest.json"

if not manifest_path.exists():
    print("No manifest.json found. Run generate_images.py first.")
    exit(1)

manifest = json.loads(manifest_path.read_text())

# Group by scene
by_scene = defaultdict(list)
for entry in manifest:
    by_scene[(entry["category"], entry["scene"])].append(entry)

# Assign new sequential numbers per category
renames = []
new_manifest = []
count_by_category = defaultdict(int)

for (category, scene), entries in by_scene.items():
    count_by_category[category] += 1
    num = count_by_category[category]

    # Sort by filename so ordering is deterministic, then assign v0, v1, v2...
    for v, entry in enumerate(sorted(entries, key=lambda e: e["file"])):
        old_file = entry["file"]
        new_file = f"{category}_{num:03d}_v{v}.png"

        if old_file != new_file:
            renames.append((old_file, new_file))

        new_manifest.append({
            "file": new_file,
            "category": category,
            "scene": scene,
        })

# Preview renames
print(f"Found {len(renames)} files to rename:\n")
for old, new in renames[:20]:
    print(f"  {old} -> {new}")
if len(renames) > 20:
    print(f"  ... and {len(renames) - 20} more")

if not renames:
    print("Nothing to rename!")
    exit(0)

confirm = input(f"\nRename {len(renames)} files? [y/N] ")
if confirm.lower() != "y":
    print("Aborted.")
    exit(0)

# Rename to temp names first to avoid collisions
for old, new in renames:
    old_path = raw_dir / old
    tmp_path = raw_dir / f"_tmp_{new}"
    if old_path.exists():
        old_path.rename(tmp_path)

for old, new in renames:
    tmp_path = raw_dir / f"_tmp_{new}"
    new_path = raw_dir / new
    if tmp_path.exists():
        tmp_path.rename(new_path)

# Update manifest
manifest_path.write_text(json.dumps(new_manifest, indent=2))

print(f"\nDone! Renamed {len(renames)} files.")
print(f"Updated manifest: {manifest_path}")
