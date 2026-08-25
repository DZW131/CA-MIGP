from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from pathm.data.manifest import load_manifest, manifest_sha256, validate_manifest  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--skip-paths", action="store_true")
    args = parser.parse_args()

    records = load_manifest(args.manifest)
    validate_manifest(records, check_paths=not args.skip_paths)
    split_counts: dict[str, int] = {}
    domain_counts: dict[str, int] = {}
    for record in records:
        split_counts[record.split] = split_counts.get(record.split, 0) + 1
        domain_counts[record.domain] = domain_counts.get(record.domain, 0) + 1
    print("manifest_ok")
    print("records", len(records))
    print("splits", split_counts)
    print("domains", domain_counts)
    print("sha256", manifest_sha256(args.manifest))


if __name__ == "__main__":
    main()
