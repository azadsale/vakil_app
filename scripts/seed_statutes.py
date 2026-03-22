#!/usr/bin/env python3
"""Seed script — ingests DV Act PDFs into the backend via the admin API.

Usage:
    python scripts/seed_statutes.py
    python scripts/seed_statutes.py --base-url http://localhost:8000

Run this ONCE after 'make up' and 'make migrate' to index the legal PDFs.
Re-running is safe — the API creates new entries each time, so only run once
or clear the legal_documents table first.
"""

import argparse
import sys
from pathlib import Path

import requests

STATUTES = [
    {
        "file": Path(__file__).parent / "dv_act_2005.pdf",
        "title": "Protection of Women from Domestic Violence Act 2005",
        "short_name": "DV Act 2005",
        "doc_type": "dv_act_2005",
        "jurisdiction": "India",
        "effective_date": "2005-09-13",
    },
    {
        "file": Path(__file__).parent / "Domestic_Voilence.pdf",
        "title": "Domestic Violence Reference Document",
        "short_name": "DV Reference",
        "doc_type": "dv_act_2005",
        "jurisdiction": "India",
        "effective_date": None,
    },
    {
        "file": Path(__file__).parent / "Draft_DV.pdf",
        "title": "DV Petition Draft Template (Lawyer Reference)",
        "short_name": "DV Draft Template",
        "doc_type": "dv_act_2005",
        "jurisdiction": "India",
        "effective_date": None,
    },
]


def upload_statute(base_url: str, statute: dict) -> None:
    pdf_path: Path = statute["file"]
    if not pdf_path.exists():
        print(f"  [SKIP] File not found: {pdf_path}")
        return

    print(f"  Uploading: {pdf_path.name} ({pdf_path.stat().st_size // 1024} KB)")

    with pdf_path.open("rb") as f:
        files = {"pdf_file": (pdf_path.name, f, "application/pdf")}
        data = {
            "title": statute["title"],
            "short_name": statute["short_name"],
            "doc_type": statute["doc_type"],
            "jurisdiction": statute["jurisdiction"],
        }
        if statute.get("effective_date"):
            data["effective_date"] = statute["effective_date"]

        resp = requests.post(
            f"{base_url}/api/v1/admin/upload-statute",
            files=files,
            data=data,
            timeout=300,  # embedding can take a few minutes
        )

    if resp.status_code == 201:
        result = resp.json()
        print(f"  [OK] {result['short_name']} — {result['total_chunks']} chunks indexed")
        print(f"       Document ID: {result['document_id']}")
    else:
        print(f"  [FAIL] HTTP {resp.status_code}: {resp.text}")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed legal statute PDFs into vakil backend")
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Backend base URL (default: http://localhost:8000)",
    )
    args = parser.parse_args()

    print(f"\nSeeding statutes into {args.base_url}")
    print("=" * 50)

    for statute in STATUTES:
        print(f"\n{statute['short_name']}")
        upload_statute(args.base_url, statute)

    print("\nDone. All statutes indexed successfully.")


if __name__ == "__main__":
    main()
