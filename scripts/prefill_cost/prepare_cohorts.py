"""Prepare equal-weight context cohorts from existing authorized archive inputs.

Raw inputs stay outside Git. >=256K samples concatenate distinct real archives;
this composition is explicitly recorded rather than called a native long trace.
"""

import argparse
import hashlib
import json
from pathlib import Path

from tokenizers import Tokenizer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--tokenizer", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False, mode=0o700)
    source = json.loads(args.manifest.read_text())["requests"]
    assert len(source) >= 200
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    records = []
    for index, row in enumerate(source):
        raw = Path(row["path"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == row["sha256"]
        archive = json.loads(raw)["archive"]
        target = (4096, 32768, 98304, 294912)[index % 4]
        parts = [archive]
        inputs = [row["sha256"]]
        tokens = tokenizer.encode(archive, add_special_tokens=False).ids
        offset = 1
        while len(tokens) < target:
            other = source[(index + offset) % len(source)]
            value = Path(other["path"]).read_bytes()
            assert hashlib.sha256(value).hexdigest() == other["sha256"]
            parts.append(json.loads(value)["archive"])
            inputs.append(other["sha256"])
            tokens = tokenizer.encode(
                "\n--- archive boundary ---\n".join(parts), add_special_tokens=False
            ).ids
            offset += 1
        text = tokenizer.decode(tokens[:target], skip_special_tokens=False)
        count = len(tokenizer.encode(text, add_special_tokens=False).ids)
        output = args.out / f"{index:04d}.json"
        output.write_text(json.dumps({"archive": text}, ensure_ascii=False))
        records.append(
            {
                "path": str(output),
                "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "target_archive_tokens": target,
                "archive_tokens": count,
                "sources": inputs,
                "composed": len(parts) > 1,
            }
        )
    manifest = {
        "schema": 1,
        "tokenizer_sha256": hashlib.sha256(args.tokenizer.read_bytes()).hexdigest(),
        "source_manifest_sha256": hashlib.sha256(
            args.manifest.read_bytes()
        ).hexdigest(),
        "note": "Archive token counts exclude chat template and cold nonce; final usage is authoritative.",
        "requests": records,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(
        json.dumps(
            {
                "requests": len(records),
                "buckets": [4096, 32768, 98304, 294912],
                "manifest": str(args.out / "manifest.json"),
            }
        )
    )


if __name__ == "__main__":
    main()
