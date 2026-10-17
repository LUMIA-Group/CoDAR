"""Convert explicitly split text/JSONL files into the main trainers' uint32 format."""
import argparse
import json
from pathlib import Path

import numpy as np


def documents(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            text = json.loads(line)["text"] if path.suffix == ".jsonl" else line.rstrip("\n")
            if not isinstance(text, str):
                raise ValueError("Each document must be a string")
            if text:
                yield text


def write_tokens(path, destination, tokenizer, append_eos):
    count = 0
    with destination.open("xb") as stream:
        for text in documents(path):
            ids = tokenizer.encode(text, add_special_tokens=False)
            if append_eos:
                if tokenizer.eos_token_id is None:
                    raise ValueError("Tokenizer has no EOS token")
                ids.append(tokenizer.eos_token_id)
            if any(i < 0 or i > np.iinfo(np.uint32).max for i in ids):
                raise ValueError("Token IDs must fit uint32")
            np.asarray(ids, dtype=np.uint32).tofile(stream)
            count += len(ids)
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_file", type=Path, required=True)
    parser.add_argument("--val_file", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--append_eos", action="store_true")
    args = parser.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    destinations = [args.output_dir / f"{split}_qwen.bin" for split in ("train", "val")]
    if any(p.exists() for p in destinations):
        parser.error("Output files already exist; choose an empty output directory")
    counts = {}
    for split, source, destination in zip(("train", "val"), (args.train_file, args.val_file), destinations):
        counts[split] = write_tokens(source, destination, tokenizer, args.append_eos)
    metadata = dict(tokenizer=args.tokenizer, append_eos=args.append_eos,
                    dtype="uint32", token_counts=counts,
                    train_file=str(args.train_file), val_file=str(args.val_file))
    (args.output_dir / "preprocessing.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
