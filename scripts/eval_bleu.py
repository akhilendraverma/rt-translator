"""
Corpus BLEU for the NMT service against a parallel test set (e.g. a FLORES-200 slice).
Expects a TSV with columns: source<TAB>reference.

Usage: python scripts/eval_bleu.py --url http://localhost:8002 --tsv flores_en_es.tsv --source en --target es
"""
import argparse
import csv

import httpx
import sacrebleu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8002")
    ap.add_argument("--tsv", required=True)
    ap.add_argument("--source", default="en")
    ap.add_argument("--target", default="es")
    args = ap.parse_args()

    hyps, refs = [], []
    with httpx.Client(timeout=60) as c, open(args.tsv, encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) < 2:
                continue
            src, ref = row[0], row[1]
            r = c.post(f"{args.url}/translate",
                       json={"text": src, "source": args.source, "target": args.target})
            hyps.append(r.json()["translation"])
            refs.append(ref)

    bleu = sacrebleu.corpus_bleu(hyps, [refs])
    verdict = "PASS" if bleu.score >= 38 else "below target"
    print(f"BLEU = {bleu.score:.2f}  (target >= 38)  -> {verdict}")


if __name__ == "__main__":
    main()
