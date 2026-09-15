#!/usr/bin/env python3
"""Build (or rebuild) the local-file retrieval index.

The HUD can do everything this script does, but a command line is what you reach
for the first time — and it is the honest way to see what the indexer actually
touched, because it prints every refusal with its reason.

    python tools/build_rag_index.py --root "C:\\Users\\me\\Documents"
    python tools/build_rag_index.py --root ~/notes --root ~/projects --force
    python tools/build_rag_index.py --report
    python tools/build_rag_index.py --ask "מה כתוב על חומת ההרשאות"
    python tools/build_rag_index.py --include "*.md" --exclude "*.log"

Nothing here talks to the network. The index is a single SQLite file under
``data/`` (git-ignored); deleting it costs you a rebuild, not your documents.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from brain.rag.engine import RagEngine, get_engine  # noqa: E402
from brain.rag.extract import human_bytes  # noqa: E402
from core.config import CONFIG  # noqa: E402


def _print_report(rep: dict) -> None:
    print("\n  ── index report " + "─" * 48)
    print(f"  roots       {', '.join(rep.get('roots') or []) or '(none)'}")
    for key in ("ok", "added", "updated", "unchanged", "skipped", "pruned", "denied"):
        if rep.get(key):
            print(f"  {key:<11} {rep[key]}")
    print(f"  chunks      {rep.get('chunks', 0)}")
    print(f"  read        {human_bytes(int(rep.get('bytes') or 0))}")
    print(f"  seconds     {rep.get('seconds', 0)}")
    fit = rep.get("vectorizer") or {}
    if fit:
        print(f"  vectoriser  {fit.get('source')} over {fit.get('chunks')} chunks")
    reasons = rep.get("skip_reasons") or {}
    if reasons:
        print("  skipped because:")
        for reason, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
            print(f"      {n:>5}  {reason}")
    if rep.get("error"):
        print(f"  ERROR       {rep['error']}")


def _print_status(eng: RagEngine) -> None:
    st = eng.status()
    print("\n  ── index status " + "─" * 47)
    print(f"  db          {st.get('db_path')}")
    print(f"  documents   {st.get('docs', 0)}")
    print(f"  chunks      {st.get('chunks', 0)}")
    print(f"  terms       {st.get('distinct_terms', 0)}")
    print(f"  avg terms   {st.get('avg_chunk_terms', 0)} per chunk")
    print(f"  indexed     {human_bytes(int(st.get('indexed_bytes') or 0))}")
    print(f"  roots       {', '.join(st.get('roots') or []) or '(none configured)'}")
    print(f"  vectoriser  {'ready' if st.get('vectorizer_ready') else 'not fitted yet'}")
    last = st.get("last_index") or {}
    if last:
        print(f"  last run    {last.get('ok', 0)} files, {last.get('chunks', 0)} chunks, "
              f"{last.get('seconds', 0)}s")
    skipped = st.get("skip_reasons") or {}
    if skipped:
        total = sum(skipped.values())
        top = ", ".join(f"{r}×{n}" for r, n in list(skipped.items())[:4])
        print(f"  skipped     {total} ({top})")


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the JARVIS local-file retrieval index.")
    ap.add_argument("--root", action="append", default=[],
                    help="folder to index (repeatable); default: the configured roots")
    ap.add_argument("--include", default="", help="comma separated glob allowlist")
    ap.add_argument("--exclude", default="", help="comma separated glob denylist (extra)")
    ap.add_argument("--force", action="store_true", help="re-read every file, ignore mtime")
    ap.add_argument("--report", action="store_true", help="print index status and exit")
    ap.add_argument("--list", action="store_true", help="list indexed documents and exit")
    ap.add_argument("--ask", default="", help="answer a question from the index, with citations")
    ap.add_argument("--search", default="", help="retrieval only: show ranked chunks")
    ap.add_argument("--k", type=int, default=8, help="how many results (default 8)")
    ap.add_argument("--forget", default="", help="remove one path from the index")
    ap.add_argument("--db", default="", help="index database path (default: config)")
    args = ap.parse_args()

    eng = RagEngine(db_path=args.db) if args.db else get_engine()

    if args.include:
        eng.config.include = tuple(
            list(eng.config.include) + [p.strip() for p in args.include.split(",") if p.strip()])
    if args.exclude:
        eng.config.exclude = tuple(
            list(eng.config.exclude) + [p.strip() for p in args.exclude.split(",") if p.strip()])

    if args.forget:
        removed = eng.forget(args.forget)
        print(f"[{'✓' if removed else '!'}] {args.forget}: "
              f"{'removed from the index' if removed else 'was not in the index'}")
        return 0 if removed else 1

    # --report on its own means "just show me the state". Combined with --ask or
    # --search it is not an exit — it would silently swallow the question, which
    # is what the first version did.
    if args.report and not (args.ask or args.search):
        _print_status(eng)
        return 0

    if args.list:
        docs = eng.docs(1000)
        if not docs:
            print("[!] the index is empty — run without --list to build it")
            return 1
        print(f"\n  {len(docs)} indexed documents")
        for d in docs:
            print(f"    {d['chunks']:>5} chunks  {human_bytes(int(d['bytes'])):>9}  "
                  f"{d['kind']:<6} {d['path']}")
        return 0

    roots = [Path(r).expanduser() for r in args.root] or None
    if roots is None and not eng._effective_roots():
        print("[!] no roots configured. Pass --root <folder> or set rag.roots in jarvis.local.json")
        return 2

    missing = [str(r) for r in (roots or []) if not r.exists()]
    if missing:
        print(f"[!] these paths do not exist: {', '.join(missing)}")
        return 2

    print(f"[*] indexing {len(eng._effective_roots(roots))} root(s)…")
    t0 = time.perf_counter()
    rep = eng.index(roots, force=args.force,
                    progress=lambda done, seen, path: print(
                        f"    … {done} files ({Path(path).name})", flush=True))
    _print_report(rep)
    if rep.get("error"):
        return 1

    if args.search:
        hits = eng.search(args.search, k=args.k)
        print(f"\n  ── search: {args.search!r} — {len(hits)} hits ──")
        for i, h in enumerate(hits, 1):
            print(f"  [{i}] {h.score:.3f}  {h.file} · {h.span}")
            print(f"       {h.text[:160].replace(chr(10), ' ')}")
        exp = eng.retriever.last_expansions
        if any(exp.values()):
            print(f"  expansions: {json.dumps({k: v for k, v in exp.items() if v}, ensure_ascii=False)}")

    if args.ask:
        ans = eng.ask(args.ask, k=args.k)
        print(f"\n  ── ask: {args.ask!r} ──")
        print(f"  type={ans.answer_type}  grounded={ans.grounded}  confidence={ans.confidence}")
        print()
        for line in ans.text.split("\n"):
            print(f"  {line}")
        verified, problems = eng.verify(ans)
        print(f"\n  grounding re-checked against the files on disk: "
              f"{'PASS' if verified else 'FAIL ' + str(problems)}")
        if not verified:
            return 1

    print(f"\n[✓] done in {time.perf_counter() - t0:.1f}s — index at {eng.db_path}")
    _print_status(eng)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
