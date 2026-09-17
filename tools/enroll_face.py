#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Enroll a face from the command line — and claim the owner slot.

    python tools/enroll_face.py --image oscar.png
    python tools/enroll_face.py --image oscar.png --name OSCAR
    python tools/enroll_face.py --list
    python tools/enroll_face.py --owner p001-...

Why this exists when the HUD already has an enrol button
--------------------------------------------------------
The first face JARVIS is ever taught becomes the owner and gets full CRITICAL
(see ``vision/faces.py``). That is a decision worth making deliberately, from a
terminal, with the gallery visible — not as a side effect of clicking something
in a panel while checking whether the camera works. This tool is also the only
way to enrol from a file, which matters on a machine whose camera is attached to
something other than the box running JARVIS.

The first enrolment is the one that counts
------------------------------------------
If the gallery is empty, whoever you enrol here becomes the owner regardless of
``--level``. The tool says so out loud rather than letting it happen silently.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _load_rgb(path: Path) -> np.ndarray:
    """Read an image file into an (H, W, 3) uint8 array."""
    from vision.png import decode_file, is_png
    if not is_png(path):
        raise SystemExit(
            f"{path.name}: JARVIS decodes PNG with its own reader (vision/png.py) — "
            "no third-party image library is bundled. Convert the file to PNG first.")
    img = decode_file(path)
    rgb = img.rgb()
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb


def _cmd_enroll(args: argparse.Namespace) -> int:
    from vision.faces import (DEFAULT_OWNER_NAME, FaceStore, LEVELS, OWNER_LEVEL,
                              encode_frame)
    from vision.scene import describe

    store = FaceStore(path=Path(args.gallery) if args.gallery else None)
    path = Path(args.image)
    if not path.exists():
        print(f"✗ no such file: {path}")
        return 2
    rgb = _load_rgb(path)
    print(f"  read {path.name} — {rgb.shape[1]}x{rgb.shape[0]} px")

    scene = describe(rgb)
    if scene.people == 0:
        print("✗ לא נמצאו פנים בתמונה.")
        print("  ודא שהפנים ממלאות חלק ניכר מהפריים, שהתאורה מקדימה ולא מאחורה,")
        print("  ושהרקע לא בצבע עור.")
        if scene.lighting:
            print(f"  תאורה: {scene.lighting.verdict} (ממוצע {scene.lighting.mean:.0f})")
        return 1
    if scene.warnings:
        print("  אזהרות: " + "; ".join(scene.warnings))
    print(f"  {scene.summary_he}")

    face = encode_frame(rgb)
    if face is None:
        print("✗ הפנים אותרו אך לא ניתן היה לקודד אותן.")
        return 1
    if not face.eyes:
        print("  ⚠ העיניים לא אותרו — היישור לפי תיבת הפנים בלבד, פחות מדויק.")

    will_be_owner = store.owner() is None
    name = args.name or (DEFAULT_OWNER_NAME if will_be_owner else "")
    if not name:
        print("✗ a --name is required (the gallery already has an owner)")
        return 2
    level = (args.level or ("CRITICAL" if will_be_owner else "SAFE")).upper()
    if level not in LEVELS:
        print(f"✗ --level must be one of {list(LEVELS)}")
        return 2

    if will_be_owner:
        print(f"\n  הגלריה ריקה — זו ההרשמה הראשונה.")
        print(f"  {name} הופך לבעלים: תפקיד owner, הרשאות {OWNER_LEVEL} (הכול).")
        print(f"  רק הבעלים יכול להגיע ל-{OWNER_LEVEL}; כל אדם אחר מוגבל ל-WRITE.\n")

    person = store.enroll(name, face.vector, level=level, note=args.note or "")
    print(f"✓ enrolled {person.name}  id={person.id}")
    print(f"  role={person.role}  level={person.level}  samples={len(person.vectors)}")
    print(f"  gallery now holds {len(store.people)} people "
          f"({sum(len(p.vectors) for p in store.people)} samples)")
    owner = store.owner()
    if owner:
        print(f"  owner: {owner.name} ({owner.id})")
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    from vision.faces import FaceStore
    store = FaceStore(path=Path(args.gallery) if args.gallery else None)
    if not store.people:
        print("הגלריה ריקה — אף פנים לא נרשמו עדיין.")
        print("ההרשמה הראשונה תהפוך לבעלים עם הרשאות מלאות:")
        print("  python tools/enroll_face.py --image your_face.png")
        return 0
    print(f"{'ID':22} {'NAME':16} {'ROLE':7} {'LEVEL':9} SAMPLES")
    for p in store.people:
        print(f"{p.id:22} {p.name:16} {p.role:7} {p.level:9} {len(p.vectors)}")
    owner = store.owner()
    print(f"\nowner: {owner.name if owner else '— (none enrolled)'}")
    print(f"within={store._within:.3f} between={store._between:.3f} "
          f"threshold=0.85")
    return 0


def _cmd_owner(args: argparse.Namespace) -> int:
    from vision.faces import FaceStore
    store = FaceStore(path=Path(args.gallery) if args.gallery else None)
    if not store.set_owner(args.owner):
        print(f"✗ no such person: {args.owner}")
        return 2
    o = store.owner()
    print(f"✓ owner is now {o.name} ({o.id}) at {o.level}")
    print("  כל השאר הורדו ל-guest; מי שהיה ב-CRITICAL ירד ל-WRITE.")
    return 0


def _cmd_forget(args: argparse.Namespace) -> int:
    from vision.faces import FaceStore
    store = FaceStore(path=Path(args.gallery) if args.gallery else None)
    target = store.get(args.forget)
    if target is None:
        print(f"✗ no such person: {args.forget}")
        return 2
    if target.is_owner and not args.force:
        print(f"✗ {target.name} is the owner. Removing the owner leaves the machine")
        print("  with nobody able to reach CRITICAL. Re-run with --force if you mean it.")
        return 1
    store.remove(args.forget)
    print(f"✓ removed {target.name}")
    return 0


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Enroll faces, and claim the owner slot on first enrolment.")
    ap.add_argument("--image", help="PNG file to enrol from")
    ap.add_argument("--name", help="name for this person (default: OSCAR if first)")
    ap.add_argument("--level", help="SAFE | WRITE | CRITICAL (ignored for the owner)")
    ap.add_argument("--note", default="", help="free-text note stored with the person")
    ap.add_argument("--gallery", help="gallery.json path (default: data/faces/gallery.json)")
    ap.add_argument("--list", action="store_true", help="list the gallery and exit")
    ap.add_argument("--owner", metavar="PERSON_ID", help="make this person the owner")
    ap.add_argument("--forget", metavar="PERSON_ID", help="remove a person")
    ap.add_argument("--force", action="store_true", help="allow removing the owner")
    args = ap.parse_args(argv)

    if args.list:
        return _cmd_list(args)
    if args.owner:
        return _cmd_owner(args)
    if args.forget:
        return _cmd_forget(args)
    if not args.image:
        ap.print_help()
        return 2
    return _cmd_enroll(args)


if __name__ == "__main__":
    raise SystemExit(main())
