"""Build TrueType Latin Modern faces for the figures (free, local, deterministic).

Latin Modern ships as OpenType/CFF. This script converts the installed faces to
TrueType outlines (glyf/loca) with fontTools' cu2qu, keeping every layout table, so
GSUB ligatures (fi, fl, ff, ffi, ffl) and GPOS kerning survive. Matplotlib's PGF
backend then typesets figure text with LuaLaTeX + fontspec, and PDFs embed the
faces as CID TrueType (no Type 3 bitmaps), so they stay sharp at any zoom.

The GUST Font License asks that modified versions not reuse the original names, so
families are renamed "LM Roman TT", "LM Sans TT", and "LM Math TT". The output is
regenerated from the local TeX installation and is not committed.

    python scripts/build_fonts.py
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fontTools.pens.cu2quPen import Cu2QuPen  # noqa: E402
from fontTools.pens.ttGlyphPen import TTGlyphPen  # noqa: E402
from fontTools.ttLib import TTFont, newTable  # noqa: E402

from mathmage import paths  # noqa: E402

OUT = paths.FONTS / "latinmodern-ttf"

# (source OTF, output file stem, new family name, style name)
FACES = [
    ("lmroman10-regular.otf", "LMRomanTT10-Regular", "LM Roman TT 10", "Regular"),
    ("lmroman10-bold.otf", "LMRomanTT10-Bold", "LM Roman TT 10", "Bold"),
    ("lmroman10-italic.otf", "LMRomanTT10-Italic", "LM Roman TT 10", "Italic"),
    ("lmroman10-bolditalic.otf", "LMRomanTT10-BoldItalic", "LM Roman TT 10", "Bold Italic"),
    ("lmroman9-regular.otf", "LMRomanTT9-Regular", "LM Roman TT 9", "Regular"),
    ("lmroman9-bold.otf", "LMRomanTT9-Bold", "LM Roman TT 9", "Bold"),
    ("lmroman9-italic.otf", "LMRomanTT9-Italic", "LM Roman TT 9", "Italic"),
    ("lmroman8-regular.otf", "LMRomanTT8-Regular", "LM Roman TT 8", "Regular"),
    ("lmroman8-bold.otf", "LMRomanTT8-Bold", "LM Roman TT 8", "Bold"),
    ("lmroman8-italic.otf", "LMRomanTT8-Italic", "LM Roman TT 8", "Italic"),
    ("lmroman7-regular.otf", "LMRomanTT7-Regular", "LM Roman TT 7", "Regular"),
    ("lmroman7-bold.otf", "LMRomanTT7-Bold", "LM Roman TT 7", "Bold"),
    ("lmroman7-italic.otf", "LMRomanTT7-Italic", "LM Roman TT 7", "Italic"),
    ("lmsans10-regular.otf", "LMSansTT10-Regular", "LM Sans TT 10", "Regular"),
    ("lmsans10-bold.otf", "LMSansTT10-Bold", "LM Sans TT 10", "Bold"),
    ("lmsans10-oblique.otf", "LMSansTT10-Oblique", "LM Sans TT 10", "Oblique"),
    ("lmsans9-regular.otf", "LMSansTT9-Regular", "LM Sans TT 9", "Regular"),
    ("lmsans8-regular.otf", "LMSansTT8-Regular", "LM Sans TT 8", "Regular"),
    ("latinmodern-math.otf", "LMMathTT-Regular", "LM Math TT", "Regular"),
]


def kpsewhich(name: str) -> Path:
    exe = shutil.which("kpsewhich")
    if exe is None:
        raise SystemExit("kpsewhich not found: a TeX installation with Latin Modern is required")
    out = subprocess.run([exe, name], capture_output=True, text=True).stdout.strip()
    if not out:
        raise SystemExit(f"{name} not found in the TeX installation")
    return Path(out)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _quadratic_glyphs(glyph_set, max_err: float = 0.5) -> dict:
    glyphs = {}
    for name in glyph_set.keys():
        tt_pen = TTGlyphPen(glyph_set)
        glyph_set[name].draw(Cu2QuPen(tt_pen, max_err, reverse_direction=True))
        glyphs[name] = tt_pen.glyph()
    return glyphs


def otf_to_ttf(src: Path, dst: Path, family: str, style: str, stem: str) -> None:
    font = TTFont(str(src))
    if font.sfntVersion != "OTTO" or "CFF " not in font:
        raise ValueError(f"{src} is not an OpenType/CFF font")
    order = font.getGlyphOrder()
    font["loca"] = newTable("loca")
    glyf = font["glyf"] = newTable("glyf")
    glyf.glyphOrder = order
    glyf.glyphs = _quadratic_glyphs(font.getGlyphSet())
    del font["CFF "]
    for table in ("VORG",):
        if table in font:
            del font[table]
    glyf.compile(font)
    # Left side bearings follow the new outlines.
    hmtx = font["hmtx"]
    for name, glyph in glyf.glyphs.items():
        width = hmtx[name][0]
        hmtx[name] = (width, getattr(glyph, "xMin", 0) if hasattr(glyph, "xMin") else 0)
    maxp = font["maxp"] = newTable("maxp")
    maxp.tableVersion = 0x00010000
    for attr in ("maxZones", "maxTwilightPoints", "maxStorage", "maxFunctionDefs", "maxInstructionDefs",
                 "maxStackElements", "maxSizeOfInstructions", "maxComponentElements"):
        setattr(maxp, attr, 0)
    maxp.maxZones = 1
    post = font["post"]
    post.formatType = 2.0
    post.extraNames, post.mapping, post.glyphOrder = [], {}, order
    font["head"].glyphDataFormat = 0
    font.sfntVersion = "\x00\x01\x00\x00"
    # Rename (GUST Font License: modified versions use new names).
    name = font["name"]
    full = f"{family} {style}".replace(" Regular", "")
    postscript = stem
    for record in list(name.names):
        if record.nameID in (1, 16):
            record.string = family
        elif record.nameID in (2, 17):
            record.string = style
        elif record.nameID == 4:
            record.string = full
        elif record.nameID == 6:
            record.string = postscript
        elif record.nameID == 3:
            record.string = f"{postscript};mathmage-ttf-build"
    font.save(str(dst))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"purpose": "TrueType builds of Latin Modern for MathMage figures (renamed per the GUST Font License)",
                "converter": "fontTools cu2qu (max_err 0.5 font units), layout tables preserved", "faces": {}}
    for src_name, stem, family, style in FACES:
        src = kpsewhich(src_name)
        dst = OUT / f"{stem}.ttf"
        otf_to_ttf(src, dst, family, style, stem)
        check = TTFont(str(dst))
        manifest["faces"][dst.name] = {
            "source": src_name, "source_sha256": sha256(src), "output_sha256": sha256(dst),
            "has_glyf": "glyf" in check, "has_GSUB": "GSUB" in check, "has_GPOS": "GPOS" in check,
            "has_MATH": "MATH" in check, "glyphs": len(check.getGlyphOrder())}
        print(f"{dst.name:30s} glyf={'glyf' in check} GSUB={'GSUB' in check} GPOS={'GPOS' in check} "
              f"MATH={'MATH' in check}")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    (OUT / "NOTICE.txt").write_text(
        "These TrueType files are mechanical conversions of the Latin Modern fonts\n"
        "(Copyright GUST e-foundry, distributed under the GUST Font License,\n"
        "http://www.gust.org.pl/fonts/licenses/GUST-FONT-LICENSE.txt). They were renamed\n"
        "as the license requests for modified versions. They are regenerated locally by\n"
        "scripts/build_fonts.py and are not redistributed with this repository.\n", encoding="utf-8")


if __name__ == "__main__":
    main()
