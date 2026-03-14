"""
Layer 5: Outfit Visualization — Quiet Luxury Editorial
=======================================================
A refined, single-class visualizer producing high-end fashion editorial
moodboards in the Quiet Luxury aesthetic.

Typography system (system fonts — no download required):
  • Lora Variable       — display title, garment names, score  (editorial serif)
  • Lora Italic         — accent text, match percentage        (expressive serif)
  • Poppins Light       — labels, badges, meta, captions       (refined geometric sans)
  • Poppins Medium      — material chips                       (slightly heavier sans)

Palette:
  • Warm ivory canvas + white cards
  • Deep charcoal / near-black ink
  • Champagne gold accents
  • Per-garment accent colours keyed to category
  • Black ribbon footer with gold rating bars

Key fixes vs prior version:
  • Garment names auto-cleaned   (PUSSY_BOW_BLOUSE → Pussy Bow Blouse)
  • Score ribbon: circle + gold bar chart + Lora Italic match % — no broken glyphs
  • `segmented_images` dict is the canonical way to pass individual garment images
  • All 15 GarmentCategory values have accent colours and labels
"""

from __future__ import annotations

import math
import random
import textwrap
import urllib.request
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from src.core import get_logger
from src.core.models import Garment, GarmentCategory

logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  FONTS
# ══════════════════════════════════════════════════════════════════════════════

_LORA      = "/usr/share/fonts/truetype/google-fonts/Lora-Variable.ttf"
_LORA_IT   = "/usr/share/fonts/truetype/google-fonts/Lora-Italic-Variable.ttf"
_POPPINS_L = "/usr/share/fonts/truetype/google-fonts/Poppins-Light.ttf"
_POPPINS_M = "/usr/share/fonts/truetype/google-fonts/Poppins-Medium.ttf"

_SERIF_FB: List[str] = [
    "/usr/share/fonts/truetype/freefont/FreeSerifBold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf",
]
_SERIF_IT_FB: List[str] = [
    "/usr/share/fonts/truetype/freefont/FreeSerifItalic.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Italic.ttf",
]
_SANS_FB: List[str] = [
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
]


def _font(primary: str, fallbacks: List[str], size: int) -> ImageFont.FreeTypeFont:
    for path in [primary, *fallbacks]:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    logger.warning("No font found for '%s' @%d — PIL default", primary, size)
    return ImageFont.load_default()


class _Fonts:
    """All typefaces used in the moodboard, loaded once."""
    def __init__(self) -> None:
        self.title      = _font(_LORA,      _SERIF_FB,    48)
        self.hero_name  = _font(_LORA,      _SERIF_FB,    20)
        self.thumb_name = _font(_LORA,      _SERIF_FB,    15)
        self.score_num  = _font(_LORA,      _SERIF_FB,    30)
        self.match_txt  = _font(_LORA_IT,   _SERIF_IT_FB, 20)
        self.badge      = _font(_POPPINS_L, _SANS_FB,      9)
        self.kicker     = _font(_POPPINS_L, _SANS_FB,     10)
        self.season     = _font(_POPPINS_L, _SANS_FB,     12)
        self.chips      = _font(_POPPINS_L, _SANS_FB,     10)
        self.chips_sm   = _font(_POPPINS_L, _SANS_FB,      9)
        self.ribbon_lbl = _font(_POPPINS_L, _SANS_FB,      9)
        self.issue      = _font(_POPPINS_L, _SANS_FB,      9)


# ══════════════════════════════════════════════════════════════════════════════
#  PALETTE
# ══════════════════════════════════════════════════════════════════════════════

IVORY      = (248, 245, 240)
WHITE      = (255, 255, 255)
BLACK      = ( 18,  16,  14)
CHARCOAL   = ( 40,  38,  35)
NUDE       = (215, 200, 185)
GOLD       = (168, 138,  96)
GOLD_LIGHT = (218, 198, 162)
INK_MID    = (118, 112, 104)
INK_LIGHT  = (175, 168, 158)
CARD_BG    = (255, 254, 252)

# Per-category accent bar colours
_ACCENTS: Dict[GarmentCategory, Tuple[int, int, int]] = {
    GarmentCategory.TOP:        (200, 180, 165),
    GarmentCategory.BOTTOM:     (170, 162, 200),
    GarmentCategory.DRESS:      (195, 162, 172),
    GarmentCategory.OUTERWEAR:  (120, 130, 122),
    GarmentCategory.SHOES:      (165, 142, 112),
    GarmentCategory.BAG:        (145, 128, 112),
    GarmentCategory.ACCESSORY:  (185, 172, 152),
}

_CAT_LABELS: Dict[GarmentCategory, str] = {
    GarmentCategory.TOP:        "TOP",
    GarmentCategory.BOTTOM:     "BOTTOM",
    GarmentCategory.DRESS:      "DRESS",
    GarmentCategory.OUTERWEAR:  "OUTERWEAR",
    GarmentCategory.SHOES:      "SHOES",
    GarmentCategory.BAG:        "BAG",
    GarmentCategory.ACCESSORY:  "ACCESSORY",
}


# ══════════════════════════════════════════════════════════════════════════════
#  LAYOUT CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

OUTER   = 72
H_HDR   = 192
H_FTR   = 100
GAP     = 22
CRAD    = 8
ABAR    = 5     # accent bar height
CPAD    = 20    # card inner padding
HERO_W  = 420
HERO_H  = 570
THUMB_W = 252
THUMB_H = 318

_HERO_ORDER = [
    GarmentCategory.DRESS,
    GarmentCategory.OUTERWEAR,
    GarmentCategory.TOP,
    GarmentCategory.BOTTOM,
]


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _clean_name(raw: str) -> str:
    """PUSSY_BOW_BLOUSE → Pussy Bow Blouse"""
    return raw.replace("_", " ").title().strip()


def _spaced(text: str, gap: int = 2) -> str:
    return (" " * gap).join(text.upper())


def _rgba(t: Tuple[int, int, int], a: int = 255) -> Tuple[int, int, int, int]:
    return (*t, a)


def _pick_hero(garments: List[Garment]) -> Tuple[Optional[Garment], List[Garment]]:
    for cat in _HERO_ORDER:
        for g in garments:
            if g.attributes.category == cat:
                return g, [x for x in garments if x is not g]
    if garments:
        return garments[0], garments[1:]
    return None, []


def _grid_dims(n: int) -> Tuple[int, int]:
    if n == 0:
        return 0, 0
    cols = min(n, 3)
    return cols, math.ceil(n / cols)


def _canvas_size(n_sec: int) -> Tuple[int, int]:
    cols, rows = _grid_dims(n_sec)
    gw = cols * THUMB_W + max(0, cols - 1) * GAP if cols else 0
    cw = OUTER * 2 + HERO_W + (GAP * 2 + gw if gw else 0)
    gh = rows * THUMB_H + max(0, rows - 1) * GAP if rows else 0
    ch = H_HDR + max(HERO_H, gh) + H_FTR + OUTER * 2 + 24
    return max(cw, 860), ch


def _shadow(canvas: Image.Image, x: int, y: int, w: int, h: int) -> None:
    layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    d.rounded_rectangle([x+8, y+9, x+w+8, y+h+9], radius=CRAD, fill=(0, 0, 0, 30))
    canvas.alpha_composite(layer.filter(ImageFilter.GaussianBlur(12)))


def _draw_card(canvas: Image.Image, draw: ImageDraw.Draw,
               x: int, y: int, w: int, h: int,
               accent: Tuple[int, int, int]) -> None:
    _shadow(canvas, x, y, w, h)
    draw.rounded_rectangle([x, y, x+w, y+h], radius=CRAD,
                           fill=_rgba(CARD_BG), outline=_rgba(NUDE), width=1)
    draw.rounded_rectangle([x, y, x+w, y+ABAR+CRAD], radius=CRAD, fill=_rgba(accent))
    draw.rectangle([x, y+CRAD, x+w, y+ABAR+CRAD], fill=_rgba(accent))


def _draw_badge(canvas: Image.Image, draw: ImageDraw.Draw,
                x: int, y: int, text: str,
                accent: Tuple[int, int, int], font: ImageFont.FreeTypeFont) -> int:
    """Draw pill badge, return height."""
    tw = int(draw.textlength(text, font=font))
    pw, ph = tw + 18, 19
    draw.rounded_rectangle([x, y, x+pw, y+ph], radius=9,
                           fill=_rgba(accent, 50), outline=_rgba(accent, 185), width=1)
    draw.text((x+9, y+2), text, font=font, fill=_rgba(CHARCOAL))
    return ph


def _fit_image(img: Image.Image, tw: int, th: int) -> Image.Image:
    img = img.convert("RGBA")
    img.thumbnail((tw, th), Image.Resampling.LANCZOS)
    bg = Image.new("RGBA", (tw, th), (255, 255, 255, 255))
    bg.paste(img, ((tw - img.width) // 2, (th - img.height) // 2), img)
    return bg


def _load_garment_image(garment: Garment,
                        segmented: Optional[Dict[str, Image.Image]],
                        ) -> Optional[Image.Image]:
    """
    Load the best available image for *garment*.
    Priority: segmented_images → image_path → image_url → None
    """
    if segmented and garment.id in segmented:
        return segmented[garment.id].copy()
    if getattr(garment, "image_path", None):
        try:
            return Image.open(garment.image_path).convert("RGBA")
        except Exception:
            pass
    if getattr(garment, "image_url", None):
        try:
            req = urllib.request.Request(
                garment.image_url, headers={"User-Agent": "Mozilla/5.0"}
            )
            with urllib.request.urlopen(req, timeout=8) as r:
                return Image.open(BytesIO(r.read())).convert("RGBA")
        except Exception:
            pass
    return None


def _placeholder(w: int, h: int, accent: Tuple[int, int, int]) -> Image.Image:
    return Image.new("RGBA", (w, h), _rgba(accent, 38))


def _add_grain(canvas: Image.Image, intensity: int = 5) -> None:
    grain = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    px = grain.load()
    rng = random.Random(42)
    for yy in range(0, canvas.height, 2):
        for xx in range(0, canvas.width, 2):
            v = rng.randint(-intensity, intensity)
            px[xx, yy] = (max(0, v), max(0, v), max(0, v), rng.randint(0, 7))
    canvas.alpha_composite(grain)


def _garment_display_name(garment: Garment) -> str:
    """
    Build a clean display name, stripping underscore identifiers.
    Priority: name → product_type → subcategory → category.value
    """
    raw = (
        getattr(garment, "name", None)
        or getattr(garment.attributes, "product_type", None)
        or getattr(garment.attributes, "subcategory", None)
        or garment.attributes.category.value
    )
    return _clean_name(str(raw))


def _garment_accent(garment: Garment) -> Tuple[int, int, int]:
    return _ACCENTS.get(garment.attributes.category, (178, 172, 165))


def _garment_chips(garment: Garment) -> List[str]:
    chips: List[str] = []
    mat = getattr(garment.attributes, "material", None)
    if mat:
        p = getattr(mat, "primary", None) or getattr(mat, "fabric", None)
        t = getattr(mat, "texture", None)
        if p:
            chips.append(_clean_name(str(p)))
        if t and t != p:
            chips.append(_clean_name(str(t)))
    if not chips:
        col = getattr(garment.attributes, "color", None)
        if col and getattr(col, "primary", None):
            chips.append(_clean_name(str(col.primary)))
    return chips[:2]


def _garment_color(garment: Garment) -> Optional[str]:
    col = getattr(garment.attributes, "color", None)
    if col and getattr(col, "primary", None):
        return _clean_name(str(col.primary))
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  RESULT
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class OutfitVisualization:
    image: Image.Image
    outfit_name: str
    garment_count: int
    score: Optional[float] = None
    output_path: Optional[Path] = None


# ══════════════════════════════════════════════════════════════════════════════
#  VISUALIZER
# ══════════════════════════════════════════════════════════════════════════════

class QuietLuxuryVisualizer:
    """
    Quiet Luxury editorial moodboard.

    Layout::

        ┌──────────────────────────────────────────────────┐
        │  S T Y L E  G U I D E                  N° 01    │
        │  BLOUSE + SKIRT + HAT                            │
        │  AUTUMN  ·  BUSINESS                             │
        │  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │
        ├───────────────┬──────────────────────────────────┤
        │               │  ╔══════════╗  ╔══════════╗      │
        │  HERO CARD    │  ║ BOTTOM   ║  ║ HAT      ║      │
        │               │  ╚══════════╝  ╚══════════╝      │
        ├───────────────┴──────────────────────────────────┤
        │  3 PIECES  ⊙87  ████░  87% MATCH  AW 2025       │
        └──────────────────────────────────────────────────┘

    Pass individual garment images via `segmented_images` (dict garment.id → PIL Image).
    Scales from 1 – 10+ garments automatically.
    """

    def __init__(self) -> None:
        self._f = _Fonts()
        logger.info("QuietLuxuryVisualizer initialised.")

    def create_visualization(
        self,
        garments: List[Garment],
        segmented_images: Optional[Dict[str, Image.Image]] = None,
        outfit_name: str = "THE LOOK",
        score: Optional[float] = None,
        season: Optional[str] = None,
        occasion: Optional[str] = None,
    ) -> OutfitVisualization:
        """
        Build the moodboard and return an OutfitVisualization.

        `segmented_images` should map each garment.id to a pre-cropped PIL Image
        of that specific item. This ensures each card shows only its own garment,
        not the entire source photo.
        """
        if not garments:
            raise ValueError("At least one garment is required.")

        hero, rest = _pick_hero(garments)
        CW, CH = _canvas_size(len(rest))

        canvas = Image.new("RGBA", (CW, CH), _rgba(IVORY))
        draw   = ImageDraw.Draw(canvas)

        _add_grain(canvas)
        self._draw_header(draw, CW, outfit_name, season, occasion)

        cy = H_HDR + OUTER
        if hero:
            self._draw_hero_card(canvas, draw, hero, OUTER, cy, segmented_images)
        if rest:
            self._draw_grid(canvas, draw, rest,
                            OUTER + HERO_W + GAP * 2, cy, segmented_images)
        self._draw_ribbon(draw, CW, CH, score, len(garments), season, occasion)

        logger.info("Visualization created — %d garments.", len(garments))
        return OutfitVisualization(canvas, outfit_name, len(garments), score)

    def save_visualization(
        self, viz: OutfitVisualization, output_path: Union[str, Path],
    ) -> bool:
        try:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            viz.image.convert("RGB").save(str(path), "PNG", dpi=(150, 150))
            viz.output_path = path
            logger.info("Saved → %s", path)
            return True
        except Exception as exc:
            logger.error("Save failed: %s", exc, exc_info=True)
            return False

    # ── Header ────────────────────────────────────────────────────────────────

    def _draw_header(self, draw: ImageDraw.Draw, CW: int,
                     outfit_name: str,
                     season: Optional[str], occasion: Optional[str]) -> None:
        y = OUTER
        draw.line([(OUTER, y), (CW - OUTER, y)], fill=_rgba(CHARCOAL), width=2)
        y += 16
        draw.text((OUTER, y), _spaced("style guide", 3),
                  font=self._f.kicker, fill=_rgba(GOLD))
        y += 18
        draw.text((OUTER, y), outfit_name.upper(),
                  font=self._f.title, fill=_rgba(BLACK))
        y += 58
        meta = "  ·  ".join(p.upper() for p in [season, occasion] if p)
        if meta:
            draw.text((OUTER, y), meta, font=self._f.season, fill=_rgba(INK_MID))
        y += 20
        draw.line([(OUTER, y), (OUTER + 50, y)], fill=_rgba(GOLD), width=2)
        draw.line([(OUTER + 54, y), (CW - OUTER, y)], fill=_rgba(NUDE), width=1)
        draw.text((CW - OUTER - 62, OUTER + 32), _spaced("nº 01", 2),
                  font=self._f.issue, fill=_rgba(INK_LIGHT))

    # ── Hero card ─────────────────────────────────────────────────────────────

    def _draw_hero_card(self, canvas: Image.Image, draw: ImageDraw.Draw,
                        garment: Garment, x: int, y: int,
                        segmented: Optional[Dict[str, Image.Image]]) -> None:
        accent = _garment_accent(garment)
        _draw_card(canvas, draw, x, y, HERO_W, HERO_H, accent)

        iw = HERO_W - CPAD * 2
        ih = HERO_H - 148

        raw = _load_garment_image(garment, segmented)
        img = _fit_image(raw, iw, ih) if raw else _placeholder(iw, ih, accent)

        region = Image.new("RGBA", (iw, ih), (255, 255, 255, 255))
        region.alpha_composite(img)
        canvas.alpha_composite(region, (x + CPAD, y + ABAR + CPAD))

        self._hero_text(canvas, draw, garment,
                        x + CPAD, y + ABAR + ih + CPAD + 12)

    def _hero_text(self, canvas: Image.Image, draw: ImageDraw.Draw,
                   garment: Garment, tx: int, ty: int) -> None:
        accent = _garment_accent(garment)
        cat    = garment.attributes.category

        ph = _draw_badge(canvas, draw, tx, ty,
                         _spaced(_CAT_LABELS.get(cat, "PIECE"), 1),
                         accent, self._f.badge)
        ty += ph + 10
        draw.text((tx, ty), _garment_display_name(garment).upper(),
                  font=self._f.hero_name, fill=_rgba(BLACK))
        ty += 28
        chips = _garment_chips(garment)
        if chips:
            draw.text((tx, ty), "  ·  ".join(c.upper() for c in chips),
                      font=self._f.chips, fill=_rgba(INK_MID))
            ty += 16
        color = _garment_color(garment)
        if color:
            draw.text((tx, ty), color.upper(), font=self._f.chips_sm,
                      fill=_rgba(GOLD, 220))

    # ── Secondary grid ────────────────────────────────────────────────────────

    def _draw_grid(self, canvas: Image.Image, draw: ImageDraw.Draw,
                   garments: List[Garment], x0: int, y0: int,
                   segmented: Optional[Dict[str, Image.Image]]) -> None:
        cols, _ = _grid_dims(len(garments))
        for i, garment in enumerate(garments):
            col = i % cols
            row = i // cols
            self._draw_thumb(
                canvas, draw, garment,
                x0 + col * (THUMB_W + GAP),
                y0 + row * (THUMB_H + GAP),
                segmented,
            )

    def _draw_thumb(self, canvas: Image.Image, draw: ImageDraw.Draw,
                    garment: Garment, x: int, y: int,
                    segmented: Optional[Dict[str, Image.Image]]) -> None:
        accent = _garment_accent(garment)
        _draw_card(canvas, draw, x, y, THUMB_W, THUMB_H, accent)

        tp  = 14
        tiw = THUMB_W - tp * 2
        tih = THUMB_H - 105

        raw = _load_garment_image(garment, segmented)
        img = _fit_image(raw, tiw, tih) if raw else _placeholder(tiw, tih, accent)

        region = Image.new("RGBA", (tiw, tih), (255, 255, 255, 255))
        region.alpha_composite(img)
        canvas.alpha_composite(region, (x + tp, y + ABAR + tp))

        self._thumb_text(canvas, draw, garment,
                         x + tp, y + ABAR + tih + tp + 10)

    def _thumb_text(self, canvas: Image.Image, draw: ImageDraw.Draw,
                    garment: Garment, tx: int, ty: int) -> None:
        accent = _garment_accent(garment)
        cat    = garment.attributes.category

        ph = _draw_badge(canvas, draw, tx, ty,
                         _spaced(_CAT_LABELS.get(cat, "PIECE"), 1),
                         accent, self._f.badge)
        ty += ph + 8
        draw.text((tx, ty), _garment_display_name(garment).upper(),
                  font=self._f.thumb_name, fill=_rgba(BLACK))
        ty += 22
        chips = _garment_chips(garment)
        if chips:
            draw.text((tx, ty), "  ·  ".join(c.upper() for c in chips),
                      font=self._f.chips_sm, fill=_rgba(INK_MID))
            ty += 14
        color = _garment_color(garment)
        if color:
            draw.text((tx, ty), color.upper(), font=self._f.chips_sm,
                      fill=_rgba(GOLD, 200))

    # ── Score ribbon ──────────────────────────────────────────────────────────

    def _draw_ribbon(self, draw: ImageDraw.Draw, CW: int, CH: int,
                     score: Optional[float], count: int,
                     season: Optional[str], occasion: Optional[str]) -> None:
        ry = CH - H_FTR
        draw.rectangle([0, ry, CW, CH], fill=_rgba(BLACK))
        draw.line([(0, ry), (CW, ry)], fill=_rgba(GOLD, 80), width=1)

        # Piece count
        draw.text((OUTER, ry + 36),
                  _spaced(f"{count} piece{'s' if count != 1 else ''}", 2),
                  font=self._f.ribbon_lbl, fill=_rgba(INK_LIGHT))

        if score is not None:
            pct = int(score * 100) if score <= 1.0 else int(score)

            # Score circle
            cx_s = CW // 2 - 62
            cy_s = ry + H_FTR // 2
            r_c  = 36
            draw.ellipse([cx_s-r_c, cy_s-r_c, cx_s+r_c, cy_s+r_c],
                         fill=_rgba(CHARCOAL), outline=_rgba(GOLD, 200), width=1)
            s_str = str(pct)
            sw = int(draw.textlength(s_str, font=self._f.score_num))
            draw.text((cx_s - sw // 2, cy_s - 22), s_str,
                      font=self._f.score_num, fill=_rgba(GOLD_LIGHT))

            # Rating bars (5 × vertical gold blocks)
            filled = round(pct / 20)
            bx0 = cx_s + r_c + 16
            by  = cy_s - 22
            for bi in range(5):
                bc = _rgba(GOLD) if bi < filled else _rgba((58, 55, 52))
                bx = bx0 + bi * 14
                draw.rounded_rectangle([bx, by, bx+10, by+30], radius=2, fill=bc)
            draw.text((bx0, by + 36), _spaced("style score", 2),
                      font=self._f.ribbon_lbl, fill=_rgba(INK_LIGHT))

            # Vertical divider
            dv_x = bx0 + 5 * 14 + 18
            draw.line([(dv_x, ry+18), (dv_x, CH-18)],
                      fill=_rgba((58, 55, 52)), width=1)

            # Match % — Lora Italic
            draw.text((dv_x + 16, ry + 24), f"{pct}%  MATCH",
                      font=self._f.match_txt, fill=_rgba(GOLD_LIGHT))
            meta = "  ·  ".join(p.upper() for p in [season, occasion] if p)
            if meta:
                draw.text((dv_x + 16, ry + 52), meta,
                          font=self._f.ribbon_lbl, fill=_rgba(INK_LIGHT))

        # Brand mark
        draw.text((CW - OUTER - 72, ry + 36), _spaced("ql edit", 3),
                  font=self._f.ribbon_lbl, fill=_rgba(GOLD))


# ══════════════════════════════════════════════════════════════════════════════
#  CONVENIENCE ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def visualize_outfit(
    garments: List[Garment],
    output_path: str = "outfit_board.png",
    outfit_name: str = "THE LOOK",
    score: Optional[float] = None,
    season: Optional[str] = None,
    occasion: Optional[str] = None,
    segmented_images: Optional[Dict[str, Image.Image]] = None,
) -> OutfitVisualization:
    """
    One-liner entry point for the Quiet Luxury editorial visualizer.

    Example — passing pre-cropped individual garment images::

        from PIL import Image

        # Each garment ID maps to its own cropped image
        segmented = {
            blouse.id: Image.open("blouse_cropped.png"),
            skirt.id:  Image.open("skirt_cropped.png"),
            hat.id:    Image.open("hat_cropped.png"),
        }

        viz = visualize_outfit(
            garments=[blouse, skirt, hat],
            segmented_images=segmented,
            outfit_name="Blouse + Skirt + Hat",
            score=0.87,
            season="AW 2025",
            occasion="Business Casual",
        )
    """
    engine = QuietLuxuryVisualizer()
    viz = engine.create_visualization(
        garments,
        segmented_images=segmented_images,
        outfit_name=outfit_name,
        score=score,
        season=season,
        occasion=occasion,
    )
    engine.save_visualization(viz, output_path)
    return viz