"""
Local Classifier  (Solution 4 — Lightweight Local Pre-classification)
======================================================================

Fast, zero-API garment **category pre-classifier** that runs entirely on
the local CPU using scikit-learn / basic image statistics.

Purpose
-------
Before sending an image to Gemini (expensive, ~10 s) this module quickly
guesses the garment category (top / bottom / shoes / outerwear / dress /
accessory) so that the Gemini prompt can be *targeted* rather than generic.

A targeted prompt is:
  - Shorter  → fewer output tokens → faster response
  - More accurate  → fewer hallucinations
  - Cheaper  → fewer API credits consumed

Implementation strategy
-----------------------
We use a two-stage heuristic approach that requires **no model weights**:

1. **Filename heuristics** — keywords in the filename/path
   (``shirt`` → TOP, ``jeans`` → BOTTOM, ``boots`` → SHOES, …)

2. **Colour-histogram aspect-ratio heuristic** — very fast PIL computation:
   - Tall narrow images  (aspect < 0.6)   → likely BOTTOM / DRESS
   - Short wide images   (aspect > 1.4)   → likely SHOES / ACCESSORY
   - Square-ish images                    → likely TOP / OUTERWEAR

Both stages are combined with confidence scores.  The output is always a
``ClassificationResult`` with a ``category``, ``confidence``, and a flag
``needs_gemini_confirmation`` that the caller uses to decide whether a
targeted or generic Gemini prompt is sent.

If ``confidence >= 0.80``  → use targeted prompt (Gemini focuses only on
                              the guessed category for the fine details)
If ``confidence < 0.80``   → use the default generic prompt

This is intentionally lightweight and dependency-free (only PIL + stdlib).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.core import get_logger
from src.core.models import GarmentCategory

try:
    from PIL import Image
except ImportError:  # PIL optional — aspect-ratio stage degrades gracefully
    Image = None  # type: ignore[assignment]

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Keyword mappings  (filename → category)
# ---------------------------------------------------------------------------
_KEYWORD_MAP: dict[str, GarmentCategory] = {
    # Tops
    "shirt": GarmentCategory.TOP,
    "blouse": GarmentCategory.TOP,
    "tshirt": GarmentCategory.TOP,
    "t-shirt": GarmentCategory.TOP,
    "top": GarmentCategory.TOP,
    "tank": GarmentCategory.TOP,
    "sweater": GarmentCategory.TOP,
    "pullover": GarmentCategory.TOP,
    "hoodie": GarmentCategory.TOP,
    "cardigan": GarmentCategory.TOP,
    "crop": GarmentCategory.TOP,
    "tee": GarmentCategory.TOP,
    "polo": GarmentCategory.TOP,
    "tunic": GarmentCategory.TOP,
    "knit": GarmentCategory.TOP,
    # Bottoms
    "pants": GarmentCategory.BOTTOM,
    "trouser": GarmentCategory.BOTTOM,
    "jeans": GarmentCategory.BOTTOM,
    "denim": GarmentCategory.BOTTOM,
    "legging": GarmentCategory.BOTTOM,
    "shorts": GarmentCategory.BOTTOM,
    "skirt": GarmentCategory.BOTTOM,
    "culottes": GarmentCategory.BOTTOM,
    "chino": GarmentCategory.BOTTOM,
    "jogger": GarmentCategory.BOTTOM,
    "sweatpant": GarmentCategory.BOTTOM,
    # Dresses
    "dress": GarmentCategory.DRESS,
    "gown": GarmentCategory.DRESS,
    "jumpsuit": GarmentCategory.DRESS,
    "romper": GarmentCategory.DRESS,
    "overall": GarmentCategory.DRESS,
    "maxi": GarmentCategory.DRESS,
    "midi": GarmentCategory.DRESS,
    # Outerwear
    "jacket": GarmentCategory.OUTERWEAR,
    "coat": GarmentCategory.OUTERWEAR,
    "blazer": GarmentCategory.OUTERWEAR,
    "parka": GarmentCategory.OUTERWEAR,
    "trench": GarmentCategory.OUTERWEAR,
    "vest": GarmentCategory.OUTERWEAR,
    "windbreaker": GarmentCategory.OUTERWEAR,
    # Shoes
    "shoe": GarmentCategory.SHOES,
    "boot": GarmentCategory.SHOES,
    "sneaker": GarmentCategory.SHOES,
    "sandal": GarmentCategory.SHOES,
    "heel": GarmentCategory.SHOES,
    "loafer": GarmentCategory.SHOES,
    "pump": GarmentCategory.SHOES,
    "mule": GarmentCategory.SHOES,
    "flat": GarmentCategory.SHOES,
    "trainer": GarmentCategory.SHOES,
    # Accessories
    "bag": GarmentCategory.ACCESSORY,
    "hat": GarmentCategory.ACCESSORY,
    "scarf": GarmentCategory.ACCESSORY,
    "belt": GarmentCategory.ACCESSORY,
    "watch": GarmentCategory.ACCESSORY,
    "sunglasses": GarmentCategory.ACCESSORY,
    "jewel": GarmentCategory.ACCESSORY,
    "necklace": GarmentCategory.ACCESSORY,
    "earring": GarmentCategory.ACCESSORY,
    "purse": GarmentCategory.ACCESSORY,
    "wallet": GarmentCategory.ACCESSORY,
    # Folder-name aliases (used when filename has no keyword)
    "tops": GarmentCategory.TOP,
    "bottoms": GarmentCategory.BOTTOM,
    "dresses": GarmentCategory.DRESS,
    "outerwear": GarmentCategory.OUTERWEAR,
    "footwear": GarmentCategory.SHOES,
    "shoes": GarmentCategory.SHOES,
    "accessories": GarmentCategory.ACCESSORY,
}

# Aspect ratio thresholds
_ASPECT_TALL = 0.55   # width/height < this → likely BOTTOM or DRESS
_ASPECT_WIDE = 1.50   # width/height > this → likely SHOES or ACCESSORY

# Module-level alias so tests / callers can do:
#   from src.layer1_vision.local_classifier import CONFIDENCE_THRESHOLD
CONFIDENCE_THRESHOLD: float = 0.80


@dataclass
class ClassificationResult:
    """Result of local pre-classification."""
    category: GarmentCategory
    confidence: float           # 0.0 – 1.0
    source: str                 # "filename" | "aspect_ratio" | "fallback"
    needs_gemini_confirmation: bool = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        # Auto-compute from confidence when not explicitly provided
        if self.needs_gemini_confirmation is None:
            self.needs_gemini_confirmation = self.confidence < CONFIDENCE_THRESHOLD

    @property
    def targeted_hint(self) -> str:
        """
        Short string to inject into a Gemini prompt when confidence is high.
        e.g. "This image shows a BOTTOM garment (jeans/trousers/skirt)."
        """
        _hints = {
            GarmentCategory.TOP:       "This image shows a TOP garment (shirt/blouse/sweater/t-shirt).",
            GarmentCategory.BOTTOM:    "This image shows a BOTTOM garment (trousers/jeans/skirt/shorts).",
            GarmentCategory.DRESS:     "This image shows a DRESS or one-piece garment.",
            GarmentCategory.OUTERWEAR: "This image shows OUTERWEAR (jacket/coat/blazer).",
            GarmentCategory.SHOES:     "This image shows FOOTWEAR (shoes/boots/sneakers).",
            GarmentCategory.ACCESSORY: "This image shows an ACCESSORY (bag/hat/belt/jewellery).",
        }
        return _hints.get(self.category, "")


class LocalGarmentClassifier:
    """
    Fast local pre-classifier for garment images.

    Usage
    -----
        clf = LocalGarmentClassifier()
        result = clf.classify(Path("data/sample_wardrobe/image.png"))
        if result.confidence >= 0.80:
            prompt = targeted_prompt(result.targeted_hint) + base_prompt
        else:
            prompt = generic_prompt
    """

    CONFIDENCE_THRESHOLD = 0.80

    def classify(self, image_path: Path) -> ClassificationResult:
        """
        Classify a garment image without any API call.

        Parameters
        ----------
        image_path : Path
            Path to the garment image.

        Returns
        -------
        ClassificationResult
        """
        # Stage 1 — filename heuristics (highest confidence)
        result = self._classify_by_filename(image_path)
        if result is not None:
            return result

        # Stage 2 — image aspect-ratio heuristic
        result = self._classify_by_aspect_ratio(image_path)
        if result is not None:
            return result

        # Stage 3 — fallback: TOP is the most common garment category
        return ClassificationResult(
            category=GarmentCategory.TOP,
            confidence=0.30,
            needs_gemini_confirmation=True,
            source="fallback",
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _classify_by_filename(self, image_path: Path) -> Optional[ClassificationResult]:
        """Check filename/parent-dir for category keywords."""
        # Combine filename stem + immediate parent folder name
        text = f"{image_path.stem} {image_path.parent.name}".lower()
        # Normalise separators
        text = re.sub(r"[_\-.]", " ", text)

        for keyword, category in _KEYWORD_MAP.items():
            if keyword in text:
                logger.debug(f"LocalClassifier: '{image_path.name}' → {category.value} (keyword='{keyword}')")
                return ClassificationResult(
                    category=category,
                    confidence=0.85,
                    needs_gemini_confirmation=False,
                    source="filename",
                )
        return None

    def _classify_by_aspect_ratio(self, image_path: Path) -> Optional[ClassificationResult]:
        """Use PIL to read image dimensions and infer category from aspect ratio."""
        try:
            if Image is None:
                return None

            with Image.open(image_path) as img:
                w, h = img.size

            if h == 0:
                return None

            aspect = w / h

            if aspect < _ASPECT_TALL:
                # Tall narrow image → BOTTOM or DRESS
                category = GarmentCategory.BOTTOM
                confidence = 0.65
            elif aspect > _ASPECT_WIDE:
                # Wide image → SHOES or ACCESSORY
                category = GarmentCategory.SHOES
                confidence = 0.55
            else:
                # Roughly square → TOP
                category = GarmentCategory.TOP
                confidence = 0.50

            logger.debug(
                f"LocalClassifier: '{image_path.name}' aspect={aspect:.2f} "
                f"→ {category.value} (confidence={confidence:.2f})"
            )
            return ClassificationResult(
                category=category,
                confidence=confidence,
                needs_gemini_confirmation=confidence < self.CONFIDENCE_THRESHOLD,
                source="aspect_ratio",
            )
        except Exception as exc:
            logger.debug(f"LocalClassifier aspect-ratio failed for {image_path.name}: {exc}")
            return None
