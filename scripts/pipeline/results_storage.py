"""
Results Storage Module
======================

Handles saving pipeline results (garments, combinations, graphs) to disk.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

# Type hints for external classes
from src.core.models import Garment


class ResultsStorage:
    """
    Stores pipeline results for analysis and debugging.
    
    Creates a timestamped directory structure:
        run_YYYYMMDD_HHMMSS/
        ├── garments/      - Individual garment JSON files
        ├── combinations/  - Outfit scores and rankings
        ├── graphs/        - Visual score comparisons
        └── final_report.json
    """
    
    def __init__(self, output_dir: Path):
        """
        Initialize results storage.
        
        Args:
            output_dir: Directory to save results
        """
        self.output_dir = Path(output_dir)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / f"run_{self.timestamp}"
        
        # Create directories
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "garments").mkdir(exist_ok=True)
        (self.run_dir / "combinations").mkdir(exist_ok=True)
        (self.run_dir / "graphs").mkdir(exist_ok=True)
        
        print(f"\n💾 Results will be saved to: {self.run_dir}")
    
    def save_garment(self, garment: Garment, index: int) -> Path:
        """Save a single garment's attributes to JSON."""
        attrs = garment.attributes
        
        garment_data = {
            "id": garment.id,
            "image_path": str(garment.image_path) if garment.image_path else None,
            "image_name": Path(garment.image_path).name if garment.image_path else None,
            "extracted_at": datetime.now().isoformat(),
            "attributes": {
                "category": attrs.category.value,
                "subcategory": attrs.subcategory,
                "product_type": attrs.product_type,
                "outfit_role": attrs.outfit_role.value if attrs.outfit_role else None,
                "color": self._serialize_color(attrs.color),
                "pattern": self._serialize_pattern(attrs.pattern),
                "material": self._serialize_material(attrs.material),
                "silhouette": self._serialize_silhouette(attrs),
                "formality_level": attrs.formality_level.value if attrs.formality_level else None,
                "style_tags": attrs.style_tags if attrs.style_tags else [],
                "seasons": [s.value for s in attrs.season_suitable] if attrs.season_suitable else [],
            }
        }
        
        # Generate filename
        img_name = Path(garment.image_path).stem if garment.image_path else f"garment_{index}"
        filename = f"{index:02d}_{attrs.category.value}_{img_name}.json"
        filepath = self.run_dir / "garments" / filename
        
        with open(filepath, "w") as f:
            json.dump(garment_data, f, indent=2, default=str)
        
        return filepath
    
    def _serialize_color(self, color) -> Optional[Dict]:
        """Serialize color attributes to dict."""
        if not color:
            return None
        return {
            "primary": color.primary if color else None,
            "secondary": color.secondary if color else None,
            "color_palette": color.color_palette if color else [],
            "hex_codes": color.hex_codes if color else [],
            "color_temperature": color.color_temperature.value if color.color_temperature else None,
            "color_depth": color.color_depth.value if color.color_depth else None,
        }
    
    def _serialize_pattern(self, pattern) -> Optional[Dict]:
        """Serialize pattern attributes to dict."""
        if not pattern:
            return None
        return {
            "type": pattern.type if pattern else None,
            "scale": pattern.scale if pattern else None,
        }
    
    def _serialize_material(self, material) -> Optional[Dict]:
        """Serialize material attributes to dict."""
        if not material:
            return None
        return {
            "primary": material.primary if material else None,
            "texture": material.texture if material else None,
            "fabric_weight": material.fabric_weight.value if material.fabric_weight else None,
        }
    
    def _serialize_silhouette(self, attrs) -> Optional[Dict]:
        """Serialize silhouette attributes to dict."""
        if not attrs.silhouette_profile and not attrs.fit:
            return None
        return {
            "fit": attrs.silhouette_profile.fit if attrs.silhouette_profile else attrs.fit,
            "structure": attrs.silhouette_profile.structure.value if attrs.silhouette_profile and attrs.silhouette_profile.structure else None,
            "volume": attrs.silhouette_profile.volume.value if attrs.silhouette_profile and attrs.silhouette_profile.volume else None,
        }
    
    def save_all_garments(self, garments: List[Garment]) -> List[Path]:
        """Save all garments to individual JSON files."""
        saved_files = []
        for i, garment in enumerate(garments):
            filepath = self.save_garment(garment, i)
            saved_files.append(filepath)
        
        # Also save a summary
        summary = {
            "total_garments": len(garments),
            "by_category": {},
            "extraction_timestamp": datetime.now().isoformat(),
            "garment_files": [str(f.name) for f in saved_files]
        }
        
        for g in garments:
            cat = g.attributes.category.value
            summary["by_category"][cat] = summary["by_category"].get(cat, 0) + 1
        
        summary_path = self.run_dir / "garments" / "_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"   📄 Saved {len(garments)} garment files to {self.run_dir / 'garments'}")
        return saved_files
    
    def save_combinations(self, candidates: list, profile: str) -> Path:
        """Save all outfit combinations with their scores."""
        sorted_candidates = sorted(candidates, key=lambda c: c.overall_score, reverse=True)
        
        combinations_data = {
            "profile": profile,
            "total_combinations": len(candidates),
            "evaluated_at": datetime.now().isoformat(),
            "combinations": []
        }
        
        for rank, candidate in enumerate(sorted_candidates, 1):
            combo = {
                "rank": rank,
                "name": candidate.name,
                "overall_score": round(candidate.overall_score, 4),
                "grade": candidate.scorecard.get_grade() if hasattr(candidate, 'scorecard') else None,
                "garments": [
                    {
                        "category": g.attributes.category.value,
                        "subcategory": g.attributes.subcategory,
                        "color": g.attributes.color.primary if g.attributes.color else "unknown",
                        "image": Path(g.image_path).name if g.image_path else None
                    }
                    for g in candidate.garments
                ],
                "scores": {}
            }
            
            # Add individual criterion scores
            if hasattr(candidate, 'scorecard'):
                scores = candidate.scorecard.get_filtered_scores()
                combo["scores"] = {k: round(v, 4) for k, v in scores.items()}
            
            combinations_data["combinations"].append(combo)
        
        # Save JSON
        filepath = self.run_dir / "combinations" / f"all_combinations_{profile}.json"
        with open(filepath, "w") as f:
            json.dump(combinations_data, f, indent=2)
        
        print(f"   📄 Saved {len(candidates)} combinations to {filepath.name}")
        return filepath
    
    def save_final_report(self, report: dict, best_outfit, candidates: list) -> Path:
        """Save comprehensive final report."""
        best_outfit_data = report.get("best_outfit") or {}
        final_report = {
            "run_timestamp": self.timestamp,
            "run_directory": str(self.run_dir),
            "summary": {
                "total_garments_analyzed": report.get("total_combinations", 0),
                "total_combinations": len(candidates) if candidates else 0,
                "best_score": best_outfit_data.get("overall_score", 0),
                "profile_used": report.get("profile_used", "default")
            },
            "best_outfit": best_outfit_data,
            "score_distribution": {
                "excellent": sum(1 for c in candidates if c.overall_score >= 0.8),
                "good": sum(1 for c in candidates if 0.6 <= c.overall_score < 0.8),
                "needs_work": sum(1 for c in candidates if c.overall_score < 0.6)
            } if candidates else {},
            "files_generated": {
                "garments": str(self.run_dir / "garments"),
                "combinations": str(self.run_dir / "combinations"),
                "graphs": str(self.run_dir / "graphs")
            }
        }
        
        filepath = self.run_dir / "final_report.json"
        with open(filepath, "w") as f:
            json.dump(final_report, f, indent=2, default=str)
        
        print(f"   📄 Saved final report to {filepath.name}")
        return filepath
