"""
Mood Board Service — Orchestrator
===================================

Coordinates the DB layer, profile builder, scorer, and LLM explainer for
all mood board operations.

Public API:
    svc = MoodBoardService(db)

    # Sharing
    shared = svc.share_outfit(user_id, request)
    svc.like_outfit(user_id, shared_outfit_id)
    svc.unlike_outfit(user_id, shared_outfit_id)

    # Boards CRUD
    board = svc.create_board(user_id, name, description)
    boards = svc.list_boards(user_id)
    svc.activate_board(user_id, board_id)
    svc.delete_board(user_id, board_id)

    # Items
    item = svc.save_outfit_to_board(user_id, board_id, shared_outfit_id, note)
    svc.remove_item(user_id, board_id, item_id)
    items = svc.list_items(board_id)

    # Analysis
    profile = svc.get_style_profile(user_id, board_id)
    gap = await svc.get_gap_analysis(user_id, board_id, wardrobe_garments)
    recs = svc.get_moodboard_recommendations(user_id, board_id, candidate_outfits)
    summary = await svc.get_board_summary(user_id, board_id)

    # Discovery
    feed = svc.get_discovery_feed(filters, page, limit)
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from sqlalchemy.orm import Session

from src.core import get_logger
from src.database.models import (
    MoodBoard,
    MoodBoardItem,
    MoodBoardStyleProfile,
    SharedOutfit,
    SharedOutfitLike,
)
from src.layer2_style.moodboard_scorer import MoodBoardScorer
from src.layer3_context.moodboard_profile import MoodBoardProfileBuilder
from src.layer4_llm.moodboard_explainer import MoodBoardExplainer

logger = get_logger(__name__)


class MoodBoardService:
    """
    Orchestrates all mood board operations across DB, Layer2, Layer3 and Layer4.
    """

    def __init__(self, db: Session) -> None:
        self._db = db
        self._profile_builder = MoodBoardProfileBuilder(db)
        self._scorer = MoodBoardScorer()
        self._explainer = MoodBoardExplainer()

    # ------------------------------------------------------------------
    # Sharing
    # ------------------------------------------------------------------

    def share_outfit(self, user_id: str, data: Dict[str, Any]) -> SharedOutfit:
        """Create a publicly-shared outfit from the provided data snapshot."""
        shared = SharedOutfit(
            id=str(uuid4()),
            owner_user_id=user_id,
            outfit_data=data.get("outfit_data", {}),
            title=data.get("title"),
            occasion_tags=data.get("occasion_tags", []),
            style_tags=data.get("style_tags", []),
            formality_score=data.get("formality_score"),
            dominant_colors=data.get("dominant_colors", []),
            dominant_styles=data.get("dominant_styles", {}),
            embedding_vector=data.get("embedding_vector", []),
            is_public=data.get("is_public", True),
        )
        self._db.add(shared)
        self._db.commit()
        self._db.refresh(shared)
        logger.info(f"User {user_id}: shared outfit {shared.id}")
        return shared

    def unpublish_outfit(self, user_id: str, shared_outfit_id: str) -> bool:
        """Mark a shared outfit as private. Returns False if not found / not owner."""
        outfit = self._get_shared_outfit_or_raise(shared_outfit_id, user_id)
        outfit.is_public = False
        self._db.commit()
        return True

    def like_outfit(self, user_id: str, shared_outfit_id: str) -> int:
        """Like a shared outfit. Returns new likes_count (idempotent)."""
        self._get_shared_outfit_or_raise(shared_outfit_id)

        existing = (
            self._db.query(SharedOutfitLike)
            .filter(
                SharedOutfitLike.user_id == user_id,
                SharedOutfitLike.shared_outfit_id == shared_outfit_id,
            )
            .first()
        )
        if existing:
            outfit = self._db.query(SharedOutfit).filter(SharedOutfit.id == shared_outfit_id).first()
            return outfit.likes_count if outfit else 0

        like = SharedOutfitLike(
            id=str(uuid4()),
            user_id=user_id,
            shared_outfit_id=shared_outfit_id,
        )
        self._db.add(like)

        outfit = self._db.query(SharedOutfit).filter(SharedOutfit.id == shared_outfit_id).first()
        if outfit:
            outfit.likes_count = (outfit.likes_count or 0) + 1
        self._db.commit()
        return outfit.likes_count if outfit else 0

    def unlike_outfit(self, user_id: str, shared_outfit_id: str) -> int:
        """Remove a like. Returns new likes_count (idempotent)."""
        like = (
            self._db.query(SharedOutfitLike)
            .filter(
                SharedOutfitLike.user_id == user_id,
                SharedOutfitLike.shared_outfit_id == shared_outfit_id,
            )
            .first()
        )
        if not like:
            outfit = self._db.query(SharedOutfit).filter(SharedOutfit.id == shared_outfit_id).first()
            return outfit.likes_count if outfit else 0

        self._db.delete(like)
        outfit = self._db.query(SharedOutfit).filter(SharedOutfit.id == shared_outfit_id).first()
        if outfit:
            outfit.likes_count = max(0, (outfit.likes_count or 1) - 1)
        self._db.commit()
        return outfit.likes_count if outfit else 0

    def get_discovery_feed(
        self,
        occasion_tag: Optional[str] = None,
        style_tag: Optional[str] = None,
        min_formality: Optional[float] = None,
        max_formality: Optional[float] = None,
        page: int = 1,
        limit: int = 20,
    ) -> List[SharedOutfit]:
        """Return paginated public shared outfits with optional filters."""
        query = self._db.query(SharedOutfit).filter(SharedOutfit.is_public.is_(True))

        if min_formality is not None:
            query = query.filter(SharedOutfit.formality_score >= min_formality)
        if max_formality is not None:
            query = query.filter(SharedOutfit.formality_score <= max_formality)

        outfits: List[SharedOutfit] = (
            query.order_by(SharedOutfit.shared_at.desc())
            .offset((page - 1) * limit)
            .limit(limit)
            .all()
        )

        # Python-side tag filtering (JSON columns are not easily SQL-filterable in SQLite)
        if occasion_tag:
            tag = occasion_tag.lower()
            outfits = [o for o in outfits if tag in [t.lower() for t in (o.occasion_tags or [])]]
        if style_tag:
            tag = style_tag.lower()
            outfits = [o for o in outfits if tag in [t.lower() for t in (o.style_tags or [])]]

        return outfits

    # ------------------------------------------------------------------
    # Mood Board CRUD
    # ------------------------------------------------------------------

    def create_board(
        self,
        user_id: str,
        name: str,
        description: Optional[str] = None,
    ) -> MoodBoard:
        """Create a new mood board for the user."""
        board = MoodBoard(
            id=str(uuid4()),
            user_id=user_id,
            name=name,
            description=description,
            is_active=False,
        )
        self._db.add(board)
        self._db.commit()
        self._db.refresh(board)
        logger.info(f"User {user_id}: created board '{name}' ({board.id})")
        return board

    def list_boards(self, user_id: str) -> List[MoodBoard]:
        return (
            self._db.query(MoodBoard)
            .filter(MoodBoard.user_id == user_id)
            .order_by(MoodBoard.created_at.desc())
            .all()
        )

    def get_board(self, user_id: str, board_id: str) -> MoodBoard:
        return self._get_board_or_raise(board_id, user_id)

    def update_board(
        self,
        user_id: str,
        board_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> MoodBoard:
        board = self._get_board_or_raise(board_id, user_id)
        if name is not None:
            board.name = name
        if description is not None:
            board.description = description
        self._db.commit()
        self._db.refresh(board)
        return board

    def delete_board(self, user_id: str, board_id: str) -> bool:
        board = self._get_board_or_raise(board_id, user_id)
        self._db.delete(board)
        self._db.commit()
        return True

    def activate_board(self, user_id: str, board_id: str) -> MoodBoard:
        """Set this board as the active one (deactivates all others)."""
        # Deactivate all boards for the user
        self._db.query(MoodBoard).filter(MoodBoard.user_id == user_id).update(
            {"is_active": False}
        )
        board = self._get_board_or_raise(board_id, user_id)
        board.is_active = True
        self._db.commit()
        self._db.refresh(board)
        return board

    # ------------------------------------------------------------------
    # Items
    # ------------------------------------------------------------------

    def save_outfit_to_board(
        self,
        user_id: str,
        board_id: str,
        shared_outfit_id: str,
        personal_note: Optional[str] = None,
    ) -> MoodBoardItem:
        """Save a shared outfit into the user's board (idempotent)."""
        board = self._get_board_or_raise(board_id, user_id)

        shared_outfit = (
            self._db.query(SharedOutfit)
            .filter(SharedOutfit.id == shared_outfit_id, SharedOutfit.is_public.is_(True))
            .first()
        )
        if not shared_outfit:
            from fastapi import HTTPException, status
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Shared outfit {shared_outfit_id} not found or not public",
            )

        # Idempotent check
        existing = (
            self._db.query(MoodBoardItem)
            .filter(
                MoodBoardItem.board_id == board_id,
                MoodBoardItem.shared_outfit_id == shared_outfit_id,
            )
            .first()
        )
        if existing:
            return existing

        item = MoodBoardItem(
            id=str(uuid4()),
            board_id=board_id,
            shared_outfit_id=shared_outfit_id,
            personal_note=personal_note,
        )
        self._db.add(item)

        # Increment counter
        shared_outfit.saves_count = (shared_outfit.saves_count or 0) + 1

        self._db.commit()
        self._db.refresh(item)

        # Invalidate the board's cached style profile
        self._profile_builder.invalidate_profile(board_id)

        return item

    def remove_item(self, user_id: str, board_id: str, item_id: str) -> bool:
        """Remove an item from the board and invalidate the profile."""
        self._get_board_or_raise(board_id, user_id)
        item = (
            self._db.query(MoodBoardItem)
            .filter(MoodBoardItem.id == item_id, MoodBoardItem.board_id == board_id)
            .first()
        )
        if not item:
            from fastapi import HTTPException, status
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Item not found")

        # Decrement counter
        shared_outfit = self._db.query(SharedOutfit).filter(SharedOutfit.id == item.shared_outfit_id).first()
        if shared_outfit:
            shared_outfit.saves_count = max(0, (shared_outfit.saves_count or 1) - 1)

        self._db.delete(item)
        self._db.commit()
        self._profile_builder.invalidate_profile(board_id)
        return True

    def list_items(self, board_id: str) -> List[MoodBoardItem]:
        return (
            self._db.query(MoodBoardItem)
            .filter(MoodBoardItem.board_id == board_id)
            .order_by(MoodBoardItem.saved_at.desc())
            .all()
        )

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def get_style_profile(self, user_id: str, board_id: str) -> Optional[MoodBoardStyleProfile]:
        """Return (or rebuild) the style profile for a board."""
        self._get_board_or_raise(board_id, user_id)
        return self._profile_builder.get_or_rebuild_profile(board_id)

    def get_gap_analysis(
        self,
        user_id: str,
        board_id: str,
        wardrobe_colors: Optional[List[str]] = None,
        wardrobe_styles: Optional[List[str]] = None,
        wardrobe_piece_types: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Compare the board profile with the user's wardrobe data and return
        a gap analysis dict.

        The caller passes simplified wardrobe signals (lists of colors, styles,
        piece types) extracted from their garments.
        """
        profile = self.get_style_profile(user_id, board_id)
        if not profile:
            return {
                "alignment_score": 0.0,
                "missing_colors": [],
                "missing_styles": [],
                "missing_piece_types": [],
                "well_covered": [],
            }

        w_colors = set(c.lower() for c in (wardrobe_colors or []))
        w_styles = set(s.lower() for s in (wardrobe_styles or []))
        w_pieces = set(p.lower() for p in (wardrobe_piece_types or []))

        board_colors = [
            (e["color"].lower() if isinstance(e, dict) else e.lower())
            for e in (profile.dominant_colors or [])
        ]
        board_styles = [s.lower() for s in (profile.dominant_styles or {}).keys()]

        missing_colors = [c for c in board_colors if c not in w_colors]
        covered_colors = [c for c in board_colors if c in w_colors]
        missing_styles = [s for s in board_styles if s not in w_styles]
        covered_styles = [s for s in board_styles if s in w_styles]
        well_covered = covered_colors + covered_styles

        # Simple alignment: fraction of board signals covered
        total_signals = len(board_colors) + len(board_styles)
        covered_signals = len(covered_colors) + len(covered_styles)
        alignment = (covered_signals / total_signals * 100.0) if total_signals > 0 else 50.0

        return {
            "alignment_score": round(alignment, 1),
            "missing_colors": missing_colors,
            "missing_styles": missing_styles,
            "missing_piece_types": [],  # Requires garment-type data — left for caller
            "well_covered": well_covered,
        }

    def get_moodboard_recommendations(
        self,
        user_id: str,
        board_id: str,
        candidate_outfits: List[Dict[str, Any]],
        user_preference: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """
        Re-rank candidate outfits by blending their base score with how well
        they match the active mood board.

        Each element of candidate_outfits should have:
            - garment_ids:       list[str]
            - base_score:        float (0-1)
            - dominant_colors:   list[str]  (optional)
            - dominant_styles:   dict       (optional)
            - formality_score:   float      (optional)
            - embedding_vector:  list[float](optional)

        Returns the same list sorted by final_score descending, with
        ``moodboard_score`` and ``final_score`` added to each entry.
        """
        profile = self.get_style_profile(user_id, board_id)

        results = []
        for outfit in candidate_outfits:
            base = float(outfit.get("base_score", 0.5))
            if profile:
                mb_score = self._scorer.score_outfit_vs_board(outfit, profile)
                final = self._scorer.apply_moodboard_boost(base, mb_score, user_preference)
                boost = final - base
            else:
                mb_score = 0.0
                final = base
                boost = 0.0

            results.append(
                {
                    **outfit,
                    "moodboard_score": mb_score,
                    "final_score": final,
                    "boost_applied": round(boost, 4),
                }
            )

        results.sort(key=lambda x: x["final_score"], reverse=True)
        return results

    async def get_board_summary(self, user_id: str, board_id: str) -> str:
        """Return an LLM-generated summary of the board's style profile."""
        board = self._get_board_or_raise(board_id, user_id)
        profile = self.get_style_profile(user_id, board_id)
        if not profile:
            return f"Your board '{board.name}' doesn't have enough saved outfits yet to build a style profile."
        return await self._explainer.generate_board_summary(profile, board.name)

    async def get_gap_analysis_with_explanation(
        self,
        user_id: str,
        board_id: str,
        wardrobe_colors: Optional[List[str]] = None,
        wardrobe_styles: Optional[List[str]] = None,
        wardrobe_piece_types: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Gap analysis enriched with an LLM explanation."""
        board = self._get_board_or_raise(board_id, user_id)
        gap = self.get_gap_analysis(
            user_id, board_id, wardrobe_colors, wardrobe_styles, wardrobe_piece_types
        )
        explanation = await self._explainer.explain_gap_analysis(
            alignment_score=gap["alignment_score"],
            missing_colors=gap["missing_colors"],
            missing_styles=gap["missing_styles"],
            missing_piece_types=gap["missing_piece_types"],
            well_covered=gap["well_covered"],
            board_name=board.name,
        )
        return {**gap, "explanation": explanation}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_board_or_raise(self, board_id: str, user_id: Optional[str] = None) -> MoodBoard:
        from fastapi import HTTPException, status

        board = self._db.query(MoodBoard).filter(MoodBoard.id == board_id).first()
        if not board:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Board {board_id} not found",
            )
        if user_id and board.user_id != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not your board",
            )
        return board

    def _get_shared_outfit_or_raise(
        self, shared_outfit_id: str, user_id: Optional[str] = None
    ) -> SharedOutfit:
        from fastapi import HTTPException, status

        outfit = self._db.query(SharedOutfit).filter(SharedOutfit.id == shared_outfit_id).first()
        if not outfit:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Shared outfit {shared_outfit_id} not found",
            )
        if user_id and outfit.owner_user_id != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not your outfit",
            )
        return outfit
