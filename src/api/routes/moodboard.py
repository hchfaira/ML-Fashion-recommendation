"""API routes for the Mood Board feature."""
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from typing import List, Optional

from src.database import get_db
from src.database.models import UserSubscription
from src.api.middleware.auth import get_current_user
from src.api.models.moodboard import (
    DiscoveryFeedRequest,
    MoodBoardCreate,
    MoodBoardGapAnalysis,
    MoodBoardItemResponse,
    MoodBoardResponse,
    MoodBoardStyleProfileResponse,
    MoodBoardUpdate,
    SaveOutfitToBoardRequest,
    ShareOutfitRequest,
    SharedOutfitResponse,
    ColorFrequency,
)
from src.api.services.moodboard_service import MoodBoardService
from src.core import get_logger

logger = get_logger(__name__)
router = APIRouter()


# ── Helpers ──────────────────────────────────────────────────────────────────

def _get_service(db: Session) -> MoodBoardService:
    return MoodBoardService(db)


# ── Shared Outfits ────────────────────────────────────────────────────────────

@router.post(
    "/shared-outfits",
    response_model=SharedOutfitResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Share an outfit publicly",
)
async def share_outfit(
    body: ShareOutfitRequest,
    user: UserSubscription = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> SharedOutfitResponse:
    """Share one of your outfits so others can discover and save it."""
    svc = _get_service(db)
    shared = svc.share_outfit(user.user_id, body.model_dump())
    return SharedOutfitResponse(**shared.to_dict())


@router.delete(
    "/shared-outfits/{shared_outfit_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Unpublish a shared outfit",
)
async def unpublish_outfit(
    shared_outfit_id: str,
    user: UserSubscription = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> None:
    """Mark your shared outfit as private (removes it from discovery)."""
    svc = _get_service(db)
    svc.unpublish_outfit(user.user_id, shared_outfit_id)


@router.post(
    "/shared-outfits/{shared_outfit_id}/like",
    summary="Like a shared outfit",
)
async def like_outfit(
    shared_outfit_id: str,
    user: UserSubscription = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    svc = _get_service(db)
    count = svc.like_outfit(user.user_id, shared_outfit_id)
    return {"likes_count": count}


@router.delete(
    "/shared-outfits/{shared_outfit_id}/like",
    summary="Unlike a shared outfit",
)
async def unlike_outfit(
    shared_outfit_id: str,
    user: UserSubscription = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    svc = _get_service(db)
    count = svc.unlike_outfit(user.user_id, shared_outfit_id)
    return {"likes_count": count}


# ── Discovery Feed ────────────────────────────────────────────────────────────

@router.get(
    "/discover",
    response_model=List[SharedOutfitResponse],
    summary="Browse publicly shared outfits",
)
async def discover_outfits(
    occasion_tag: Optional[str] = Query(None),
    style_tag: Optional[str] = Query(None),
    min_formality: Optional[float] = Query(None, ge=0.0, le=1.0),
    max_formality: Optional[float] = Query(None, ge=0.0, le=1.0),
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
) -> List[SharedOutfitResponse]:
    """Paginated feed of public shared outfits with optional filters."""
    svc = _get_service(db)
    outfits = svc.get_discovery_feed(
        occasion_tag=occasion_tag,
        style_tag=style_tag,
        min_formality=min_formality,
        max_formality=max_formality,
        page=page,
        limit=limit,
    )
    return [SharedOutfitResponse(**o.to_dict()) for o in outfits]


# ── Mood Board CRUD ───────────────────────────────────────────────────────────

@router.post(
    "/moodboards",
    response_model=MoodBoardResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a mood board",
)
async def create_board(
    body: MoodBoardCreate,
    user: UserSubscription = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> MoodBoardResponse:
    svc = _get_service(db)
    board = svc.create_board(user.user_id, body.name, body.description)
    d = board.to_dict()
    d["items_count"] = 0
    return MoodBoardResponse(**d)


@router.get(
    "/moodboards",
    response_model=List[MoodBoardResponse],
    summary="List all mood boards",
)
async def list_boards(
    user: UserSubscription = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> List[MoodBoardResponse]:
    svc = _get_service(db)
    boards = svc.list_boards(user.user_id)
    return [MoodBoardResponse(**b.to_dict()) for b in boards]


@router.get(
    "/moodboards/{board_id}",
    response_model=MoodBoardResponse,
    summary="Get a mood board",
)
async def get_board(
    board_id: str,
    user: UserSubscription = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> MoodBoardResponse:
    svc = _get_service(db)
    board = svc.get_board(user.user_id, board_id)
    return MoodBoardResponse(**board.to_dict())


@router.put(
    "/moodboards/{board_id}",
    response_model=MoodBoardResponse,
    summary="Update a mood board",
)
async def update_board(
    board_id: str,
    body: MoodBoardUpdate,
    user: UserSubscription = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> MoodBoardResponse:
    svc = _get_service(db)
    board = svc.update_board(user.user_id, board_id, body.name, body.description)
    return MoodBoardResponse(**board.to_dict())


@router.delete(
    "/moodboards/{board_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a mood board",
)
async def delete_board(
    board_id: str,
    user: UserSubscription = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> None:
    svc = _get_service(db)
    svc.delete_board(user.user_id, board_id)


@router.post(
    "/moodboards/{board_id}/activate",
    response_model=MoodBoardResponse,
    summary="Set this board as the active one for recommendations",
)
async def activate_board(
    board_id: str,
    user: UserSubscription = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> MoodBoardResponse:
    svc = _get_service(db)
    board = svc.activate_board(user.user_id, board_id)
    return MoodBoardResponse(**board.to_dict())


# ── Mood Board Items ──────────────────────────────────────────────────────────

@router.post(
    "/moodboards/{board_id}/items",
    response_model=MoodBoardItemResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Save a shared outfit to a mood board",
)
async def save_outfit(
    board_id: str,
    body: SaveOutfitToBoardRequest,
    user: UserSubscription = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> MoodBoardItemResponse:
    svc = _get_service(db)
    item = svc.save_outfit_to_board(
        user.user_id, board_id, body.shared_outfit_id, body.personal_note
    )
    return MoodBoardItemResponse(**item.to_dict())


@router.get(
    "/moodboards/{board_id}/items",
    response_model=List[MoodBoardItemResponse],
    summary="List items in a mood board",
)
async def list_items(
    board_id: str,
    user: UserSubscription = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> List[MoodBoardItemResponse]:
    svc = _get_service(db)
    svc.get_board(user.user_id, board_id)  # ownership check
    items = svc.list_items(board_id)
    return [MoodBoardItemResponse(**i.to_dict()) for i in items]


@router.delete(
    "/moodboards/{board_id}/items/{item_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Remove an item from a mood board",
)
async def remove_item(
    board_id: str,
    item_id: str,
    user: UserSubscription = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> None:
    svc = _get_service(db)
    svc.remove_item(user.user_id, board_id, item_id)


# ── Analysis ──────────────────────────────────────────────────────────────────

@router.get(
    "/moodboards/{board_id}/style-profile",
    response_model=MoodBoardStyleProfileResponse,
    summary="Get the aggregated style profile of a mood board",
)
async def get_style_profile(
    board_id: str,
    user: UserSubscription = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> MoodBoardStyleProfileResponse:
    svc = _get_service(db)
    profile = svc.get_style_profile(user.user_id, board_id)
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No style profile yet — add at least one outfit to this board first.",
        )
    raw = profile.to_dict()
    raw["dominant_colors"] = [
        ColorFrequency(**c) if isinstance(c, dict) else ColorFrequency(color=c, frequency=0.0)
        for c in (raw.get("dominant_colors") or [])
    ]
    return MoodBoardStyleProfileResponse(**raw)


@router.get(
    "/moodboards/{board_id}/gap-analysis",
    response_model=MoodBoardGapAnalysis,
    summary="Gap analysis between the board and the user's wardrobe",
)
async def gap_analysis(
    board_id: str,
    wardrobe_colors: Optional[str] = Query(
        None, description="Comma-separated list of wardrobe colors"
    ),
    wardrobe_styles: Optional[str] = Query(
        None, description="Comma-separated list of wardrobe styles"
    ),
    user: UserSubscription = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> MoodBoardGapAnalysis:
    svc = _get_service(db)
    colors = [c.strip() for c in wardrobe_colors.split(",")] if wardrobe_colors else []
    styles = [s.strip() for s in wardrobe_styles.split(",")] if wardrobe_styles else []
    gap = await svc.get_gap_analysis_with_explanation(
        user.user_id, board_id, wardrobe_colors=colors, wardrobe_styles=styles
    )
    return MoodBoardGapAnalysis(**gap)


@router.get(
    "/moodboards/{board_id}/summary",
    summary="LLM-generated style summary of the board",
)
async def board_summary(
    board_id: str,
    user: UserSubscription = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    svc = _get_service(db)
    text = await svc.get_board_summary(user.user_id, board_id)
    return {"summary": text}
