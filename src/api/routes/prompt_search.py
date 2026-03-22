"""
API routes for Natural Language Outfit Search.

Prefix: /outfits/search  (registered in api/__init__.py)

Endpoints:
  POST /outfits/search/prompt          — one-shot search
  POST /outfits/search/prompt/refine   — refine an existing session
  POST /outfits/search/prompt/parse    — debug: show prompt interpretation
  GET  /outfits/search/prompt/suggestions — example prompts
  POST /outfits/search/conversation    — start a conversation session (alias)
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from src.database import get_db
from src.database.models import UserSubscription
from src.api.middleware.auth import get_current_user
from src.api.models.prompt_search import (
    ConversationRefineRequest,
    ConversationStartRequest,
    ParseOnlyResponse,
    ParsedPromptResponse,
    PromptSearchRequest,
    PromptSearchResponse,
    SuggestionsResponse,
    WardrobeFiltersResponse,
)
from src.api.services.prompt_search_service import PromptSearchService
from src.layer3_context.prompt_to_filters import PromptToFilters
from src.core import get_logger

logger = get_logger(__name__)
router = APIRouter()

_svc = PromptSearchService()
_translator = PromptToFilters()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_prompt_response(result) -> PromptSearchResponse:
    d = result.to_dict()
    return PromptSearchResponse(
        outfits=d["outfits"],
        prompt_interpretation=ParsedPromptResponse(**d["prompt_interpretation"]),
        confidence=d["confidence"],
        explanation=d["explanation"],
        compromise_note=d["compromise_note"],
        missing_piece_suggestion=d["missing_piece_suggestion"],
        session_id=d["session_id"],
        total_candidates=d["total_candidates"],
        is_off_topic=d["is_off_topic"],
        clarification_needed=d["clarification_needed"],
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get(
    "/outfits/search/prompt/suggestions",
    response_model=SuggestionsResponse,
    summary="Get example search prompts",
    tags=["Prompt Search"],
)
async def get_suggestions() -> SuggestionsResponse:
    """Return a list of example prompts to help users get started."""
    return SuggestionsResponse(suggestions=_svc.get_suggestions())


@router.post(
    "/outfits/search/prompt/parse",
    response_model=ParseOnlyResponse,
    summary="Debug: parse a prompt without running the full search",
    tags=["Prompt Search"],
)
async def parse_prompt(
    body: PromptSearchRequest,
    user: UserSubscription = Depends(get_current_user),
) -> ParseOnlyResponse:
    """
    Parse a free-text prompt and return the structured interpretation
    along with the concrete wardrobe filters — no outfit scoring performed.
    """
    parsed = await _svc.parse_only(body.prompt)
    filters = _translator.translate(parsed)
    return ParseOnlyResponse(
        parsed=ParsedPromptResponse(**parsed.to_dict()),
        filters=WardrobeFiltersResponse(**filters.to_dict()),
    )


@router.post(
    "/outfits/search/prompt",
    response_model=PromptSearchResponse,
    summary="Search outfits using a free-text prompt",
    tags=["Prompt Search"],
)
async def search_by_prompt(
    body: PromptSearchRequest,
    user: UserSubscription = Depends(get_current_user),
) -> PromptSearchResponse:
    """
    Convert a free-text request (e.g. "something blue for a wedding") into a
    ranked list of matching outfits from the provided candidates.

    - **prompt**: natural language description of the desired outfit
    - **candidate_outfits**: outfit dicts from the user's wardrobe
    - **base_scores**: optional pre-computed style scores
    - **preference**: 0 = classic scoring only, 1 = prompt scoring only
    """
    result = await _svc.search(
        user_id=user.user_id,
        prompt=body.prompt,
        candidate_outfits=body.candidate_outfits,
        base_scores=body.base_scores,
        max_results=body.max_results,
        preference=body.preference,
        explain=body.explain,
    )

    if result.parsed_prompt.is_off_topic:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "code": "OFF_TOPIC",
                "message": "The prompt doesn't seem to be about clothing. Please describe the outfit you're looking for.",
            },
        )

    return _to_prompt_response(result)


@router.post(
    "/outfits/search/conversation",
    response_model=PromptSearchResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Start a conversational outfit search session",
    tags=["Prompt Search"],
)
async def start_conversation(
    body: ConversationStartRequest,
    user: UserSubscription = Depends(get_current_user),
) -> PromptSearchResponse:
    """
    Like ``/search/prompt`` but opens a session so you can call
    ``/search/prompt/refine`` afterwards to narrow down results.
    """
    result = await _svc.start_conversation(
        user_id=user.user_id,
        prompt=body.prompt,
        candidate_outfits=body.candidate_outfits,
        base_scores=body.base_scores,
        max_results=body.max_results,
        preference=body.preference,
    )

    if result.parsed_prompt.is_off_topic:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"code": "OFF_TOPIC", "message": "The prompt doesn't seem to be about clothing."},
        )

    return _to_prompt_response(result)


@router.post(
    "/outfits/search/prompt/refine",
    response_model=PromptSearchResponse,
    summary="Refine an existing conversational search",
    tags=["Prompt Search"],
)
async def refine_conversation(
    session_id: str,
    body: ConversationRefineRequest,
    user: UserSubscription = Depends(get_current_user),
) -> PromptSearchResponse:
    """
    Apply a natural-language refinement to an existing session.

    Supported refinements (detected automatically):
    - "More formal" / "Less formal"
    - "Without heels"
    - "Show me more" (next page)
    - "I don't like this style"
    - Any new colour, occasion, or style preference
    """
    try:
        result = await _svc.refine_conversation(
            session_id=session_id,
            refinement=body.refinement,
            candidate_outfits=body.candidate_outfits,
            base_scores=body.base_scores,
            max_results=body.max_results,
            preference=body.preference,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))

    return _to_prompt_response(result)
