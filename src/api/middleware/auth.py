"""Authentication and authorization middleware for the API."""
from fastapi import Depends, HTTPException, status, Header
from sqlalchemy.orm import Session
from typing import Optional
import jwt
import os
from datetime import datetime

from src.database import get_db
from src.database.models import UserSubscription

# JWT Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")
ALGORITHM = "HS256"


class AuthenticationError(Exception):
    """Raised when authentication fails."""
    pass


class AuthorizationError(Exception):
    """Raised when user lacks required permissions."""
    pass


def decode_token(token: str) -> dict:
    """
    Decode and validate JWT token.
    
    Returns:
        dict with token payload
        
    Raises:
        AuthenticationError if token is invalid or expired
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise AuthenticationError("Invalid token: missing subject")
        return {"user_id": user_id, **{k: v for k, v in payload.items() if k != "sub"}}
    except jwt.ExpiredSignatureError:
        raise AuthenticationError("Token has expired")
    except jwt.InvalidTokenError as e:
        raise AuthenticationError(f"Invalid token: {str(e)}")


def get_current_user(
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
) -> UserSubscription:
    """
    Extract and validate current user from JWT token in Authorization header.
    
    Returns:
        UserSubscription model for the authenticated user
        
    Raises:
        HTTPException 401 if authentication fails
    """
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        scheme, credentials = authorization.split()
        if scheme.lower() != "bearer":
            raise ValueError("Invalid authentication scheme")
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header format. Use: 'Bearer <token>'",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        token_data = decode_token(credentials)
        user_id = token_data.get("user_id")
    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Look up user subscription
    user = db.query(UserSubscription).filter(
        UserSubscription.user_id == user_id
    ).first()

    if not user:
        # Create subscription for new user
        user = UserSubscription(user_id=user_id, subscription_tier="free")
        db.add(user)
        db.commit()
        db.refresh(user)

    return user


def require_premium(user: UserSubscription = Depends(get_current_user)) -> UserSubscription:
    """
    Dependency to require premium subscription.
    
    Returns:
        UserSubscription if user has active premium subscription
        
    Raises:
        HTTPException 403 if user does not have premium access
    """
    if not user.is_premium or not user.is_active():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This feature requires a premium subscription. Please upgrade to continue.",
            headers={"X-Premium-Required": "true"},
        )
    return user


def create_token(user_id: str, expires_delta: Optional[int] = None) -> str:
    """
    Create a JWT token for a user.
    
    Args:
        user_id: User identifier
        expires_delta: Token expiration in seconds (default: 7 days)
        
    Returns:
        Encoded JWT token string
    """
    from datetime import timedelta
    
    if expires_delta is None:
        expires_delta = 7 * 24 * 60 * 60  # 7 days default

    expire = datetime.utcnow() + timedelta(seconds=expires_delta)
    payload = {"sub": user_id, "exp": expire}
    
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
