"""Logging configuration for the Fashion Recommendation System."""
import sys
from loguru import logger
from config import get_settings


def get_logger(name: str = None):
    """
    Get a configured logger instance.
    
    Args:
        name: Optional name for the logger context
        
    Returns:
        Configured loguru logger
    """
    settings = get_settings()
    
    # Remove default handler
    logger.remove()
    
    # Add console handler
    logger.add(
        sys.stderr,
        level=settings.log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=True
    )
    
    # Add file handler for production
    if settings.app_env == "production":
        logger.add(
            "logs/app.log",
            rotation="500 MB",
            retention="10 days",
            level=settings.log_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
        )
    
    if name:
        return logger.bind(name=name)
    
    return logger
