#!/usr/bin/env python3
"""
Script to generate embeddings for all wardrobe items.

Usage:
    python scripts/generate_embeddings.py --input data/processed/wardrobe.json
"""
import argparse
import asyncio
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.layer1_vision import EmbeddingGenerator
from src.core.models import Garment
from src.core import get_logger

logger = get_logger(__name__)


async def generate_embeddings(garments: list[Garment]) -> list[Garment]:
    """Generate embeddings for all garments."""
    generator = EmbeddingGenerator()
    
    logger.info(f"Generating embeddings for {len(garments)} garments...")
    
    embeddings = await generator.generate_batch_embeddings(garments)
    
    for garment, embedding in zip(garments, embeddings):
        garment.embedding = embedding
    
    logger.info("Embedding generation complete!")
    return garments


def main():
    parser = argparse.ArgumentParser(description="Generate wardrobe embeddings")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    
    args = parser.parse_args()
    
    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        return
    
    # Load garments
    with open(args.input) as f:
        data = json.load(f)
    
    garments = [Garment(**g) for g in data.get("garments", [])]
    
    if not garments:
        logger.warning("No garments found in input file")
        return
    
    # Generate embeddings
    garments = asyncio.run(generate_embeddings(garments))
    
    # Save
    output_path = args.output or args.input.with_suffix(".embeddings.json")
    with open(output_path, "w") as f:
        json.dump({
            "garments": [g.model_dump() for g in garments]
        }, f, indent=2, default=str)
    
    logger.info(f"Saved embeddings to {output_path}")


if __name__ == "__main__":
    main()
