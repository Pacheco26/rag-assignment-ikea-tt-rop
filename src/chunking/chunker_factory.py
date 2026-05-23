"""Factory for instantiating chunkers by name."""

from src.chunking.base_chunker import BaseChunker
from src.chunking.fixed_size_chunker import FixedSizeChunker
from src.chunking.semantic_chunker import SemanticChunker
from src.chunking.hierarchical_chunker import HierarchicalChunker


_CHUNKERS = {
    "fixed_size": FixedSizeChunker,
    "semantic": SemanticChunker,
    "hierarchical": HierarchicalChunker,
}


def create_chunker(strategy: str, **kwargs) -> BaseChunker:
    """Create a chunker by strategy name ('fixed_size', 'semantic', 'hierarchical')."""
    if strategy not in _CHUNKERS:
        raise ValueError(
            f"Unknown strategy '{strategy}'. "
            f"Available: {list(_CHUNKERS.keys())}"
        )
    return _CHUNKERS[strategy](**kwargs)


def available_strategies() -> list[str]:
    """Return list of available chunking strategy names."""
    return list(_CHUNKERS.keys())
