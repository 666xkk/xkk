"""Detection package with lazy submodule loading.

Standalone analysis modules must not import the legacy CUDA training stack.
"""

__all__ = [
    "evaluation",
    "orthrus_gnn_testing",
    "orthrus_gnn_training",
]
