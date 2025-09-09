"""
Utilities module for CT-AI project

This module contains:
- CT transforms (Radon, inverse Radon)
- Visualization utilities
- Helper functions
"""

# Import CT transforms
from .ct_transforms import (
    radon_transform,
    iradon_transform, 
    fbp_reconstruction,
    simulate_sparse_view_ct,
    compare_reconstruction_filters,
    CTTransforms
)

# Placeholder for visualization (will be implemented)
def visualize_results(original, reconstruction, save_path=None):
    """Placeholder visualization function."""
    print(f"Visualizing results: {original.shape} -> {reconstruction.shape}")
    if save_path:
        print(f"Would save to: {save_path}")

__all__ = [
    'radon_transform',
    'iradon_transform', 
    'fbp_reconstruction',
    'simulate_sparse_view_ct',
    'compare_reconstruction_filters',
    'CTTransforms',
    'visualize_results'
]
