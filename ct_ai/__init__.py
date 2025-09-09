"""
CT-AI: Physics-Informed Deep Learning for Sparse-View CT Reconstruction

A comprehensive package for advanced CT reconstruction using deep learning,
combining classical algorithms with modern neural networks and physics-informed
constraints.

Main Components:
- models: Neural network architectures (CompactCTNet, etc.)
- data: Dataset generation and phantom creation
- training: Training pipelines and optimization
- evaluation: Metrics calculation and method comparison
- utils: Utility functions and helpers

Author: Your Name
Date: September 2024
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@university.edu"

# Main imports for easy access
from .models import CompactCTNet, PhysicsInformedLoss
from .utils import (
    radon_transform, 
    iradon_transform, 
    fbp_reconstruction,
    visualize_results
)

# Placeholder imports for modules not yet implemented
# from .data import PhantomGenerator, CTDataset, create_phantom_dataset
# from .training import CTTrainer, TrainingConfig  
# from .evaluation import MetricsCalculator, SOTAComparison

# Package-level configuration
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Set random seeds for reproducibility
import numpy as np
import tensorflow as tf

def set_random_seeds(seed=42):
    """Set random seeds for reproducible results."""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
# Set default seed
set_random_seeds()

print(f"CT-AI v{__version__} loaded successfully! 🧠✨")
print("Ready for advanced CT reconstruction with deep learning.")
