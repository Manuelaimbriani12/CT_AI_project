"""
Training module for CT-AI project

This module contains:
- CTTrainer: Advanced training system for CT reconstruction
- Training utilities and optimizers
"""

from .ct_trainer import CTTrainer, create_trainer_from_config

__all__ = [
    'CTTrainer',
    'create_trainer_from_config'
]
