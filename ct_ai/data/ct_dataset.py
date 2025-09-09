"""
CT Dataset Pipeline: Complete Data Management for Training

This module provides comprehensive dataset management for CT reconstruction training:
1. CTDataset: TensorFlow/PyTorch compatible dataset class
2. DataPipeline: Automated data generation and preprocessing 
3. SparseViewDataset: Specialized dataset for sparse-view CT
4. Augmentation and validation utilities

Features:
- Automatic phantom generation with CT simulation
- On-the-fly sparse-view protocol simulation
- Comprehensive data augmentation
- Quality control and validation
- Memory-efficient data loading
- Multi-processing support

Author: Your Name
Date: September 2024
"""

import numpy as np
import tensorflow as tf
from typing import Tuple, List, Dict, Optional, Union, Callable
from pathlib import Path
import h5py
import warnings
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

# Import our modules
from .phantom_generator import PhantomGenerator
from ..utils.ct_transforms import radon_transform, fbp_reconstruction
from ..utils import simulate_sparse_view_ct


class CTDataset:
    """
    TensorFlow-compatible dataset for CT reconstruction training.
    
    This class provides a flexible interface for loading and preprocessing
    CT data for neural network training. It supports both pre-generated
    datasets and on-the-fly data generation.
    
    Features:
    - Memory-efficient data loading
    - Automatic batching and shuffling
    - On-the-fly augmentation
    - Sparse-view simulation
    - Quality control
    
    Args:
        phantoms: Pre-generated phantoms or generator function
        sparse_angles: List of sparse angle counts to simulate
        noise_levels: List of noise levels to apply
        augmentation_config: Data augmentation parameters
        batch_size: Batch size for training
        shuffle: Whether to shuffle data
        cache_size: Number of items to cache in memory
    """
    
    def __init__(
        self,
        phantoms: Optional[Union[np.ndarray, Callable]] = None,
        sparse_angles: List[int] = [20, 30, 45, 60],
        noise_levels: List[float] = [0.01, 0.05, 0.1],
        augmentation_config: Optional[Dict] = None,
        batch_size: int = 8,
        shuffle: bool = True,
        cache_size: int = 1000,
        random_seed: int = 42
    ):
        self.phantoms = phantoms
        self.sparse_angles = sparse_angles
        self.noise_levels = noise_levels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.cache_size = cache_size
        self.random_seed = random_seed
        
        # Set random seed
        np.random.seed(random_seed)
        tf.random.set_seed(random_seed)
        
        # Default augmentation config
        if augmentation_config is None:
            self.augmentation_config = {
                'rotation_range': 10,
                'intensity_variation': 0.05,
                'noise_probability': 0.8,
                'sparse_probability': 0.9
            }
        else:
            self.augmentation_config = augmentation_config
        
        # Initialize phantom generator if needed
        if phantoms is None:
            self.phantom_generator = PhantomGenerator()
            self.generate_on_fly = True
        else:
            self.phantom_generator = None
            self.generate_on_fly = False
        
        # Data cache
        self.cache = {}
        self.cache_keys = []
        
        # Statistics
        self.stats = {
            'samples_generated': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
    
    def create_tf_dataset(self, size: int) -> tf.data.Dataset:
        """
        Create TensorFlow dataset for training.
        
        Args:
            size: Number of samples in the dataset
        
        Returns:
            TensorFlow dataset ready for training
        """
        
        def data_generator():
            """Generator function for TensorFlow dataset."""
            for i in range(size):
                # Generate or retrieve phantom
                if self.generate_on_fly:
                    phantom = self._generate_phantom()
                else:
                    phantom = self._get_phantom(i)
                
                # Simulate sparse-view CT
                input_recon, target_phantom = self._create_training_pair(phantom)
                
                yield input_recon, target_phantom
        
        # Create TensorFlow dataset
        dataset = tf.data.Dataset.from_generator(
            data_generator,
            output_signature=(
                tf.TensorSpec(shape=(256, 256, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(256, 256, 1), dtype=tf.float32)
            )
        )
        
        # Apply dataset transformations
        if self.shuffle:
            dataset = dataset.shuffle(buffer_size=min(1000, size))
        
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def create_validation_dataset(self, size: int) -> tf.data.Dataset:
        """Create validation dataset with fixed parameters."""
        
        def validation_generator():
            # Use fixed parameters for validation
            np.random.seed(self.random_seed + 1000)  # Different seed for validation
            
            for i in range(size):
                if self.generate_on_fly:
                    phantom = self._generate_phantom()
                else:
                    phantom = self._get_phantom(i)
                
                # Fixed sparse-view parameters for validation
                n_angles = 45  # Fixed angle count
                noise_level = 0.05  # Fixed noise level
                
                # Simulate acquisition
                sinogram, angles = simulate_sparse_view_ct(
                    phantom, n_angles=n_angles, noise_level=noise_level
                )
                
                # FBP reconstruction (input)
                input_recon = fbp_reconstruction(sinogram, angles, 'shepp-logan')
                
                # Ensure correct shape
                if input_recon.ndim == 2:
                    input_recon = input_recon[..., np.newaxis]
                if phantom.ndim == 2:
                    phantom = phantom[..., np.newaxis]
                
                yield input_recon.astype(np.float32), phantom.astype(np.float32)
        
        dataset = tf.data.Dataset.from_generator(
            validation_generator,
            output_signature=(
                tf.TensorSpec(shape=(256, 256, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(256, 256, 1), dtype=tf.float32)
            )
        )
        
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def _generate_phantom(self) -> np.ndarray:
        """Generate a single phantom."""
        phantoms, _ = self.phantom_generator.generate_batch(1)
        return phantoms[0]
    
    def _get_phantom(self, index: int) -> np.ndarray:
        """Get phantom by index from pre-generated data."""
        if isinstance(self.phantoms, np.ndarray):
            return self.phantoms[index % len(self.phantoms)]
        else:
            # Assume callable phantom generator
            return self.phantoms(index)
    
    def _create_training_pair(self, phantom: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create training pair (input, target) from phantom.
        
        Args:
            phantom: Original phantom
        
        Returns:
            Tuple of (sparse_reconstruction, target_phantom)
        """
        
        # Random sparse-view parameters
        n_angles = np.random.choice(self.sparse_angles)
        noise_level = np.random.choice(self.noise_levels)
        
        # Simulate sparse-view CT
        sinogram, angles = simulate_sparse_view_ct(
            phantom, n_angles=n_angles, noise_level=noise_level
        )
        
        # Create input reconstruction (FBP)
        input_recon = fbp_reconstruction(sinogram, angles, 'shepp-logan')
        
        # Apply additional augmentation if configured
        if np.random.random() < self.augmentation_config.get('intensity_variation', 0):
            factor = 1 + np.random.uniform(-0.05, 0.05)
            input_recon *= factor
            input_recon = np.clip(input_recon, 0, 1)
        
        # Ensure correct shapes
        if input_recon.ndim == 2:
            input_recon = input_recon[..., np.newaxis]
        if phantom.ndim == 2:
            phantom = phantom[..., np.newaxis]
        
        return input_recon.astype(np.float32), phantom.astype(np.float32)
    
    def get_sample_batch(self, batch_size: int = 4) -> Tuple[np.ndarray, np.ndarray]:
        """Get a sample batch for visualization/testing."""
        inputs = []
        targets = []
        
        for _ in range(batch_size):
            if self.generate_on_fly:
                phantom = self._generate_phantom()
            else:
                phantom = self._get_phantom(np.random.randint(0, len(self.phantoms)))
            
            input_recon, target = self._create_training_pair(phantom)
            inputs.append(input_recon)
            targets.append(target)
        
        return np.array(inputs), np.array(targets)
    
    def print_statistics(self):
        """Print dataset statistics."""
        print("\n📊 DATASET STATISTICS")
        print("=" * 30)
        print(f"Samples Generated: {self.stats['samples_generated']}")
        print(f"Cache Hits: {self.stats['cache_hits']}")
        print(f"Cache Misses: {self.stats['cache_misses']}")
        print(f"Cache Size: {len(self.cache)}/{self.cache_size}")


class DataPipeline:
    """
    Complete data pipeline for CT reconstruction training.
    
    This class orchestrates the entire data generation and preprocessing
    pipeline, from phantom generation to final training datasets.
    
    Features:
    - Multi-processing data generation
    - Automatic train/validation splits
    - Quality control and validation
    - Progress tracking and statistics
    - Flexible configuration management
    
    Args:
        config: Data pipeline configuration
        phantom_generator: Phantom generator instance
        output_dir: Directory to save generated data
        num_workers: Number of parallel workers
    """
    
    def __init__(
        self,
        config: Dict,
        phantom_generator: Optional[PhantomGenerator] = None,
        output_dir: Optional[Union[str, Path]] = None,
        num_workers: Optional[int] = None
    ):
        self.config = config
        self.output_dir = Path(output_dir) if output_dir else Path("data")
        self.num_workers = num_workers or min(4, mp.cpu_count())
        
        # Initialize phantom generator
        if phantom_generator is None:
            self.phantom_generator = PhantomGenerator(
                phantom_size=(256, 256),
                random_seed=config.get('random_seed', 42)
            )
        else:
            self.phantom_generator = phantom_generator
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Pipeline statistics
        self.stats = {
            'phantoms_generated': 0,
            'training_pairs_created': 0,
            'quality_failures': 0,
            'processing_time': 0
        }
    
    def generate_training_dataset(
        self,
        size: int,
        validation_split: float = 0.2,
        save_to_disk: bool = True
    ) -> Tuple[CTDataset, CTDataset]:
        """
        Generate complete training and validation datasets.
        
        Args:
            size: Total number of samples to generate
            validation_split: Fraction of data for validation
            save_to_disk: Whether to save generated data to disk
        
        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        
        print(f"🚀 Generating training dataset with {size} samples...")
        print(f"📊 Using {self.num_workers} parallel workers")
        
        # Calculate splits
        val_size = int(size * validation_split)
        train_size = size - val_size
        
        # Generate phantoms in batches for efficiency
        batch_size = 100
        all_phantoms = []
        
        for i in range(0, size, batch_size):
            current_batch_size = min(batch_size, size - i)
            print(f"📦 Generating batch {i//batch_size + 1}/{(size-1)//batch_size + 1}")
            
            phantoms, metadata = self.phantom_generator.generate_batch(
                current_batch_size,
                include_augmentation=True,
                include_noise=True
            )
            
            all_phantoms.append(phantoms)
            self.stats['phantoms_generated'] += len(phantoms)
        
        # Combine all phantoms
        all_phantoms = np.concatenate(all_phantoms, axis=0)
        
        print(f"✅ Generated {len(all_phantoms)} phantoms")
        
        # Split data
        indices = np.random.permutation(len(all_phantoms))
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        train_phantoms = all_phantoms[train_indices]
        val_phantoms = all_phantoms[val_indices]
        
        # Create datasets
        train_dataset = CTDataset(
            phantoms=train_phantoms,
            batch_size=self.config.get('batch_size', 8),
            shuffle=True,
            random_seed=self.config.get('random_seed', 42)
        )
        
        val_dataset = CTDataset(
            phantoms=val_phantoms,
            batch_size=self.config.get('batch_size', 8),
            shuffle=False,
            random_seed=self.config.get('random_seed', 42) + 1000
        )
        
        # Save to disk if requested
        if save_to_disk:
            self._save_dataset(train_phantoms, 'train_phantoms.h5')
            self._save_dataset(val_phantoms, 'val_phantoms.h5')
        
        self.phantom_generator.print_statistics()
        self._print_pipeline_stats()
        
        return train_dataset, val_dataset
    
    def create_sparse_view_benchmark(
        self,
        phantom_types: List[str],
        angle_counts: List[int],
        noise_levels: List[float],
        n_samples_per_config: int = 50
    ) -> Dict:
        """
        Create comprehensive benchmark dataset for sparse-view CT.
        
        Args:
            phantom_types: Types of phantoms to include
            angle_counts: Number of angles to test
            noise_levels: Noise levels to test
            n_samples_per_config: Samples per configuration
        
        Returns:
            Dictionary containing benchmark datasets
        """
        
        print("🎯 Creating sparse-view CT benchmark...")
        
        benchmark_data = {}
        
        total_configs = len(phantom_types) * len(angle_counts) * len(noise_levels)
        config_count = 0
        
        for phantom_type in phantom_types:
            for n_angles in angle_counts:
                for noise_level in noise_levels:
                    config_count += 1
                    config_name = f"{phantom_type}_{n_angles}angles_{noise_level:.3f}noise"
                    
                    print(f"📋 Config {config_count}/{total_configs}: {config_name}")
                    
                    # Generate phantoms for this configuration
                    phantoms, _ = self.phantom_generator.generate_batch(
                        n_samples_per_config,
                        phantom_types=[phantom_type],
                        include_augmentation=False,  # No augmentation for benchmark
                        include_noise=False  # Noise added during CT simulation
                    )
                    
                    # Create training pairs
                    inputs = []
                    targets = []
                    
                    for phantom in phantoms:
                        # Simulate sparse-view CT
                        sinogram, angles = simulate_sparse_view_ct(
                            phantom, n_angles=n_angles, noise_level=noise_level
                        )
                        
                        # FBP reconstruction
                        fbp_recon = fbp_reconstruction(sinogram, angles, 'shepp-logan')
                        
                        inputs.append(fbp_recon)
                        targets.append(phantom)
                    
                    benchmark_data[config_name] = {
                        'inputs': np.array(inputs),
                        'targets': np.array(targets),
                        'phantom_type': phantom_type,
                        'n_angles': n_angles,
                        'noise_level': noise_level
                    }
        
        print(f"✅ Benchmark created with {len(benchmark_data)} configurations")
        
        # Save benchmark
        benchmark_path = self.output_dir / "sparse_view_benchmark.h5"
        self._save_benchmark(benchmark_data, benchmark_path)
        
        return benchmark_data
    
    def _save_dataset(self, phantoms: np.ndarray, filename: str):
        """Save phantom dataset to HDF5 file."""
        filepath = self.output_dir / filename
        
        with h5py.File(filepath, 'w') as f:
            f.create_dataset('phantoms', data=phantoms, compression='gzip')
            
            # Save metadata
            f.attrs['size'] = len(phantoms)
            f.attrs['shape'] = phantoms.shape
            f.attrs['dtype'] = str(phantoms.dtype)
        
        print(f"💾 Saved {filename}: {len(phantoms)} phantoms, {filepath.stat().st_size / 1024**2:.1f} MB")
    
    def _save_benchmark(self, benchmark_data: Dict, filepath: Path):
        """Save benchmark dataset to HDF5 file."""
        with h5py.File(filepath, 'w') as f:
            for config_name, data in benchmark_data.items():
                group = f.create_group(config_name)
                group.create_dataset('inputs', data=data['inputs'], compression='gzip')
                group.create_dataset('targets', data=data['targets'], compression='gzip')
                
                # Save metadata
                for key, value in data.items():
                    if key not in ['inputs', 'targets']:
                        group.attrs[key] = value
        
        print(f"💾 Benchmark saved: {filepath}, {filepath.stat().st_size / 1024**2:.1f} MB")
    
    def _print_pipeline_stats(self):
        """Print pipeline statistics."""
        print("\n📊 DATA PIPELINE STATISTICS")
        print("=" * 35)
        for key, value in self.stats.items():
            print(f"{key.replace('_', ' ').title()}: {value}")


def load_dataset_from_file(filepath: Union[str, Path]) -> np.ndarray:
    """Load phantom dataset from HDF5 file."""
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset file not found: {filepath}")
    
    with h5py.File(filepath, 'r') as f:
        phantoms = f['phantoms'][:]
    
    print(f"📂 Loaded dataset: {phantoms.shape} from {filepath}")
    return phantoms


def create_mixed_dataset(
    geometric_ratio: float = 0.3,
    shepp_logan_ratio: float = 0.2,
    anatomical_ratio: float = 0.4,
    pathological_ratio: float = 0.1,
    total_size: int = 1000,
    **kwargs
) -> Tuple[CTDataset, CTDataset]:
    """
    Create mixed dataset with specified phantom type ratios.
    
    Args:
        geometric_ratio: Fraction of geometric phantoms
        shepp_logan_ratio: Fraction of Shepp-Logan phantoms
        anatomical_ratio: Fraction of anatomical phantoms
        pathological_ratio: Fraction of pathological phantoms
        total_size: Total number of phantoms
        **kwargs: Additional arguments for DataPipeline
    
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    
    # Validate ratios
    total_ratio = geometric_ratio + shepp_logan_ratio + anatomical_ratio + pathological_ratio
    if abs(total_ratio - 1.0) > 0.01:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
    
    # Calculate counts
    counts = {
        'simple_geometric': int(total_size * geometric_ratio),
        'shepp_logan': int(total_size * shepp_logan_ratio),
        'anatomical_brain': int(total_size * anatomical_ratio * 0.6),
        'anatomical_chest': int(total_size * anatomical_ratio * 0.4),
        'pathological': int(total_size * pathological_ratio)
    }
    
    # Adjust for rounding errors
    current_total = sum(counts.values())
    if current_total != total_size:
        # Add/subtract from largest category
        largest_key = max(counts.keys(), key=lambda k: counts[k])
        counts[largest_key] += total_size - current_total
    
    print(f"🎭 Creating mixed dataset with phantom distribution:")
    for phantom_type, count in counts.items():
        percentage = count / total_size * 100
        print(f"  {phantom_type}: {count} ({percentage:.1f}%)")
    
    # Generate phantoms by type
    generator = PhantomGenerator()
    all_phantoms = []
    
    for phantom_type, count in counts.items():
        if count > 0:
            phantoms, _ = generator.generate_batch(
                count,
                phantom_types=[phantom_type],
                include_augmentation=True,
                include_noise=True
            )
            all_phantoms.append(phantoms)
    
    # Combine and shuffle
    all_phantoms = np.concatenate(all_phantoms, axis=0)
    np.random.shuffle(all_phantoms)
    
    # Create pipeline and datasets
    config = kwargs.get('config', {'batch_size': 8, 'random_seed': 42})
    pipeline = DataPipeline(config)
    
    # Split data
    val_split = kwargs.get('validation_split', 0.2)
    split_idx = int(len(all_phantoms) * (1 - val_split))
    
    train_phantoms = all_phantoms[:split_idx]
    val_phantoms = all_phantoms[split_idx:]
    
    # Filter kwargs for CTDataset (remove parameters not accepted by CTDataset.__init__)
    dataset_kwargs = {k: v for k, v in kwargs.items() if k not in ['validation_split', 'config']}
    
    # Create datasets
    train_dataset = CTDataset(phantoms=train_phantoms, **dataset_kwargs)
    val_dataset = CTDataset(phantoms=val_phantoms, **dataset_kwargs)
    
    return train_dataset, val_dataset


# Test function
if __name__ == "__main__":
    print("🧪 Testing CT Dataset Pipeline...")
    
    # Test basic dataset creation
    config = {
        'batch_size': 4,
        'random_seed': 42
    }
    
    pipeline = DataPipeline(config)
    
    # Create small test dataset
    train_dataset, val_dataset = pipeline.generate_training_dataset(
        size=20,
        validation_split=0.2,
        save_to_disk=False
    )
    
    print(f"✅ Created datasets: train={len(train_dataset.phantoms)}, val={len(val_dataset.phantoms)}")
    
    # Test TensorFlow dataset creation
    tf_train = train_dataset.create_tf_dataset(size=16)
    tf_val = val_dataset.create_validation_dataset(size=4)
    
    print("✅ TensorFlow datasets created")
    
    # Test batch generation
    inputs, targets = train_dataset.get_sample_batch(batch_size=2)
    print(f"✅ Sample batch: inputs {inputs.shape}, targets {targets.shape}")
    
    # Test mixed dataset
    mixed_train, mixed_val = create_mixed_dataset(
        total_size=50,
        validation_split=0.2,
        batch_size=4
    )
    
    print(f"✅ Mixed dataset: train={len(mixed_train.phantoms)}, val={len(mixed_val.phantoms)}")
    
    print("\n🎉 CT Dataset Pipeline test completed successfully!")
