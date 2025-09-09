"""
PhantomGenerator: Advanced Dataset Generation for CT Reconstruction

This module creates diverse, realistic phantoms for training robust CT reconstruction models.
It generates multiple types of phantoms with varying complexity, noise, and pathologies
to ensure comprehensive training coverage.

Key Features:
1. Multiple phantom types (geometric, anatomical, pathological)
2. Realistic noise modeling (Gaussian, Poisson, artifact simulation)
3. Automatic data augmentation
4. Sparse-view protocol simulation
5. Quality control and validation

Author: Your Name
Date: September 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional, Union
import warnings
from pathlib import Path
import h5py
from scipy import ndimage
from skimage import morphology, filters, draw
from skimage.transform import resize, rotate
import cv2

# For progress tracking
from tqdm import tqdm


class PhantomGenerator:
    """
    Advanced phantom generator for CT reconstruction training.
    
    This class creates diverse phantoms including:
    - Simple geometric shapes (circles, ellipses, rectangles)
    - Shepp-Logan phantom variants
    - Anatomical phantoms (brain, chest, abdomen simulations)
    - Pathological cases (tumors, lesions, fractures)
    
    Features:
    - Automatic quality control
    - Realistic noise modeling
    - Data augmentation pipeline
    - Sparse-view simulation
    - Batch generation for efficiency
    
    Args:
        phantom_size: Size of generated phantoms (height, width)
        phantom_types: List of phantom types to generate
        noise_models: Noise characteristics for different scenarios
        augmentation_config: Data augmentation parameters
        quality_threshold: Minimum quality threshold for generated phantoms
    """
    
    def __init__(
        self,
        phantom_size: Tuple[int, int] = (256, 256),
        phantom_types: Optional[List[str]] = None,
        noise_models: Optional[Dict] = None,
        augmentation_config: Optional[Dict] = None,
        quality_threshold: float = 0.8,
        random_seed: int = 42
    ):
        self.phantom_size = phantom_size
        self.quality_threshold = quality_threshold
        self.random_seed = random_seed
        
        # Set random seeds for reproducibility
        np.random.seed(random_seed)
        
        # Default phantom types
        if phantom_types is None:
            self.phantom_types = [
                'simple_geometric',
                'shepp_logan',
                'anatomical_brain',
                'anatomical_chest', 
                'pathological'
            ]
        else:
            self.phantom_types = phantom_types
        
        # Default noise models
        if noise_models is None:
            self.noise_models = {
                'gaussian': {'std_range': (0.01, 0.15)},
                'poisson': {'peak_range': (100, 1000)},
                'salt_pepper': {'prob_range': (0.001, 0.01)},
                'speckle': {'var_range': (0.01, 0.1)}
            }
        else:
            self.noise_models = noise_models
        
        # Default augmentation config
        if augmentation_config is None:
            self.augmentation_config = {
                'rotation_range': 15,
                'zoom_range': 0.1,
                'shift_range': 0.05,
                'intensity_range': 0.1,
                'elastic_deformation': True,
                'flip_probability': 0.2
            }
        else:
            self.augmentation_config = augmentation_config
        
        # Initialize phantom statistics
        self.generation_stats = {
            'total_generated': 0,
            'quality_rejected': 0,
            'type_distribution': {ptype: 0 for ptype in self.phantom_types}
        }
    
    def create_simple_geometric_phantom(
        self, 
        complexity: str = 'medium'
    ) -> np.ndarray:
        """
        Create phantom with geometric shapes (circles, ellipses, rectangles).
        
        Args:
            complexity: Complexity level ('simple', 'medium', 'complex')
        
        Returns:
            Generated phantom as numpy array
        """
        phantom = np.zeros(self.phantom_size, dtype=np.float32)
        height, width = self.phantom_size
        
        # Define complexity levels
        complexity_config = {
            'simple': {'n_shapes': (2, 4), 'overlap_prob': 0.1},
            'medium': {'n_shapes': (3, 6), 'overlap_prob': 0.3},
            'complex': {'n_shapes': (5, 10), 'overlap_prob': 0.5}
        }
        
        config = complexity_config[complexity]
        n_shapes = np.random.randint(*config['n_shapes'])
        
        for _ in range(n_shapes):
            shape_type = np.random.choice(['circle', 'ellipse', 'rectangle'])
            intensity = np.random.uniform(0.2, 1.0)
            
            if shape_type == 'circle':
                self._add_circle(phantom, intensity)
            elif shape_type == 'ellipse':
                self._add_ellipse(phantom, intensity)
            else:  # rectangle
                self._add_rectangle(phantom, intensity)
        
        return phantom
    
    def create_shepp_logan_phantom(
        self, 
        variant: str = 'modified'
    ) -> np.ndarray:
        """
        Create Shepp-Logan phantom with variants.
        
        Args:
            variant: Phantom variant ('original', 'modified', 'custom')
        
        Returns:
            Generated Shepp-Logan phantom
        """
        # Shepp-Logan ellipse parameters
        # Format: [A, a, b, x0, y0, phi]
        # A: amplitude, a/b: semi-axes, x0/y0: center, phi: angle
        
        if variant == 'original':
            ellipses = [
                [1, 0.69, 0.92, 0, 0, 0],
                [-0.8, 0.6624, 0.8740, 0, -0.0184, 0],
                [-0.2, 0.1100, 0.3100, 0.22, 0, -18],
                [-0.2, 0.1600, 0.4100, -0.22, 0, 18],
                [0.1, 0.2100, 0.2500, 0, 0.35, 0],
                [0.1, 0.0460, 0.0460, 0, 0.1, 0],
                [0.1, 0.0460, 0.0460, 0, -0.1, 0],
                [0.1, 0.0460, 0.0230, -0.08, -0.605, 0],
                [0.1, 0.0230, 0.0230, 0, -0.606, 0],
                [0.1, 0.0230, 0.0460, 0.06, -0.605, 0]
            ]
        elif variant == 'modified':
            # Modified Shepp-Logan with higher contrast
            ellipses = [
                [1, 0.69, 0.92, 0, 0, 0],
                [-0.98, 0.6624, 0.8740, 0, -0.0184, 0],
                [-0.02, 0.1100, 0.3100, 0.22, 0, -18],
                [-0.02, 0.1600, 0.4100, -0.22, 0, 18],
                [0.01, 0.2100, 0.2500, 0, 0.35, 0],
                [0.01, 0.0460, 0.0460, 0, 0.1, 0],
                [0.01, 0.0460, 0.0460, 0, -0.1, 0],
                [0.01, 0.0460, 0.0230, -0.08, -0.605, 0],
                [0.01, 0.0230, 0.0230, 0, -0.606, 0],
                [0.01, 0.0230, 0.0460, 0.06, -0.605, 0]
            ]
        else:  # custom variant
            ellipses = self._generate_custom_ellipses()
        
        return self._create_ellipse_phantom(ellipses)
    
    def create_anatomical_phantom(
        self, 
        anatomy_type: str = 'brain'
    ) -> np.ndarray:
        """
        Create anatomically-inspired phantoms.
        
        Args:
            anatomy_type: Type of anatomy ('brain', 'chest', 'abdomen')
        
        Returns:
            Generated anatomical phantom
        """
        if anatomy_type == 'brain':
            return self._create_brain_phantom()
        elif anatomy_type == 'chest':
            return self._create_chest_phantom()
        elif anatomy_type == 'abdomen':
            return self._create_abdomen_phantom()
        else:
            raise ValueError(f"Unknown anatomy type: {anatomy_type}")
    
    def create_pathological_phantom(
        self, 
        base_phantom: Optional[np.ndarray] = None,
        pathology_type: str = 'tumor'
    ) -> np.ndarray:
        """
        Create phantom with pathological features.
        
        Args:
            base_phantom: Base phantom to add pathology to
            pathology_type: Type of pathology ('tumor', 'lesion', 'fracture')
        
        Returns:
            Phantom with pathological features
        """
        if base_phantom is None:
            base_phantom = self.create_anatomical_phantom('brain')
        
        phantom = base_phantom.copy()
        
        if pathology_type == 'tumor':
            self._add_tumor(phantom)
        elif pathology_type == 'lesion':
            self._add_lesion(phantom)
        elif pathology_type == 'fracture':
            self._add_fracture(phantom)
        
        return phantom
    
    def add_noise(
        self, 
        phantom: np.ndarray, 
        noise_type: str = 'gaussian',
        noise_level: float = 0.05
    ) -> np.ndarray:
        """
        Add realistic noise to phantom.
        
        Args:
            phantom: Input phantom
            noise_type: Type of noise to add
            noise_level: Intensity of noise
        
        Returns:
            Noisy phantom
        """
        noisy_phantom = phantom.copy()
        
        if noise_type == 'gaussian':
            noise = np.random.normal(0, noise_level, phantom.shape)
            noisy_phantom += noise
            
        elif noise_type == 'poisson':
            # Scale phantom to simulate photon counting
            # Ensure phantom is non-negative and scaling factor is positive
            phantom_safe = np.clip(phantom, 0, None)  # Remove negative values
            scale_factor = max(0.1, 1 - noise_level + 0.1)  # Ensure positive scaling
            scaled = phantom_safe * 1000 * scale_factor
            # Ensure scaled values are valid for Poisson distribution
            scaled = np.clip(scaled, 0.1, None)  # Minimum lambda = 0.1
            noisy = np.random.poisson(scaled) / 1000
            noisy_phantom = noisy
            
        elif noise_type == 'salt_pepper':
            prob = noise_level
            mask = np.random.random(phantom.shape)
            noisy_phantom[mask < prob/2] = 0
            noisy_phantom[mask > 1 - prob/2] = phantom.max()
            
        elif noise_type == 'speckle':
            noise = np.random.normal(0, noise_level, phantom.shape)
            noisy_phantom = phantom + phantom * noise
        
        # Clip to valid range
        return np.clip(noisy_phantom, 0, 1)
    
    def augment_phantom(
        self, 
        phantom: np.ndarray
    ) -> np.ndarray:
        """
        Apply data augmentation to phantom.
        
        Args:
            phantom: Input phantom
        
        Returns:
            Augmented phantom
        """
        augmented = phantom.copy()
        
        # Rotation
        if np.random.random() < 0.5:
            angle = np.random.uniform(-self.augmentation_config['rotation_range'],
                                    self.augmentation_config['rotation_range'])
            augmented = rotate(augmented, angle, preserve_range=True)
        
        # Zoom
        if np.random.random() < 0.3:
            zoom_factor = 1 + np.random.uniform(-self.augmentation_config['zoom_range'],
                                              self.augmentation_config['zoom_range'])
            h, w = augmented.shape
            augmented = resize(augmented, (int(h*zoom_factor), int(w*zoom_factor)))
            
            # Crop or pad to original size
            if augmented.shape[0] > h:
                start = (augmented.shape[0] - h) // 2
                augmented = augmented[start:start+h, start:start+w]
            else:
                pad_h = (h - augmented.shape[0]) // 2
                pad_w = (w - augmented.shape[1]) // 2
                augmented = np.pad(augmented, ((pad_h, h-augmented.shape[0]-pad_h),
                                             (pad_w, w-augmented.shape[1]-pad_w)))
        
        # Intensity variation
        if np.random.random() < 0.4:
            intensity_factor = 1 + np.random.uniform(-self.augmentation_config['intensity_range'],
                                                   self.augmentation_config['intensity_range'])
            augmented *= intensity_factor
            augmented = np.clip(augmented, 0, 1)
        
        # Elastic deformation
        if self.augmentation_config['elastic_deformation'] and np.random.random() < 0.2:
            augmented = self._elastic_deformation(augmented)
        
        return augmented
    
    def generate_batch(
        self, 
        batch_size: int,
        phantom_types: Optional[List[str]] = None,
        include_augmentation: bool = True,
        include_noise: bool = True
    ) -> Tuple[np.ndarray, Dict]:
        """
        Generate a batch of phantoms with metadata.
        
        Args:
            batch_size: Number of phantoms to generate
            phantom_types: Types of phantoms to include
            include_augmentation: Whether to apply augmentation
            include_noise: Whether to add noise
        
        Returns:
            Tuple of (phantom_batch, metadata)
        """
        if phantom_types is None:
            phantom_types = self.phantom_types
        
        phantoms = []
        metadata = []
        
        print(f"🎭 Generating {batch_size} phantoms...")
        
        for i in tqdm(range(batch_size)):
            # Select phantom type
            phantom_type = np.random.choice(phantom_types)
            
            # Generate base phantom
            if phantom_type == 'simple_geometric':
                complexity = np.random.choice(['simple', 'medium', 'complex'])
                phantom = self.create_simple_geometric_phantom(complexity)
                meta = {'type': phantom_type, 'complexity': complexity}
                
            elif phantom_type == 'shepp_logan':
                variant = np.random.choice(['original', 'modified', 'custom'])
                phantom = self.create_shepp_logan_phantom(variant)
                meta = {'type': phantom_type, 'variant': variant}
                
            elif phantom_type.startswith('anatomical'):
                anatomy = phantom_type.split('_')[1]
                phantom = self.create_anatomical_phantom(anatomy)
                meta = {'type': phantom_type, 'anatomy': anatomy}
                
            elif phantom_type == 'pathological':
                base_type = np.random.choice(['brain', 'chest'])
                pathology = np.random.choice(['tumor', 'lesion'])
                base_phantom = self.create_anatomical_phantom(base_type)
                phantom = self.create_pathological_phantom(base_phantom, pathology)
                meta = {'type': phantom_type, 'base': base_type, 'pathology': pathology}
            
            # Apply augmentation
            if include_augmentation and np.random.random() < 0.7:
                phantom = self.augment_phantom(phantom)
                meta['augmented'] = True
            else:
                meta['augmented'] = False
            
            # Add noise
            if include_noise and np.random.random() < 0.8:
                noise_type = np.random.choice(list(self.noise_models.keys()))
                noise_level = np.random.uniform(0.01, 0.1)
                phantom = self.add_noise(phantom, noise_type, noise_level)
                meta['noise_type'] = noise_type
                meta['noise_level'] = noise_level
            else:
                meta['noise_type'] = None
                meta['noise_level'] = 0
            
            # Quality control
            if self._quality_check(phantom):
                phantoms.append(phantom)
                metadata.append(meta)
                self.generation_stats['type_distribution'][phantom_type] += 1
            else:
                self.generation_stats['quality_rejected'] += 1
                # Generate replacement
                i -= 1
                continue
        
        self.generation_stats['total_generated'] += len(phantoms)
        
        return np.array(phantoms), metadata
    
    def save_dataset(
        self, 
        phantoms: np.ndarray,
        metadata: List[Dict],
        save_path: Union[str, Path],
        compression: str = 'gzip'
    ):
        """Save generated dataset to HDF5 file."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(save_path, 'w') as f:
            # Save phantom data
            f.create_dataset('phantoms', data=phantoms, compression=compression)
            
            # Save metadata
            meta_group = f.create_group('metadata')
            for i, meta in enumerate(metadata):
                sample_group = meta_group.create_group(f'sample_{i}')
                for key, value in meta.items():
                    if value is not None:
                        sample_group.attrs[key] = value
            
            # Save generation statistics
            stats_group = f.create_group('statistics')
            for key, value in self.generation_stats.items():
                if isinstance(value, dict):
                    sub_group = stats_group.create_group(key)
                    for sub_key, sub_value in value.items():
                        sub_group.attrs[sub_key] = sub_value
                else:
                    stats_group.attrs[key] = value
        
        print(f"💾 Dataset saved to {save_path}")
        print(f"📊 {len(phantoms)} phantoms, {save_path.stat().st_size / 1024**2:.1f} MB")
    
    # Helper methods (implementation details)
    def _add_circle(self, phantom: np.ndarray, intensity: float):
        """Add a circle to the phantom."""
        h, w = phantom.shape
        center_y = np.random.randint(h//4, 3*h//4)
        center_x = np.random.randint(w//4, 3*w//4)
        radius = np.random.randint(10, min(h, w)//6)
        
        rr, cc = draw.disk((center_y, center_x), radius, shape=phantom.shape)
        phantom[rr, cc] = intensity
    
    def _add_ellipse(self, phantom: np.ndarray, intensity: float):
        """Add an ellipse to the phantom."""
        h, w = phantom.shape
        center_y = np.random.randint(h//4, 3*h//4)
        center_x = np.random.randint(w//4, 3*w//4)
        r_radius = np.random.randint(8, h//8)
        c_radius = np.random.randint(8, w//8)
        
        rr, cc = draw.ellipse(center_y, center_x, r_radius, c_radius, shape=phantom.shape)
        phantom[rr, cc] = intensity
    
    def _add_rectangle(self, phantom: np.ndarray, intensity: float):
        """Add a rectangle to the phantom."""
        h, w = phantom.shape
        start_y = np.random.randint(h//6, 2*h//3)
        start_x = np.random.randint(w//6, 2*w//3)
        height = np.random.randint(10, h//4)
        width = np.random.randint(10, w//4)
        
        end_y = min(start_y + height, h-1)
        end_x = min(start_x + width, w-1)
        
        rr, cc = draw.rectangle((start_y, start_x), (end_y, end_x), shape=phantom.shape)
        phantom[rr, cc] = intensity
    
    def _create_ellipse_phantom(self, ellipses: List) -> np.ndarray:
        """Create phantom from ellipse parameters."""
        phantom = np.zeros(self.phantom_size, dtype=np.float32)
        h, w = self.phantom_size
        
        # Create coordinate grids
        y, x = np.ogrid[:h, :w]
        y = (y - h/2) / (h/2)  # Normalize to [-1, 1]
        x = (x - w/2) / (w/2)  # Normalize to [-1, 1]
        
        for params in ellipses:
            A, a, b, x0, y0, phi = params
            
            # Convert angle to radians
            phi = np.radians(phi)
            
            # Rotation matrix
            cos_phi = np.cos(phi)
            sin_phi = np.sin(phi)
            
            # Rotated coordinates
            x_rot = cos_phi * (x - x0) + sin_phi * (y - y0)
            y_rot = -sin_phi * (x - x0) + cos_phi * (y - y0)
            
            # Ellipse equation
            mask = (x_rot/a)**2 + (y_rot/b)**2 <= 1
            phantom[mask] += A
        
        return phantom
    
    def _create_brain_phantom(self) -> np.ndarray:
        """Create a brain-like phantom."""
        phantom = np.zeros(self.phantom_size, dtype=np.float32)
        h, w = phantom.shape
        
        # Brain outline (skull)
        center_y, center_x = h//2, w//2
        rr, cc = draw.ellipse(center_y, center_x, h//3, w//3, shape=phantom.shape)
        phantom[rr, cc] = 0.2
        
        # Brain tissue
        rr, cc = draw.ellipse(center_y, center_x, h//3-10, w//3-10, shape=phantom.shape)
        phantom[rr, cc] = 0.8
        
        # Ventricles
        rr, cc = draw.ellipse(center_y-20, center_x-10, 15, 25, shape=phantom.shape)
        phantom[rr, cc] = 0.1
        rr, cc = draw.ellipse(center_y-20, center_x+10, 15, 25, shape=phantom.shape)
        phantom[rr, cc] = 0.1
        
        # Add some random structures
        for _ in range(3):
            cy = np.random.randint(center_y-40, center_y+40)
            cx = np.random.randint(center_x-40, center_x+40)
            r = np.random.randint(5, 15)
            rr, cc = draw.disk((cy, cx), r, shape=phantom.shape)
            phantom[rr, cc] = np.random.uniform(0.3, 0.7)
        
        return np.clip(phantom, 0, 1)
    
    def _create_chest_phantom(self) -> np.ndarray:
        """Create a chest-like phantom."""
        phantom = np.zeros(self.phantom_size, dtype=np.float32)
        h, w = phantom.shape
        
        # Chest outline
        rr, cc = draw.ellipse(h//2, w//2, h//3, w//2, shape=phantom.shape)
        phantom[rr, cc] = 0.3
        
        # Lungs
        rr, cc = draw.ellipse(h//2, w//2-30, h//4, w//6, shape=phantom.shape)
        phantom[rr, cc] = 0.1
        rr, cc = draw.ellipse(h//2, w//2+30, h//4, w//6, shape=phantom.shape)
        phantom[rr, cc] = 0.1
        
        # Heart
        rr, cc = draw.ellipse(h//2+10, w//2-10, 20, 15, shape=phantom.shape)
        phantom[rr, cc] = 0.6
        
        # Ribs (simplified)
        for i in range(3):
            y_offset = -30 + i * 30
            rr, cc = draw.ellipse(h//2+y_offset, w//2, 3, w//3, shape=phantom.shape)
            phantom[rr, cc] = 0.8
        
        return np.clip(phantom, 0, 1)
    
    def _create_abdomen_phantom(self) -> np.ndarray:
        """Create an abdomen-like phantom."""
        phantom = np.zeros(self.phantom_size, dtype=np.float32)
        h, w = phantom.shape
        
        # Abdomen outline
        rr, cc = draw.ellipse(h//2, w//2, h//2-20, w//2-10, shape=phantom.shape)
        phantom[rr, cc] = 0.4
        
        # Organs (simplified)
        organs = [
            (h//2-30, w//2-20, 15, 20, 0.7),  # Liver
            (h//2+20, w//2-30, 12, 12, 0.6),  # Kidney
            (h//2+20, w//2+30, 12, 12, 0.6),  # Kidney
            (h//2, w//2, 8, 8, 0.5)           # Other organ
        ]
        
        for cy, cx, ry, rx, intensity in organs:
            rr, cc = draw.ellipse(cy, cx, ry, rx, shape=phantom.shape)
            phantom[rr, cc] = intensity
        
        return np.clip(phantom, 0, 1)
    
    def _add_tumor(self, phantom: np.ndarray):
        """Add tumor-like structure to phantom."""
        h, w = phantom.shape
        # Find region with tissue
        tissue_mask = phantom > 0.5
        if not np.any(tissue_mask):
            return
        
        # Random location in tissue
        tissue_coords = np.where(tissue_mask)
        idx = np.random.randint(len(tissue_coords[0]))
        center_y, center_x = tissue_coords[0][idx], tissue_coords[1][idx]
        
        # Tumor size
        radius = np.random.randint(5, 20)
        rr, cc = draw.disk((center_y, center_x), radius, shape=phantom.shape)
        
        # Tumor intensity (different from surrounding tissue)
        surrounding_intensity = phantom[center_y, center_x]
        tumor_intensity = surrounding_intensity + np.random.uniform(-0.3, 0.3)
        tumor_intensity = np.clip(tumor_intensity, 0, 1)
        
        phantom[rr, cc] = tumor_intensity
    
    def _add_lesion(self, phantom: np.ndarray):
        """Add lesion-like structure to phantom."""
        # Similar to tumor but smaller and irregular
        h, w = phantom.shape
        center_y = np.random.randint(h//4, 3*h//4)
        center_x = np.random.randint(w//4, 3*w//4)
        
        # Irregular lesion
        radius = np.random.randint(3, 10)
        rr, cc = draw.disk((center_y, center_x), radius, shape=phantom.shape)
        
        # Make it irregular
        noise = np.random.random((len(rr),)) > 0.3
        rr, cc = rr[noise], cc[noise]
        
        lesion_intensity = np.random.uniform(0.1, 0.9)
        phantom[rr, cc] = lesion_intensity
    
    def _add_fracture(self, phantom: np.ndarray):
        """Add fracture-like linear structure."""
        h, w = phantom.shape
        
        # Random line parameters
        start_y = np.random.randint(h//4, 3*h//4)
        start_x = np.random.randint(w//4, 3*w//4)
        
        angle = np.random.uniform(0, 2*np.pi)
        length = np.random.randint(20, 60)
        
        end_y = int(start_y + length * np.sin(angle))
        end_x = int(start_x + length * np.cos(angle))
        
        # Ensure endpoints are within bounds
        end_y = np.clip(end_y, 0, h-1)
        end_x = np.clip(end_x, 0, w-1)
        
        rr, cc = draw.line(start_y, start_x, end_y, end_x)
        
        # Fracture appears as dark line
        phantom[rr, cc] = 0.1
    
    def _generate_custom_ellipses(self) -> List:
        """Generate custom ellipse parameters for variant Shepp-Logan."""
        # Random variation of Shepp-Logan parameters
        base_ellipses = [
            [1, 0.69, 0.92, 0, 0, 0],
            [-0.8, 0.6624, 0.8740, 0, -0.0184, 0]
        ]
        
        # Add random ellipses
        n_additional = np.random.randint(3, 8)
        for _ in range(n_additional):
            A = np.random.uniform(-0.3, 0.3)
            a = np.random.uniform(0.02, 0.3)
            b = np.random.uniform(0.02, 0.3)
            x0 = np.random.uniform(-0.5, 0.5)
            y0 = np.random.uniform(-0.5, 0.5)
            phi = np.random.uniform(-90, 90)
            
            base_ellipses.append([A, a, b, x0, y0, phi])
        
        return base_ellipses
    
    def _elastic_deformation(self, phantom: np.ndarray) -> np.ndarray:
        """Apply elastic deformation to phantom."""
        # Simple elastic deformation using random displacement fields
        h, w = phantom.shape
        
        # Generate random displacement fields
        dx = np.random.uniform(-5, 5, (h//8, w//8))
        dy = np.random.uniform(-5, 5, (h//8, w//8))
        
        # Upsample displacement fields
        dx = resize(dx, (h, w))
        dy = resize(dy, (h, w))
        
        # Apply deformation
        y, x = np.mgrid[:h, :w]
        new_y = np.clip(y + dy, 0, h-1).astype(int)
        new_x = np.clip(x + dx, 0, w-1).astype(int)
        
        deformed = phantom[new_y, new_x]
        return deformed
    
    def _quality_check(self, phantom: np.ndarray) -> bool:
        """Check if generated phantom meets quality criteria."""
        # Basic quality checks
        
        # Check for reasonable intensity range (allow small negative values due to precision)
        if phantom.max() < 0.1 or phantom.min() < -0.01:
            return False
        
        # Check for sufficient contrast
        if np.std(phantom) < 0.05:
            return False
        
        # Check for reasonable amount of non-zero content (more lenient for anatomical phantoms)
        non_zero_ratio = np.sum(phantom > 0.01) / phantom.size
        if non_zero_ratio < 0.05 or non_zero_ratio > 0.98:
            return False
        
        return True
    
    def print_statistics(self):
        """Print generation statistics."""
        print("\n📊 PHANTOM GENERATION STATISTICS")
        print("=" * 40)
        print(f"Total Generated: {self.generation_stats['total_generated']}")
        print(f"Quality Rejected: {self.generation_stats['quality_rejected']}")
        
        print("\nType Distribution:")
        for ptype, count in self.generation_stats['type_distribution'].items():
            percentage = count / max(self.generation_stats['total_generated'], 1) * 100
            print(f"  {ptype}: {count} ({percentage:.1f}%)")


# Factory function for easy usage
def create_phantom_dataset(
    size: int = 1000,
    phantom_size: Tuple[int, int] = (256, 256),
    save_path: Optional[Union[str, Path]] = None,
    **kwargs
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Factory function to create a complete phantom dataset.
    
    Args:
        size: Number of phantoms to generate
        phantom_size: Size of each phantom
        save_path: Optional path to save dataset
        **kwargs: Additional arguments for PhantomGenerator
    
    Returns:
        Tuple of (phantoms, metadata)
    """
    generator = PhantomGenerator(phantom_size=phantom_size, **kwargs)
    phantoms, metadata = generator.generate_batch(size)
    
    if save_path:
        generator.save_dataset(phantoms, metadata, save_path)
    
    generator.print_statistics()
    
    return phantoms, metadata


if __name__ == "__main__":
    # Test the phantom generator
    print("🧪 Testing PhantomGenerator...")
    
    generator = PhantomGenerator(phantom_size=(256, 256))
    
    # Test individual phantom types
    simple_phantom = generator.create_simple_geometric_phantom('medium')
    shepp_phantom = generator.create_shepp_logan_phantom('modified')
    brain_phantom = generator.create_anatomical_phantom('brain')
    
    print(f"✅ Simple phantom: {simple_phantom.shape}, range: [{simple_phantom.min():.3f}, {simple_phantom.max():.3f}]")
    print(f"✅ Shepp phantom: {shepp_phantom.shape}, range: [{shepp_phantom.min():.3f}, {shepp_phantom.max():.3f}]")
    print(f"✅ Brain phantom: {brain_phantom.shape}, range: [{brain_phantom.min():.3f}, {brain_phantom.max():.3f}]")
    
    # Test batch generation
    phantoms, metadata = generator.generate_batch(10)
    print(f"✅ Batch generation: {phantoms.shape}")
    
    # Test with noise and augmentation
    noisy_phantom = generator.add_noise(simple_phantom, 'gaussian', 0.1)
    augmented_phantom = generator.augment_phantom(simple_phantom)
    
    print(f"✅ Noise addition: {noisy_phantom.shape}")
    print(f"✅ Augmentation: {augmented_phantom.shape}")
    
    generator.print_statistics()
    
    print("\n🎉 PhantomGenerator test completed successfully!")
