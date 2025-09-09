"""
CT Transforms: Radon Transform and Reconstruction Algorithms

This module implements the core mathematical transforms used in CT imaging:
1. Radon Transform (forward projection)
2. Inverse Radon Transform (filtered backprojection)
3. Classical reconstruction algorithms (FBP with various filters)
4. Optimized implementations for neural network integration

These implementations are crucial for:
- Physics-informed loss functions
- Classical baseline comparisons
- Sparse-view CT simulation
- Training data generation

Author: Your Name
Date: September 2024
"""

import numpy as np
import tensorflow as tf
from typing import Union, List, Optional, Tuple, Dict
from scipy import ndimage
from scipy.fft import fft, ifft, fftfreq
from skimage.transform import radon, iradon
import warnings
warnings.filterwarnings('ignore')


class CTTransforms:
    """
    Comprehensive CT transform implementations.
    
    This class provides both NumPy and TensorFlow implementations of CT transforms
    for maximum flexibility and performance.
    """
    
    @staticmethod
    def radon_transform(
        image: Union[np.ndarray, tf.Tensor],
        angles: Union[np.ndarray, tf.Tensor, List[float]],
        circle: bool = True,
        implementation: str = 'skimage'
    ) -> Union[np.ndarray, tf.Tensor]:
        """
        Compute Radon transform (forward projection) of an image.
        
        Args:
            image: Input image(s). Shape: (batch, height, width) or (height, width)
            angles: Projection angles in degrees. Shape: (n_angles,)
            circle: Whether to assume circular object support
            implementation: Implementation to use ('skimage', 'tensorflow', 'custom')
        
        Returns:
            Sinogram(s). Shape: (batch, n_detectors, n_angles) or (n_detectors, n_angles)
        """
        
        if implementation == 'skimage':
            return CTTransforms._radon_skimage(image, angles, circle)
        elif implementation == 'tensorflow':
            return CTTransforms._radon_tensorflow(image, angles, circle)
        elif implementation == 'custom':
            return CTTransforms._radon_custom(image, angles, circle)
        else:
            raise ValueError(f"Unknown implementation: {implementation}")
    
    @staticmethod
    def iradon_transform(
        sinogram: Union[np.ndarray, tf.Tensor],
        angles: Union[np.ndarray, tf.Tensor, List[float]],
        filter_name: str = 'ramp',
        circle: bool = True,
        implementation: str = 'skimage'
    ) -> Union[np.ndarray, tf.Tensor]:
        """
        Compute inverse Radon transform (filtered backprojection).
        
        Args:
            sinogram: Input sinogram(s). Shape: (batch, n_detectors, n_angles) or (n_detectors, n_angles)
            angles: Projection angles in degrees. Shape: (n_angles,)
            filter_name: Reconstruction filter ('ramp', 'shepp-logan', 'hamming', 'hann', 'cosine')
            circle: Whether to assume circular object support
            implementation: Implementation to use ('skimage', 'tensorflow', 'custom')
        
        Returns:
            Reconstructed image(s). Shape: (batch, height, width) or (height, width)
        """
        
        if implementation == 'skimage':
            return CTTransforms._iradon_skimage(sinogram, angles, filter_name, circle)
        elif implementation == 'tensorflow':
            return CTTransforms._iradon_tensorflow(sinogram, angles, filter_name, circle)
        elif implementation == 'custom':
            return CTTransforms._iradon_custom(sinogram, angles, filter_name, circle)
        else:
            raise ValueError(f"Unknown implementation: {implementation}")
    
    @staticmethod
    def _radon_skimage(
        image: Union[np.ndarray, tf.Tensor],
        angles: Union[np.ndarray, List[float]],
        circle: bool = True
    ) -> np.ndarray:
        """Radon transform using scikit-image."""
        
        # Convert TensorFlow tensors to NumPy
        if isinstance(image, tf.Tensor):
            image = image.numpy()
        if isinstance(angles, tf.Tensor):
            angles = angles.numpy()
        
        # Handle batch dimension
        if image.ndim == 3:
            # Batch of images
            batch_size = image.shape[0]
            sinograms = []
            
            for i in range(batch_size):
                sino = radon(image[i], theta=angles, circle=circle)
                sinograms.append(sino)
            
            return np.array(sinograms)
        else:
            # Single image
            return radon(image, theta=angles, circle=circle)
    
    @staticmethod
    def _iradon_skimage(
        sinogram: Union[np.ndarray, tf.Tensor],
        angles: Union[np.ndarray, List[float]],
        filter_name: str = 'ramp',
        circle: bool = True
    ) -> np.ndarray:
        """Inverse Radon transform using scikit-image."""
        
        # Convert TensorFlow tensors to NumPy
        if isinstance(sinogram, tf.Tensor):
            sinogram = sinogram.numpy()
        if isinstance(angles, tf.Tensor):
            angles = angles.numpy()
        
        # Handle batch dimension
        if sinogram.ndim == 3:
            # Batch of sinograms
            batch_size = sinogram.shape[0]
            reconstructions = []
            
            for i in range(batch_size):
                recon = iradon(sinogram[i], theta=angles, filter_name=filter_name, circle=circle)
                reconstructions.append(recon)
            
            return np.array(reconstructions)
        else:
            # Single sinogram
            return iradon(sinogram, theta=angles, filter_name=filter_name, circle=circle)
    
    @staticmethod
    def _radon_tensorflow(
        image: tf.Tensor,
        angles: tf.Tensor,
        circle: bool = True
    ) -> tf.Tensor:
        """
        TensorFlow implementation of Radon transform.
        
        This is a simplified implementation for demonstration.
        In practice, you might want to use a more sophisticated approach
        or integrate with libraries like ASTRA or TomoPy.
        """
        
        # This is a placeholder implementation
        # A full TensorFlow implementation would require custom operations
        # or using tf.py_function to call NumPy implementations
        
        def radon_py_function(img, ang):
            return CTTransforms._radon_skimage(img.numpy(), ang.numpy(), circle)
        
        # Use tf.py_function to wrap NumPy implementation
        sinogram = tf.py_function(
            radon_py_function,
            [image, angles],
            tf.float32
        )
        
        # Set shape information
        if len(image.shape) == 3:
            batch_size = image.shape[0]
            n_angles = len(angles)
            n_detectors = image.shape[1]  # Assuming square images
            sinogram.set_shape([batch_size, n_detectors, n_angles])
        else:
            n_angles = len(angles)
            n_detectors = image.shape[0]
            sinogram.set_shape([n_detectors, n_angles])
        
        return sinogram
    
    @staticmethod
    def _iradon_tensorflow(
        sinogram: tf.Tensor,
        angles: tf.Tensor,
        filter_name: str = 'ramp',
        circle: bool = True
    ) -> tf.Tensor:
        """TensorFlow implementation of inverse Radon transform."""
        
        def iradon_py_function(sino, ang):
            return CTTransforms._iradon_skimage(sino.numpy(), ang.numpy(), filter_name, circle)
        
        # Use tf.py_function to wrap NumPy implementation
        reconstruction = tf.py_function(
            iradon_py_function,
            [sinogram, angles],
            tf.float32
        )
        
        # Set shape information
        if len(sinogram.shape) == 3:
            batch_size = sinogram.shape[0]
            img_size = sinogram.shape[1]  # Assuming square reconstruction
            reconstruction.set_shape([batch_size, img_size, img_size])
        else:
            img_size = sinogram.shape[0]
            reconstruction.set_shape([img_size, img_size])
        
        return reconstruction
    
    @staticmethod
    def _radon_custom(
        image: Union[np.ndarray, tf.Tensor],
        angles: Union[np.ndarray, List[float]],
        circle: bool = True
    ) -> Union[np.ndarray, tf.Tensor]:
        """
        Custom implementation of Radon transform.
        
        This is a simplified implementation for educational purposes.
        """
        
        # Convert to NumPy for computation
        if isinstance(image, tf.Tensor):
            image_np = image.numpy()
            return_tf = True
        else:
            image_np = image
            return_tf = False
        
        if isinstance(angles, tf.Tensor):
            angles_np = angles.numpy()
        else:
            angles_np = np.array(angles)
        
        # Handle batch dimension
        if image_np.ndim == 3:
            batch_size = image_np.shape[0]
            sinograms = []
            
            for i in range(batch_size):
                sino = CTTransforms._radon_single_custom(image_np[i], angles_np, circle)
                sinograms.append(sino)
            
            result = np.array(sinograms)
        else:
            result = CTTransforms._radon_single_custom(image_np, angles_np, circle)
        
        return tf.constant(result, dtype=tf.float32) if return_tf else result
    
    @staticmethod
    def _radon_single_custom(
        image: np.ndarray,
        angles: np.ndarray,
        circle: bool = True
    ) -> np.ndarray:
        """Custom Radon transform for a single image."""
        
        h, w = image.shape
        diagonal = int(np.sqrt(h**2 + w**2))
        
        # Pad image to handle rotation
        pad_width = (diagonal - h) // 2, (diagonal - w) // 2
        padded_image = np.pad(image, (pad_width, pad_width), mode='constant')
        
        n_detectors = diagonal
        n_angles = len(angles)
        sinogram = np.zeros((n_detectors, n_angles))
        
        center = diagonal // 2
        
        for i, angle in enumerate(angles):
            # Rotate image
            rotated = ndimage.rotate(padded_image, -angle, reshape=False, order=1)
            
            # Sum along columns (project along rows)
            projection = np.sum(rotated, axis=0)
            
            # Handle different sizes
            if len(projection) > n_detectors:
                start = (len(projection) - n_detectors) // 2
                projection = projection[start:start + n_detectors]
            elif len(projection) < n_detectors:
                pad = (n_detectors - len(projection)) // 2
                projection = np.pad(projection, (pad, n_detectors - len(projection) - pad))
            
            sinogram[:, i] = projection
        
        # Apply circular mask if requested
        if circle:
            # Create circular mask for detector array
            detector_indices = np.arange(n_detectors) - center
            max_radius = min(h, w) // 2
            mask = np.abs(detector_indices) <= max_radius
            sinogram[~mask, :] = 0
        
        return sinogram
    
    @staticmethod
    def _iradon_custom(
        sinogram: Union[np.ndarray, tf.Tensor],
        angles: Union[np.ndarray, List[float]],
        filter_name: str = 'ramp',
        circle: bool = True
    ) -> Union[np.ndarray, tf.Tensor]:
        """Custom implementation of inverse Radon transform (FBP)."""
        
        # Convert to NumPy for computation
        if isinstance(sinogram, tf.Tensor):
            sinogram_np = sinogram.numpy()
            return_tf = True
        else:
            sinogram_np = sinogram
            return_tf = False
        
        if isinstance(angles, tf.Tensor):
            angles_np = angles.numpy()
        else:
            angles_np = np.array(angles)
        
        # Handle batch dimension
        if sinogram_np.ndim == 3:
            batch_size = sinogram_np.shape[0]
            reconstructions = []
            
            for i in range(batch_size):
                recon = CTTransforms._iradon_single_custom(
                    sinogram_np[i], angles_np, filter_name, circle
                )
                reconstructions.append(recon)
            
            result = np.array(reconstructions)
        else:
            result = CTTransforms._iradon_single_custom(
                sinogram_np, angles_np, filter_name, circle
            )
        
        return tf.constant(result, dtype=tf.float32) if return_tf else result
    
    @staticmethod
    def _iradon_single_custom(
        sinogram: np.ndarray,
        angles: np.ndarray,
        filter_name: str = 'ramp',
        circle: bool = True
    ) -> np.ndarray:
        """Custom FBP reconstruction for a single sinogram."""
        
        n_detectors, n_angles = sinogram.shape
        
        # Apply reconstruction filter
        filtered_sinogram = CTTransforms._apply_filter(sinogram, filter_name)
        
        # Initialize reconstruction
        reconstruction = np.zeros((n_detectors, n_detectors))
        
        # Backprojection
        center = n_detectors // 2
        y, x = np.ogrid[:n_detectors, :n_detectors]
        y = y - center
        x = x - center
        
        for i, angle in enumerate(angles):
            # Convert angle to radians
            theta = np.radians(angle)
            
            # Compute projection coordinates
            t = x * np.cos(theta) + y * np.sin(theta)
            
            # Interpolate projection values
            t_indices = t + center
            
            # Clip to valid range
            valid_mask = (t_indices >= 0) & (t_indices < n_detectors)
            
            # Linear interpolation
            t_floor = np.floor(t_indices).astype(int)
            t_ceil = np.ceil(t_indices).astype(int)
            
            # Ensure indices are within bounds
            t_floor = np.clip(t_floor, 0, n_detectors - 1)
            t_ceil = np.clip(t_ceil, 0, n_detectors - 1)
            
            # Interpolation weights
            alpha = t_indices - t_floor
            
            # Interpolated values
            values = (1 - alpha) * filtered_sinogram[t_floor, i] + alpha * filtered_sinogram[t_ceil, i]
            values[~valid_mask] = 0
            
            reconstruction += values
        
        # Normalize by number of angles
        reconstruction /= len(angles)
        
        # Apply circular mask if requested
        if circle:
            radius = n_detectors // 2
            mask = (y**2 + x**2) <= radius**2
            reconstruction[~mask] = 0
        
        return reconstruction
    
    @staticmethod
    def _apply_filter(sinogram: np.ndarray, filter_name: str) -> np.ndarray:
        """Apply reconstruction filter to sinogram."""
        
        n_detectors = sinogram.shape[0]
        
        # Create frequency array
        freqs = fftfreq(n_detectors, 1.0)
        
        # Create filter
        if filter_name == 'ramp' or filter_name == 'ram-lak':
            # Ideal ramp filter
            filt = np.abs(freqs)
        elif filter_name == 'shepp-logan':
            # Shepp-Logan filter
            filt = np.abs(freqs) * np.sinc(freqs / (2 * np.max(np.abs(freqs))))
        elif filter_name == 'hamming':
            # Hamming windowed ramp filter
            window = np.hamming(n_detectors)
            window = np.abs(np.fft.fftshift(window))
            filt = np.abs(freqs) * window
        elif filter_name == 'hann':
            # Hann windowed ramp filter
            window = np.hanning(n_detectors)
            window = np.abs(np.fft.fftshift(window))
            filt = np.abs(freqs) * window
        elif filter_name == 'cosine':
            # Cosine filter
            filt = np.abs(freqs) * np.cos(np.pi * freqs / (2 * np.max(np.abs(freqs))))
        else:
            raise ValueError(f"Unknown filter: {filter_name}")
        
        # Apply filter in frequency domain
        filtered_sinogram = np.zeros_like(sinogram)
        
        for i in range(sinogram.shape[1]):
            projection = sinogram[:, i]
            
            # Apply filter
            projection_fft = fft(projection)
            filtered_fft = projection_fft * filt
            filtered_projection = np.real(ifft(filtered_fft))
            
            filtered_sinogram[:, i] = filtered_projection
        
        return filtered_sinogram


# Convenience functions for easy usage
def radon_transform(
    image: Union[np.ndarray, tf.Tensor],
    angles: Union[np.ndarray, tf.Tensor, List[float]],
    circle: bool = True
) -> Union[np.ndarray, tf.Tensor]:
    """
    Compute Radon transform of an image.
    
    Convenience function that automatically selects the best implementation.
    """
    return CTTransforms.radon_transform(image, angles, circle, implementation='skimage')


def iradon_transform(
    sinogram: Union[np.ndarray, tf.Tensor],
    angles: Union[np.ndarray, tf.Tensor, List[float]],
    filter_name: str = 'ramp',
    circle: bool = True
) -> Union[np.ndarray, tf.Tensor]:
    """
    Compute inverse Radon transform (filtered backprojection).
    
    Convenience function that automatically selects the best implementation.
    """
    return CTTransforms.iradon_transform(sinogram, angles, filter_name, circle, implementation='skimage')


def fbp_reconstruction(
    sinogram: Union[np.ndarray, tf.Tensor],
    angles: Union[np.ndarray, tf.Tensor, List[float]],
    filter_name: str = 'shepp-logan',
    circle: bool = True
) -> Union[np.ndarray, tf.Tensor]:
    """
    Perform filtered backprojection reconstruction.
    
    This is an alias for iradon_transform with sensible defaults.
    """
    return iradon_transform(sinogram, angles, filter_name, circle)


def simulate_sparse_view_ct(
    phantom: Union[np.ndarray, tf.Tensor],
    n_angles: int = 45,
    noise_level: float = 0.05,
    angle_range: Tuple[float, float] = (0, 180)
) -> Tuple[Union[np.ndarray, tf.Tensor], Union[np.ndarray, tf.Tensor]]:
    """
    Simulate sparse-view CT acquisition.
    
    Args:
        phantom: Input phantom
        n_angles: Number of projection angles
        noise_level: Amount of noise to add
        angle_range: Range of angles (start, end) in degrees
    
    Returns:
        Tuple of (sinogram, angles)
    """
    
    # Generate angles
    angles = np.linspace(angle_range[0], angle_range[1], n_angles, endpoint=False)
    
    # Forward projection
    sinogram = radon_transform(phantom, angles)
    
    # Add noise if requested
    if noise_level > 0:
        if isinstance(sinogram, tf.Tensor):
            noise = tf.random.normal(tf.shape(sinogram), stddev=noise_level)
            sinogram = sinogram + noise
        else:
            noise = np.random.normal(0, noise_level, sinogram.shape)
            sinogram = sinogram + noise
    
    return sinogram, angles


def compare_reconstruction_filters(
    sinogram: Union[np.ndarray, tf.Tensor],
    angles: Union[np.ndarray, tf.Tensor, List[float]],
    filters: Optional[List[str]] = None
) -> Dict[str, Union[np.ndarray, tf.Tensor]]:
    """
    Compare different reconstruction filters.
    
    Args:
        sinogram: Input sinogram
        angles: Projection angles
        filters: List of filters to compare
    
    Returns:
        Dictionary mapping filter names to reconstructions
    """
    
    if filters is None:
        filters = ['ramp', 'shepp-logan', 'hamming', 'hann', 'cosine']
    
    reconstructions = {}
    
    for filter_name in filters:
        try:
            recon = fbp_reconstruction(sinogram, angles, filter_name)
            reconstructions[filter_name] = recon
        except Exception as e:
            print(f"Warning: Filter {filter_name} failed: {e}")
    
    return reconstructions


# Test functions
if __name__ == "__main__":
    print("🧪 Testing CT Transforms...")
    
    # Create test phantom
    size = 128
    phantom = np.zeros((size, size))
    
    # Add some shapes
    center = size // 2
    rr, cc = np.ogrid[:size, :size]
    
    # Large circle
    mask1 = (rr - center)**2 + (cc - center)**2 < (size//4)**2
    phantom[mask1] = 0.8
    
    # Small circle
    mask2 = (rr - center + 20)**2 + (cc - center - 20)**2 < (size//8)**2
    phantom[mask2] = 1.0
    
    print(f"✅ Test phantom created: {phantom.shape}, range: [{phantom.min():.3f}, {phantom.max():.3f}]")
    
    # Test Radon transform
    angles = np.linspace(0, 180, 45, endpoint=False)
    sinogram = radon_transform(phantom, angles)
    print(f"✅ Radon transform: {phantom.shape} -> {sinogram.shape}")
    
    # Test inverse Radon transform
    reconstruction = iradon_transform(sinogram, angles, 'shepp-logan')
    print(f"✅ Inverse Radon: {sinogram.shape} -> {reconstruction.shape}")
    
    # Test different filters
    filters = ['ramp', 'shepp-logan', 'hamming']
    filter_results = compare_reconstruction_filters(sinogram, angles, filters)
    print(f"✅ Filter comparison: {len(filter_results)} filters tested")
    
    # Test sparse-view simulation
    sparse_sino, sparse_angles = simulate_sparse_view_ct(phantom, n_angles=20, noise_level=0.05)
    print(f"✅ Sparse-view simulation: {len(sparse_angles)} angles, noise added")
    
    # Test with TensorFlow tensors
    tf_phantom = tf.constant(phantom, dtype=tf.float32)
    tf_angles = tf.constant(angles, dtype=tf.float32)
    
    tf_sinogram = radon_transform(tf_phantom, tf_angles)
    tf_reconstruction = iradon_transform(tf_sinogram, tf_angles)
    
    print(f"✅ TensorFlow compatibility: {tf_phantom.shape} -> {tf_sinogram.shape} -> {tf_reconstruction.shape}")
    
    # Compute reconstruction quality
    mse = np.mean((phantom - reconstruction)**2)
    psnr = 20 * np.log10(phantom.max()) - 10 * np.log10(mse) if mse > 0 else float('inf')
    
    print(f"✅ Reconstruction quality: MSE = {mse:.6f}, PSNR = {psnr:.2f} dB")
    
    print("\n🎉 CT Transforms test completed successfully!")
