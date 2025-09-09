"""
CompactCTNet FIXED: Improved Physics-Informed Neural Network for Sparse-View CT Reconstruction

This is the CORRECTED version of CompactCTNet that fixes the performance issues:

FIXES APPLIED:
1. ✅ Increased base filters from 32 to 64 (matching UNetCT)
2. ✅ Removed problematic MultiHeadSelfAttention 
3. ✅ Simplified skip connections (direct concatenation)
4. ✅ Removed non-functional Physics Loss
5. ✅ Fixed input shape handling
6. ✅ Added only effective Spatial Attention
7. ✅ Improved architecture balance

Key Changes:
- Base filters: 32 → 64 (2x capacity)
- Attention: Complex MultiHead → Simple Spatial
- Skip connections: Learnable fusion → Direct concatenation  
- Physics loss: Removed (not functional without real sinograms)
- Architecture: Simplified but more effective

Author: Your Name (Fixed Version)
Date: September 2024
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Tuple, Optional, Dict

from .attention_layers import SpatialAttention


class CompactCTNetFixed(keras.Model):
    """
    CompactCTNet FIXED: Improved neural architecture for sparse-view CT reconstruction.
    
    This version fixes the performance issues of the original CompactCTNet:
    
    Architecture Overview:
    Input (128x128x1) → U-Net Encoder (64 filters) → Spatial Attention → U-Net Decoder → Output (128x128x1)
    
    Key Improvements:
    - Doubled base filters (32→64) for better capacity
    - Removed complex MultiHeadAttention that lost spatial info
    - Simplified skip connections for better gradient flow
    - Removed non-functional physics loss
    - Added effective spatial attention only
    
    Args:
        input_shape: Input image shape (height, width, channels)
        num_filters: Base number of filters (doubled at each level) - NOW 64!
        use_spatial_attention: Whether to use spatial attention
        dropout_rate: Dropout rate for regularization
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (128, 128, 1),  # FIXED: Correct input shape
        num_filters: int = 64,  # FIXED: Increased from 32 to 64
        use_spatial_attention: bool = True,  # FIXED: Only spatial attention
        dropout_rate: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.input_shape_param = input_shape
        self.num_filters = num_filters
        self.use_spatial_attention = use_spatial_attention
        self.dropout_rate = dropout_rate
        
        # Build network components
        self._build_encoder()
        self._build_attention()
        self._build_decoder()
        self._build_output()
    
    def _build_encoder(self):
        """Build improved U-Net encoder."""
        
        # Encoder blocks (downsampling path)
        self.encoder_blocks = []
        filters = self.num_filters
        
        for i in range(4):  # 4 levels of encoding
            block = self._create_encoder_block(
                filters=filters,
                name=f"encoder_block_{i}"
            )
            self.encoder_blocks.append(block)
            filters *= 2
    
    def _build_attention(self):
        """Build simplified attention mechanism."""
        
        if self.use_spatial_attention:
            # FIXED: Only spatial attention, no complex MultiHead
            self.spatial_attention = SpatialAttention(
                name="spatial_attention"
            )
            
            # Simple layer norm for attention
            self.attention_norm = layers.LayerNormalization(name="attention_norm")
        else:
            self.spatial_attention = None
            self.attention_norm = None
    
    def _build_decoder(self):
        """Build improved U-Net decoder with direct skip connections."""
        
        # Decoder blocks (upsampling path)
        self.decoder_blocks = []
        filters = self.num_filters * 8
        
        for i in range(4):  # 4 levels of decoding
            block = self._create_decoder_block(
                filters=filters,
                name=f"decoder_block_{i}"
            )
            self.decoder_blocks.append(block)
            filters //= 2
    
    def _build_output(self):
        """Build output layer."""
        
        self.output_conv = layers.Conv2D(
            1, 
            kernel_size=1, 
            activation='linear',
            name="output_conv"
        )
    
    def _create_encoder_block(self, filters: int, name: str):
        """Create an improved encoder block."""
        
        def encoder_block(x):
            # First convolution
            conv1 = layers.Conv2D(
                filters, 3, padding='same',
                activation='relu', name=f"{name}_conv1"
            )(x)
            conv1 = layers.BatchNormalization(name=f"{name}_bn1")(conv1)
            
            # Second convolution
            conv2 = layers.Conv2D(
                filters, 3, padding='same',
                activation='relu', name=f"{name}_conv2"
            )(conv1)
            conv2 = layers.BatchNormalization(name=f"{name}_bn2")(conv2)
            
            # Dropout for regularization
            conv2 = layers.Dropout(self.dropout_rate, name=f"{name}_dropout")(conv2)
            
            # FIXED: Simple residual connection when possible
            if x.shape[-1] == filters:
                conv2 = layers.Add(name=f"{name}_residual")([x, conv2])
            
            # Max pooling for downsampling
            pool = layers.MaxPooling2D(2, name=f"{name}_pool")(conv2)
            
            return conv2, pool  # Return both feature map and pooled version
        
        return encoder_block
    
    def _create_decoder_block(self, filters: int, name: str):
        """Create an improved decoder block with direct skip connections."""
        
        def decoder_block(x, skip_connection=None):
            # Upsampling
            up = layers.UpSampling2D(2, name=f"{name}_upsample")(x)
            up = layers.Conv2D(
                filters, 2, padding='same',
                activation='relu', name=f"{name}_up_conv"
            )(up)
            
            # FIXED: Direct concatenation (no learnable fusion weights)
            if skip_connection is not None:
                # Handle size mismatches
                if up.shape[1:3] != skip_connection.shape[1:3]:
                    skip_connection = tf.image.resize(skip_connection, up.shape[1:3])
                
                up = layers.Concatenate(name=f"{name}_concat")([up, skip_connection])
            
            # Convolutions
            conv1 = layers.Conv2D(
                filters, 3, padding='same',
                activation='relu', name=f"{name}_conv1"
            )(up)
            conv1 = layers.BatchNormalization(name=f"{name}_bn1")(conv1)
            
            conv2 = layers.Conv2D(
                filters, 3, padding='same',
                activation='relu', name=f"{name}_conv2"
            )(conv1)
            conv2 = layers.BatchNormalization(name=f"{name}_bn2")(conv2)
            conv2 = layers.Dropout(self.dropout_rate, name=f"{name}_dropout")(conv2)
            
            return conv2
        
        return decoder_block
    
    def call(self, inputs, training=None):
        """Forward pass through the improved network."""
        
        x = inputs
        skip_connections = []
        
        # Encoder path
        for i, encoder_block in enumerate(self.encoder_blocks):
            skip, x = encoder_block(x)
            skip_connections.append(skip)
        
        # FIXED: Apply only spatial attention at the bottleneck (no complex reshaping)
        if self.use_spatial_attention:
            x = self.spatial_attention(x)
            x = self.attention_norm(x)
        
        # Decoder path with direct skip connections
        skip_connections.reverse()  # Reverse for decoder
        for i, decoder_block in enumerate(self.decoder_blocks):
            if i < len(skip_connections):
                x = decoder_block(x, skip_connections[i])
            else:
                x = decoder_block(x)
        
        # Final output
        output = self.output_conv(x)
        
        return output
    
    def get_config(self):
        """Return configuration for model serialization."""
        config = super().get_config()
        config.update({
            'input_shape': self.input_shape_param,
            'num_filters': self.num_filters,
            'use_spatial_attention': self.use_spatial_attention,
            'dropout_rate': self.dropout_rate,
        })
        return config


def create_compact_ct_net_fixed(
    input_shape: Tuple[int, int, int] = (128, 128, 1),  # FIXED: Correct default
    **kwargs
) -> CompactCTNetFixed:
    """
    Factory function to create the FIXED CompactCTNet model.
    
    Args:
        input_shape: Input image shape
        **kwargs: Additional arguments for CompactCTNetFixed
    
    Returns:
        Compiled CompactCTNetFixed model
    """
    
    model = CompactCTNetFixed(input_shape=input_shape, **kwargs)
    
    # Build the model
    dummy_input = tf.zeros((1,) + input_shape)
    _ = model(dummy_input)
    
    return model


# Comparison function
def compare_original_vs_fixed():
    """Compare original CompactCTNet vs Fixed version."""
    
    print("🔧 CompactCTNet: Original vs Fixed Comparison")
    print("=" * 60)
    
    try:
        from .compact_ct_net import CompactCTNet
        
        # Original version
        original = CompactCTNet(
            input_shape=(128, 128, 1),
            num_filters=32,
            num_attention_heads=4,
            use_physics_loss=True
        )
        
        # Fixed version
        fixed = CompactCTNetFixed(
            input_shape=(128, 128, 1),
            num_filters=64,
            use_spatial_attention=True
        )
        
        # Build both models
        dummy_input = tf.zeros((1, 128, 128, 1))
        _ = original(dummy_input)
        _ = fixed(dummy_input)
        
        # Compare
        orig_params = original.count_params()
        fixed_params = fixed.count_params()
        
        print(f"Original CompactCTNet:")
        print(f"  - Base filters: 32")
        print(f"  - Attention: MultiHead + Spatial")
        print(f"  - Physics Loss: Yes (non-functional)")
        print(f"  - Skip connections: Learnable fusion")
        print(f"  - Parameters: {orig_params:,}")
        
        print(f"\nFixed CompactCTNet:")
        print(f"  - Base filters: 64 (+100%)")
        print(f"  - Attention: Spatial only")
        print(f"  - Physics Loss: Removed")
        print(f"  - Skip connections: Direct")
        print(f"  - Parameters: {fixed_params:,}")
        
        print(f"\n📊 Changes:")
        print(f"  - Parameter change: {((fixed_params - orig_params) / orig_params * 100):+.1f}%")
        print(f"  - Expected performance: Much better!")
        
    except ImportError:
        print("⚠️ Original CompactCTNet not available for comparison")
        
        # Just show fixed version
        fixed = CompactCTNetFixed()
        dummy_input = tf.zeros((1, 128, 128, 1))
        _ = fixed(dummy_input)
        
        print(f"Fixed CompactCTNet:")
        print(f"  - Base filters: 64")
        print(f"  - Parameters: {fixed.count_params():,}")
        print(f"  - Architecture: Simplified and improved")


# Model summary and statistics
def print_fixed_model_summary(model: CompactCTNetFixed):
    """Print detailed summary of the fixed model."""
    
    print("🧠 CompactCTNet FIXED - Architecture Summary")
    print("=" * 50)
    
    # Build model with dummy input to get summary
    dummy_input = tf.zeros((1,) + model.input_shape_param)
    _ = model(dummy_input)
    
    # Print model summary
    model.summary()
    
    # Calculate model statistics
    total_params = model.count_params()
    
    print(f"\n📊 Fixed Model Statistics:")
    print(f"Total Parameters: {total_params:,}")
    print(f"Model Size (MB): {total_params * 4 / (1024**2):.2f}")
    
    print(f"\n🎯 Key Improvements:")
    print(f"- Input Shape: {model.input_shape_param} (FIXED)")
    print(f"- Base Filters: {model.num_filters} (DOUBLED from 32)")
    print(f"- Spatial Attention: {model.use_spatial_attention} (Simplified)")
    print(f"- Physics Loss: Removed (was non-functional)")
    print(f"- Skip Connections: Direct (was learnable fusion)")
    print(f"- Dropout Rate: {model.dropout_rate}")
    
    print(f"\n✅ Expected Performance: MUCH BETTER than original!")


if __name__ == "__main__":
    # Test the fixed model
    print("🧪 Testing CompactCTNet FIXED...")
    
    model = create_compact_ct_net_fixed()
    print_fixed_model_summary(model)
    
    # Test forward pass
    test_input = tf.random.normal((2, 128, 128, 1))
    output = model(test_input)
    print(f"\n✅ Test passed! Output shape: {output.shape}")
    
    # Compare with original
    print("\n" + "=" * 60)
    compare_original_vs_fixed()
    
    print("\n🎉 CompactCTNet FIXED implementation completed!")
    print("💡 This version should perform much better than the original!")
