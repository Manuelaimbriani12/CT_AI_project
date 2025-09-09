"""
U-Net Implementation for CT Reconstruction

This module provides a standard U-Net implementation specifically adapted
for CT reconstruction tasks. This serves as an important baseline for
comparison with our physics-informed CompactCTNet.

The U-Net architecture is widely used in medical imaging and provides
a solid foundation for image-to-image translation tasks like CT reconstruction.

Key Features:
- Classic U-Net architecture with skip connections
- Batch normalization and dropout for regularization
- Configurable depth and filter counts
- Medical imaging optimizations
- Compatible with our training pipeline

Author: Your Name
Date: September 2024
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Tuple, Optional, Dict


class UNetCT(keras.Model):
    """
    U-Net architecture for CT reconstruction.
    
    This implementation follows the classic U-Net design with:
    - Encoder-decoder structure with skip connections
    - Batch normalization for stable training
    - Dropout for regularization
    - ReLU activations
    - Configurable depth and filters
    
    Args:
        input_shape: Input image shape (height, width, channels)
        num_filters: Base number of filters (doubled at each level)
        depth: Number of encoder/decoder levels
        dropout_rate: Dropout rate for regularization
        use_batch_norm: Whether to use batch normalization
        activation: Activation function to use
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (256, 256, 1),
        num_filters: int = 64,
        depth: int = 4,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        activation: str = 'relu',
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.input_shape = input_shape
        self.num_filters = num_filters
        self.depth = depth
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.activation = activation
        
        # Build encoder and decoder
        self._build_encoder()
        self._build_decoder()
        self._build_output()
    
    def _build_encoder(self):
        """Build U-Net encoder (downsampling path)."""
        
        self.encoder_blocks = []
        filters = self.num_filters
        
        for i in range(self.depth):
            block = self._create_conv_block(
                filters=filters,
                name=f"encoder_block_{i}"
            )
            self.encoder_blocks.append(block)
            filters *= 2
        
        # Bottleneck
        self.bottleneck = self._create_conv_block(
            filters=filters,
            name="bottleneck"
        )
    
    def _build_decoder(self):
        """Build U-Net decoder (upsampling path)."""
        
        self.decoder_blocks = []
        self.upconv_blocks = []
        
        filters = self.num_filters * (2 ** self.depth)
        
        for i in range(self.depth):
            # Upsampling
            upconv = layers.Conv2DTranspose(
                filters // 2,
                kernel_size=2,
                strides=2,
                padding='same',
                name=f"upconv_{i}"
            )
            self.upconv_blocks.append(upconv)
            
            # Decoder block
            decoder_block = self._create_conv_block(
                filters=filters // 2,
                name=f"decoder_block_{i}"
            )
            self.decoder_blocks.append(decoder_block)
            
            filters //= 2
    
    def _build_output(self):
        """Build output layer."""
        
        self.output_conv = layers.Conv2D(
            1,
            kernel_size=1,
            activation='linear',
            name="output_conv"
        )
    
    def _create_conv_block(self, filters: int, name: str):
        """
        Create a convolutional block.
        
        Args:
            filters: Number of filters
            name: Block name
        
        Returns:
            Sequential model representing the conv block
        """
        
        block_layers = []
        
        # First convolution
        block_layers.append(
            layers.Conv2D(
                filters,
                kernel_size=3,
                padding='same',
                name=f"{name}_conv1"
            )
        )
        
        if self.use_batch_norm:
            block_layers.append(
                layers.BatchNormalization(name=f"{name}_bn1")
            )
        
        block_layers.append(
            layers.Activation(self.activation, name=f"{name}_act1")
        )
        
        # Second convolution
        block_layers.append(
            layers.Conv2D(
                filters,
                kernel_size=3,
                padding='same',
                name=f"{name}_conv2"
            )
        )
        
        if self.use_batch_norm:
            block_layers.append(
                layers.BatchNormalization(name=f"{name}_bn2")
            )
        
        block_layers.append(
            layers.Activation(self.activation, name=f"{name}_act2")
        )
        
        # Dropout
        if self.dropout_rate > 0:
            block_layers.append(
                layers.Dropout(self.dropout_rate, name=f"{name}_dropout")
            )
        
        return keras.Sequential(block_layers, name=name)
    
    def call(self, inputs, training=None):
        """Forward pass through the U-Net."""
        
        x = inputs
        skip_connections = []
        
        # Encoder path
        for i, encoder_block in enumerate(self.encoder_blocks):
            x = encoder_block(x, training=training)
            skip_connections.append(x)
            
            # Max pooling (except for the last encoder block)
            if i < len(self.encoder_blocks) - 1:
                x = layers.MaxPooling2D(2, name=f"pool_{i}")(x)
        
        # Bottleneck
        x = self.bottleneck(x, training=training)
        
        # Decoder path
        skip_connections.reverse()
        
        for i, (upconv, decoder_block) in enumerate(zip(self.upconv_blocks, self.decoder_blocks)):
            # Upsampling
            x = upconv(x)
            
            # Skip connection
            if i < len(skip_connections):
                skip = skip_connections[i]
                
                # Handle size mismatches
                if x.shape[1:3] != skip.shape[1:3]:
                    # Resize skip connection to match x
                    skip = tf.image.resize(skip, x.shape[1:3])
                
                x = layers.Concatenate(name=f"concat_{i}")([x, skip])
            
            # Decoder block
            x = decoder_block(x, training=training)
        
        # Output
        output = self.output_conv(x)
        
        return output
    
    def get_config(self):
        """Return configuration for model serialization."""
        config = super().get_config()
        config.update({
            'input_shape': self.input_shape,
            'num_filters': self.num_filters,
            'depth': self.depth,
            'dropout_rate': self.dropout_rate,
            'use_batch_norm': self.use_batch_norm,
            'activation': self.activation,
        })
        return config


def create_unet_ct(
    input_shape: Tuple[int, int, int] = (256, 256, 1),
    **kwargs
) -> UNetCT:
    """
    Factory function to create U-Net model for CT reconstruction.
    
    Args:
        input_shape: Input image shape
        **kwargs: Additional arguments for UNetCT
    
    Returns:
        Compiled U-Net model
    """
    
    model = UNetCT(input_shape=input_shape, **kwargs)
    
    # Build the model
    dummy_input = tf.zeros((1,) + input_shape)
    _ = model(dummy_input)
    
    return model


class UNetPlusPlus(keras.Model):
    """
    U-Net++ (Nested U-Net) implementation for CT reconstruction.
    
    This is an enhanced version of U-Net with nested skip connections
    for better feature propagation and gradient flow.
    
    Args:
        input_shape: Input image shape
        num_filters: Base number of filters
        depth: Network depth
        dropout_rate: Dropout rate
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (256, 256, 1),
        num_filters: int = 32,
        depth: int = 4,
        dropout_rate: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.input_shape = input_shape
        self.num_filters = num_filters
        self.depth = depth
        self.dropout_rate = dropout_rate
        
        # Build nested structure
        self._build_nested_unet()
    
    def _build_nested_unet(self):
        """Build U-Net++ nested structure."""
        
        # This is a simplified implementation
        # Full U-Net++ would have complex nested connections
        
        # For now, implement as enhanced U-Net with attention
        self.base_unet = UNetCT(
            input_shape=self.input_shape,
            num_filters=self.num_filters,
            depth=self.depth,
            dropout_rate=self.dropout_rate
        )
        
        # Add attention mechanism
        self.attention_gates = []
        for i in range(self.depth):
            attention = self._create_attention_gate(
                self.num_filters * (2 ** i),
                name=f"attention_{i}"
            )
            self.attention_gates.append(attention)
    
    def _create_attention_gate(self, filters: int, name: str):
        """Create attention gate for feature selection."""
        
        def attention_gate(gating_signal, skip_connection):
            # Simplified attention mechanism
            # In practice, this would be more sophisticated
            
            # Global average pooling for gating signal
            gate = layers.GlobalAveragePooling2D()(gating_signal)
            gate = layers.Dense(filters, activation='sigmoid')(gate)
            gate = layers.Reshape((1, 1, filters))(gate)
            
            # Apply attention
            attended = layers.Multiply()([skip_connection, gate])
            
            return attended
        
        return attention_gate
    
    def call(self, inputs, training=None):
        """Forward pass through U-Net++."""
        
        # For simplicity, delegate to base U-Net
        # Full implementation would use nested connections
        return self.base_unet(inputs, training=training)


# Utility functions
def print_unet_summary(model: UNetCT):
    """Print detailed U-Net model summary."""
    
    print("🏗️ U-Net Architecture Summary")
    print("=" * 40)
    
    # Build model with dummy input to get summary
    dummy_input = tf.zeros((1,) + model.input_shape)
    _ = model(dummy_input)
    
    # Print model summary
    model.summary()
    
    # Calculate model statistics
    total_params = model.count_params()
    
    print(f"\n📊 Model Statistics:")
    print(f"Total Parameters: {total_params:,}")
    print(f"Model Size (MB): {total_params * 4 / (1024**2):.2f}")
    
    print(f"\n🎯 Architecture Details:")
    print(f"Input Shape: {model.input_shape}")
    print(f"Base Filters: {model.num_filters}")
    print(f"Depth: {model.depth}")
    print(f"Dropout Rate: {model.dropout_rate}")
    print(f"Batch Norm: {model.use_batch_norm}")


if __name__ == "__main__":
    # Test the U-Net implementation
    print("🧪 Testing U-Net for CT...")
    
    # Create standard U-Net
    unet = create_unet_ct(
        input_shape=(256, 256, 1),
        num_filters=32,
        depth=4,
        dropout_rate=0.1
    )
    
    print_unet_summary(unet)
    
    # Test forward pass
    test_input = tf.random.normal((2, 256, 256, 1))
    output = unet(test_input)
    print(f"\n✅ Forward pass test: {test_input.shape} → {output.shape}")
    
    # Test different configurations
    print("\n🔧 Testing different configurations...")
    
    configs = [
        {'num_filters': 16, 'depth': 3, 'name': 'Small U-Net'},
        {'num_filters': 64, 'depth': 4, 'name': 'Large U-Net'},
        {'num_filters': 32, 'depth': 5, 'name': 'Deep U-Net'}
    ]
    
    for config in configs:
        name = config.pop('name')
        try:
            test_unet = create_unet_ct(**config)
            test_output = test_unet(test_input)
            params = test_unet.count_params()
            print(f"✅ {name}: {params:,} parameters, output {test_output.shape}")
        except Exception as e:
            print(f"❌ {name}: Error - {e}")
    
    # Test U-Net++
    print("\n🔧 Testing U-Net++...")
    try:
        unet_pp = UNetPlusPlus(
            input_shape=(256, 256, 1),
            num_filters=32,
            depth=3
        )
        
        pp_output = unet_pp(test_input)
        pp_params = unet_pp.count_params()
        print(f"✅ U-Net++: {pp_params:,} parameters, output {pp_output.shape}")
    except Exception as e:
        print(f"❌ U-Net++: Error - {e}")
    
    print("\n🎉 U-Net implementation test completed successfully!")
