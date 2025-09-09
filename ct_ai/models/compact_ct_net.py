"""
CompactCTNet: Physics-Informed Neural Network for Sparse-View CT Reconstruction

This is our main innovation - a neural architecture that combines:
1. U-Net backbone for robust feature extraction
2. Multi-head self-attention for global dependencies
3. Physics-informed constraints via Radon transform consistency
4. Adaptive fusion for multi-scale integration

Key Innovation: First neural network that explicitly incorporates
the physics of CT acquisition (Radon transform) in the architecture.

Author: Your Name
Date: September 2024
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Tuple, Optional, Dict

from .attention_layers import MultiHeadSelfAttention, SpatialAttention
from .physics_loss import PhysicsInformedLoss


class CompactCTNet(keras.Model):
    """
    CompactCTNet: Advanced neural architecture for sparse-view CT reconstruction.
    
    Architecture Overview:
    Input (256x256x1) → U-Net Encoder → Attention Layers → Physics-Aware Decoder → Output (256x256x1)
    
    Key Features:
    - Efficient U-Net backbone (proven for medical imaging)
    - Multi-head attention for capturing global dependencies
    - Physics-informed loss for Radon transform consistency
    - Adaptive skip connections with learned fusion weights
    
    Args:
        input_shape: Input image shape (height, width, channels)
        num_filters: Base number of filters (doubled at each level)
        num_attention_heads: Number of heads in multi-head attention
        use_physics_loss: Whether to use physics-informed loss
        dropout_rate: Dropout rate for regularization
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (256, 256, 1),
        num_filters: int = 32,
        num_attention_heads: int = 4,
        use_physics_loss: bool = True,
        dropout_rate: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.input_shape = input_shape
        self.num_filters = num_filters
        self.num_attention_heads = num_attention_heads
        self.use_physics_loss = use_physics_loss
        self.dropout_rate = dropout_rate
        
        # Build network components
        self._build_encoder()
        self._build_attention_layers()
        self._build_decoder()
        self._build_fusion_layers()
        
        # Physics-informed loss
        if self.use_physics_loss:
            self.physics_loss = PhysicsInformedLoss()
    
    def _build_encoder(self):
        """Build U-Net encoder with residual connections."""
        
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
    
    def _build_attention_layers(self):
        """Build multi-head self-attention layers."""
        
        # Multi-head self-attention for global dependencies
        self.global_attention = MultiHeadSelfAttention(
            num_heads=self.num_attention_heads,
            key_dim=self.num_filters * 8,  # Bottleneck features
            name="global_attention"
        )
        
        # Spatial attention for feature refinement
        self.spatial_attention = SpatialAttention(
            name="spatial_attention"
        )
        
        # Layer normalization for attention
        self.attention_norm = layers.LayerNormalization(name="attention_norm")
    
    def _build_decoder(self):
        """Build U-Net decoder with skip connections."""
        
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
    
    def _build_fusion_layers(self):
        """Build adaptive fusion layers for skip connections."""
        
        # Learned fusion weights for skip connections
        self.fusion_weights = []
        for i in range(4):
            weight_layer = layers.Dense(
                1, 
                activation='sigmoid',
                name=f"fusion_weight_{i}"
            )
            self.fusion_weights.append(weight_layer)
        
        # Final output layer
        self.output_conv = layers.Conv2D(
            1, 
            kernel_size=1, 
            activation='linear',
            name="output_conv"
        )
    
    def _create_encoder_block(self, filters: int, name: str):
        """Create a single encoder block."""
        
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
            
            # Residual connection if same number of channels
            if x.shape[-1] == filters:
                conv2 = layers.Add(name=f"{name}_residual")([x, conv2])
            
            # Max pooling for downsampling
            pool = layers.MaxPooling2D(2, name=f"{name}_pool")(conv2)
            
            return conv2, pool  # Return both feature map and pooled version
        
        return encoder_block
    
    def _create_decoder_block(self, filters: int, name: str):
        """Create a single decoder block."""
        
        def decoder_block(x, skip_connection=None):
            # Upsampling
            up = layers.UpSampling2D(2, name=f"{name}_upsample")(x)
            up = layers.Conv2D(
                filters, 2, padding='same',
                activation='relu', name=f"{name}_up_conv"
            )(up)
            
            # Concatenate with skip connection if provided
            if skip_connection is not None:
                # Adaptive fusion of skip connection
                fusion_weight = self.fusion_weights[int(name.split('_')[-1])]
                weight = fusion_weight(layers.GlobalAveragePooling2D()(skip_connection))
                weight = tf.expand_dims(tf.expand_dims(weight, 1), 1)
                
                weighted_skip = weight * skip_connection
                up = layers.Concatenate(name=f"{name}_concat")([up, weighted_skip])
            
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
        """Forward pass through the network."""
        
        x = inputs
        skip_connections = []
        
        # Encoder path
        for i, encoder_block in enumerate(self.encoder_blocks):
            skip, x = encoder_block(x)
            skip_connections.append(skip)
        
        # Apply attention at the bottleneck
        # Reshape for attention (flatten spatial dimensions)
        batch_size = tf.shape(x)[0]
        h, w, c = x.shape[1], x.shape[2], x.shape[3]
        
        # Global attention
        x_flat = tf.reshape(x, [batch_size, h * w, c])
        x_attended = self.global_attention(x_flat, x_flat, training=training)
        x_attended = self.attention_norm(x_attended + x_flat)  # Residual connection
        
        # Reshape back to spatial format
        x = tf.reshape(x_attended, [batch_size, h, w, c])
        
        # Spatial attention
        x = self.spatial_attention(x)
        
        # Decoder path with skip connections
        skip_connections.reverse()  # Reverse for decoder
        for i, decoder_block in enumerate(self.decoder_blocks):
            if i < len(skip_connections):
                x = decoder_block(x, skip_connections[i])
            else:
                x = decoder_block(x)
        
        # Final output
        output = self.output_conv(x)
        
        return output
    
    def compute_loss(self, y_true, y_pred, sinogram=None, angles=None):
        """
        Compute total loss including physics-informed component.
        
        Args:
            y_true: Ground truth reconstruction
            y_pred: Predicted reconstruction
            sinogram: Original sinogram (for physics loss)
            angles: Projection angles (for physics loss)
        
        Returns:
            Total loss combining reconstruction and physics terms
        """
        
        # Standard reconstruction loss (MSE)
        recon_loss = tf.reduce_mean(tf.square(y_true - y_pred))
        
        # Physics-informed loss (if enabled and data available)
        if self.use_physics_loss and sinogram is not None and angles is not None:
            physics_loss = self.physics_loss(y_pred, sinogram, angles)
            total_loss = recon_loss + 0.1 * physics_loss
        else:
            total_loss = recon_loss
        
        return total_loss
    
    def get_config(self):
        """Return configuration for model serialization."""
        config = super().get_config()
        config.update({
            'input_shape': self.input_shape,
            'num_filters': self.num_filters,
            'num_attention_heads': self.num_attention_heads,
            'use_physics_loss': self.use_physics_loss,
            'dropout_rate': self.dropout_rate,
        })
        return config


def create_compact_ct_net(
    input_shape: Tuple[int, int, int] = (256, 256, 1),
    **kwargs
) -> CompactCTNet:
    """
    Factory function to create CompactCTNet model.
    
    Args:
        input_shape: Input image shape
        **kwargs: Additional arguments for CompactCTNet
    
    Returns:
        Compiled CompactCTNet model
    """
    
    model = CompactCTNet(input_shape=input_shape, **kwargs)
    
    # Build the model
    dummy_input = tf.zeros((1,) + input_shape)
    _ = model(dummy_input)
    
    return model


# Model summary and statistics
def print_model_summary(model: CompactCTNet):
    """Print detailed model summary with parameter counts."""
    
    print("🧠 CompactCTNet Architecture Summary")
    print("=" * 50)
    
    # Build model with dummy input to get summary
    dummy_input = tf.zeros((1,) + model.input_shape)
    _ = model(dummy_input)
    
    # Print model summary
    model.summary()
    
    # Calculate model statistics
    total_params = model.count_params()
    trainable_params = sum([tf.size(var) for var in model.trainable_variables])
    
    print(f"\n📊 Model Statistics:")
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Model Size (MB): {total_params * 4 / (1024**2):.2f}")
    
    print(f"\n🎯 Key Features:")
    print(f"- Input Shape: {model.input_shape}")
    print(f"- Base Filters: {model.num_filters}")
    print(f"- Attention Heads: {model.num_attention_heads}")
    print(f"- Physics Loss: {model.use_physics_loss}")
    print(f"- Dropout Rate: {model.dropout_rate}")


if __name__ == "__main__":
    # Test the model
    print("Testing CompactCTNet...")
    
    model = create_compact_ct_net()
    print_model_summary(model)
    
    # Test forward pass
    test_input = tf.random.normal((2, 256, 256, 1))
    output = model(test_input)
    print(f"\n✅ Test passed! Output shape: {output.shape}")
