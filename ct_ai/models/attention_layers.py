"""
Attention Mechanisms for CT Reconstruction

Custom attention layers designed specifically for CT imaging:
1. MultiHeadSelfAttention: Captures global dependencies in reconstructions
2. SpatialAttention: Focuses on important anatomical regions
3. ChannelAttention: Emphasizes relevant feature channels

These attention mechanisms help the network focus on important image regions
and learn better feature representations for sparse-view CT reconstruction.

Author: Your Name
Date: September 2024
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


class MultiHeadSelfAttention(layers.Layer):
    """
    Multi-Head Self-Attention layer optimized for CT reconstruction.
    
    This layer captures long-range dependencies in the image, which is crucial
    for sparse-view CT where information from different projection angles
    needs to be integrated effectively.
    
    Args:
        num_heads: Number of attention heads
        key_dim: Dimension of attention keys/queries
        dropout_rate: Dropout rate for attention weights
    """
    
    def __init__(
        self, 
        num_heads: int = 4,
        key_dim: int = 64,
        dropout_rate: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.dropout_rate = dropout_rate
        
        # Multi-head attention layer
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            dropout=dropout_rate,
            name="multi_head_attention"
        )
        
        # Layer normalization
        self.layer_norm = layers.LayerNormalization(name="attention_layer_norm")
        
        # Feed-forward network
        self.ffn = keras.Sequential([
            layers.Dense(key_dim * 4, activation='relu', name="ffn_dense1"),
            layers.Dropout(dropout_rate, name="ffn_dropout"),
            layers.Dense(key_dim, name="ffn_dense2")
        ], name="feed_forward_network")
        
        self.ffn_norm = layers.LayerNormalization(name="ffn_layer_norm")
    
    def call(self, query, value, training=None):
        """
        Apply multi-head self-attention.
        
        Args:
            query: Query tensor (typically the input features)
            value: Value tensor (typically the same as query for self-attention)
            training: Training mode flag
        
        Returns:
            Attention-enhanced features
        """
        
        # Self-attention with residual connection
        attention_output = self.attention(
            query=query, 
            value=value, 
            training=training
        )
        attention_output = self.layer_norm(query + attention_output)
        
        # Feed-forward network with residual connection
        ffn_output = self.ffn(attention_output, training=training)
        output = self.ffn_norm(attention_output + ffn_output)
        
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'num_heads': self.num_heads,
            'key_dim': self.key_dim,
            'dropout_rate': self.dropout_rate,
        })
        return config


class SpatialAttention(layers.Layer):
    """
    Spatial Attention mechanism for CT reconstruction.
    
    This layer learns to focus on important spatial regions in the image,
    such as anatomical structures or areas with high reconstruction uncertainty.
    
    Args:
        reduction_ratio: Ratio for channel reduction in attention computation
    """
    
    def __init__(self, reduction_ratio: int = 8, **kwargs):
        super().__init__(**kwargs)
        
        self.reduction_ratio = reduction_ratio
        
        # Spatial attention computation
        self.conv1 = layers.Conv2D(
            1, 
            kernel_size=7, 
            padding='same',
            activation='sigmoid',
            name="spatial_attention_conv"
        )
        
        # Global pooling for spatial statistics
        self.global_avg_pool = layers.GlobalAveragePooling2D(keepdims=True)
        self.global_max_pool = layers.GlobalMaxPooling2D(keepdims=True)
    
    def call(self, inputs, training=None):
        """
        Apply spatial attention to input features.
        
        Args:
            inputs: Input feature maps (batch, height, width, channels)
            training: Training mode flag
        
        Returns:
            Spatially-attended features
        """
        
        # Compute spatial statistics
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)
        
        # Concatenate statistics
        spatial_stats = tf.concat([avg_pool, max_pool], axis=-1)
        
        # Compute spatial attention weights
        attention_weights = self.conv1(spatial_stats)
        
        # Apply attention weights
        attended_features = inputs * attention_weights
        
        return attended_features
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'reduction_ratio': self.reduction_ratio,
        })
        return config


class ChannelAttention(layers.Layer):
    """
    Channel Attention mechanism (Squeeze-and-Excitation style).
    
    This layer learns to emphasize important feature channels while
    suppressing less relevant ones, improving feature discrimination.
    
    Args:
        reduction_ratio: Ratio for channel reduction in attention computation
    """
    
    def __init__(self, reduction_ratio: int = 16, **kwargs):
        super().__init__(**kwargs)
        
        self.reduction_ratio = reduction_ratio
        
        # Global average pooling
        self.global_avg_pool = layers.GlobalAveragePooling2D()
        
        # Channel attention network will be built in build()
        self.dense1 = None
        self.dense2 = None
    
    def build(self, input_shape):
        """Build the layer based on input shape."""
        super().build(input_shape)
        
        channels = input_shape[-1]
        reduced_channels = max(channels // self.reduction_ratio, 1)
        
        # Channel attention network
        self.dense1 = layers.Dense(
            reduced_channels,
            activation='relu',
            name="channel_attention_dense1"
        )
        
        self.dense2 = layers.Dense(
            channels,
            activation='sigmoid',
            name="channel_attention_dense2"
        )
    
    def call(self, inputs, training=None):
        """
        Apply channel attention to input features.
        
        Args:
            inputs: Input feature maps (batch, height, width, channels)
            training: Training mode flag
        
        Returns:
            Channel-attended features
        """
        
        # Global context
        global_context = self.global_avg_pool(inputs)
        
        # Channel attention weights
        attention_weights = self.dense1(global_context)
        attention_weights = self.dense2(attention_weights)
        
        # Reshape for broadcasting
        attention_weights = tf.expand_dims(tf.expand_dims(attention_weights, 1), 1)
        
        # Apply channel attention
        attended_features = inputs * attention_weights
        
        return attended_features
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'reduction_ratio': self.reduction_ratio,
        })
        return config


class CombinedAttention(layers.Layer):
    """
    Combined Spatial and Channel Attention mechanism.
    
    This layer applies both spatial and channel attention sequentially,
    providing comprehensive attention across both spatial and feature dimensions.
    
    Args:
        channel_reduction_ratio: Reduction ratio for channel attention
        spatial_reduction_ratio: Reduction ratio for spatial attention
    """
    
    def __init__(
        self, 
        channel_reduction_ratio: int = 16,
        spatial_reduction_ratio: int = 8,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.channel_reduction_ratio = channel_reduction_ratio
        self.spatial_reduction_ratio = spatial_reduction_ratio
        
        # Initialize attention modules
        self.channel_attention = ChannelAttention(
            reduction_ratio=channel_reduction_ratio,
            name="channel_attention"
        )
        
        self.spatial_attention = SpatialAttention(
            reduction_ratio=spatial_reduction_ratio,
            name="spatial_attention"
        )
    
    def call(self, inputs, training=None):
        """
        Apply combined spatial and channel attention.
        
        Args:
            inputs: Input feature maps
            training: Training mode flag
        
        Returns:
            Features with combined attention applied
        """
        
        # Apply channel attention first
        channel_attended = self.channel_attention(inputs, training=training)
        
        # Then apply spatial attention
        output = self.spatial_attention(channel_attended, training=training)
        
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'channel_reduction_ratio': self.channel_reduction_ratio,
            'spatial_reduction_ratio': self.spatial_reduction_ratio,
        })
        return config


def create_attention_block(
    attention_type: str = "combined",
    **kwargs
) -> layers.Layer:
    """
    Factory function to create different types of attention layers.
    
    Args:
        attention_type: Type of attention ('spatial', 'channel', 'combined', 'multi_head')
        **kwargs: Additional arguments for specific attention types
    
    Returns:
        Attention layer instance
    """
    
    if attention_type == "spatial":
        return SpatialAttention(**kwargs)
    elif attention_type == "channel":
        return ChannelAttention(**kwargs)
    elif attention_type == "combined":
        return CombinedAttention(**kwargs)
    elif attention_type == "multi_head":
        return MultiHeadSelfAttention(**kwargs)
    else:
        raise ValueError(f"Unknown attention type: {attention_type}")


# Test functions
if __name__ == "__main__":
    print("Testing Attention Layers...")
    
    # Test input
    batch_size, height, width, channels = 2, 64, 64, 128
    test_input = tf.random.normal((batch_size, height, width, channels))
    
    print(f"Input shape: {test_input.shape}")
    
    # Test Spatial Attention
    spatial_att = SpatialAttention()
    spatial_output = spatial_att(test_input)
    print(f"Spatial Attention output shape: {spatial_output.shape}")
    
    # Test Channel Attention
    channel_att = ChannelAttention()
    channel_output = channel_att(test_input)
    print(f"Channel Attention output shape: {channel_output.shape}")
    
    # Test Multi-Head Self-Attention
    # Reshape for sequence input
    seq_input = tf.reshape(test_input, (batch_size, height * width, channels))
    multi_head_att = MultiHeadSelfAttention(num_heads=4, key_dim=channels)
    multi_head_output = multi_head_att(seq_input, seq_input)
    print(f"Multi-Head Attention output shape: {multi_head_output.shape}")
    
    # Test Combined Attention
    combined_att = CombinedAttention()
    combined_output = combined_att(test_input)
    print(f"Combined Attention output shape: {combined_output.shape}")
    
    print("✅ All attention layers working correctly!")
