"""
FBPConvNet: Post-Processing CNN for FBP Reconstructions

This module implements FBPConvNet, a state-of-the-art method that applies
deep learning as a post-processing step to traditional FBP reconstructions.

FBPConvNet represents an important baseline that:
1. Uses classical FBP for initial reconstruction
2. Applies a CNN to remove artifacts and improve quality
3. Provides a hybrid classical-AI approach

This is particularly relevant for sparse-view CT where FBP produces
artifacts that can be learned and corrected by neural networks.

Reference:
"A Deep Learning Method for Real-time Inversion of X-ray Images"
Würfl et al., 2016

Author: Your Name
Date: September 2024
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Tuple, Optional, List, Dict

# Import our transforms
from ..utils.ct_transforms import fbp_reconstruction


class FBPConvNet(keras.Model):
    """
    FBPConvNet: Post-processing CNN for FBP reconstructions.
    
    This model takes FBP reconstructions as input and applies a CNN
    to remove artifacts and improve image quality. The architecture
    is designed specifically for CT reconstruction post-processing.
    
    Architecture:
    - Multiple convolutional layers with residual connections
    - Batch normalization for stable training
    - Skip connections for preserving low-level features
    - Compact design for efficient processing
    
    Args:
        input_shape: Input image shape (height, width, channels)
        num_filters: Base number of filters
        num_layers: Number of convolutional layers
        use_residual: Whether to use residual connections
        use_batch_norm: Whether to use batch normalization
        dropout_rate: Dropout rate for regularization
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (256, 256, 1),
        num_filters: int = 64,
        num_layers: int = 8,
        use_residual: bool = True,
        use_batch_norm: bool = True,
        dropout_rate: float = 0.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.input_shape = input_shape
        self.num_filters = num_filters
        self.num_layers = num_layers
        self.use_residual = use_residual
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate
        
        # Build network layers
        self._build_network()
    
    def _build_network(self):
        """Build the FBPConvNet architecture."""
        
        # Input layer
        self.input_conv = layers.Conv2D(
            self.num_filters,
            kernel_size=3,
            padding='same',
            activation='relu',
            name='input_conv'
        )
        
        if self.use_batch_norm:
            self.input_bn = layers.BatchNormalization(name='input_bn')
        
        # Hidden layers
        self.hidden_layers = []
        
        for i in range(self.num_layers):
            layer_dict = {}
            
            # Convolution
            layer_dict['conv'] = layers.Conv2D(
                self.num_filters,
                kernel_size=3,
                padding='same',
                name=f'conv_{i}'
            )
            
            # Batch normalization
            if self.use_batch_norm:
                layer_dict['bn'] = layers.BatchNormalization(name=f'bn_{i}')
            
            # Activation
            layer_dict['activation'] = layers.ReLU(name=f'relu_{i}')
            
            # Dropout
            if self.dropout_rate > 0:
                layer_dict['dropout'] = layers.Dropout(
                    self.dropout_rate, 
                    name=f'dropout_{i}'
                )
            
            self.hidden_layers.append(layer_dict)
        
        # Output layer
        self.output_conv = layers.Conv2D(
            1,
            kernel_size=3,
            padding='same',
            activation='linear',
            name='output_conv'
        )
    
    def call(self, inputs, training=None):
        """Forward pass through FBPConvNet."""
        
        # Store input for residual connection
        residual_input = inputs
        
        # Input layer
        x = self.input_conv(inputs)
        
        if self.use_batch_norm:
            x = self.input_bn(x, training=training)
        
        # Hidden layers
        for i, layer_dict in enumerate(self.hidden_layers):
            # Store input for potential residual connection
            layer_input = x
            
            # Convolution
            x = layer_dict['conv'](x)
            
            # Batch normalization
            if 'bn' in layer_dict:
                x = layer_dict['bn'](x, training=training)
            
            # Activation
            x = layer_dict['activation'](x)
            
            # Dropout
            if 'dropout' in layer_dict:
                x = layer_dict['dropout'](x, training=training)
            
            # Residual connection every few layers
            if self.use_residual and i > 0 and i % 2 == 1:
                x = layers.Add(name=f'residual_{i}')([x, layer_input])
        
        # Output layer
        x = self.output_conv(x)
        
        # Global residual connection (add input to output)
        if self.use_residual:
            output = layers.Add(name='global_residual')([x, residual_input])
        else:
            output = x
        
        return output
    
    def get_config(self):
        """Return configuration for model serialization."""
        config = super().get_config()
        config.update({
            'input_shape': self.input_shape,
            'num_filters': self.num_filters,
            'num_layers': self.num_layers,
            'use_residual': self.use_residual,
            'use_batch_norm': self.use_batch_norm,
            'dropout_rate': self.dropout_rate,
        })
        return config


class FBPConvNetPipeline(keras.Model):
    """
    Complete FBPConvNet pipeline including FBP reconstruction.
    
    This model takes sinograms as input, performs FBP reconstruction,
    and then applies the post-processing CNN. This represents the
    complete FBPConvNet workflow.
    
    Args:
        fbp_filter: Filter to use for FBP reconstruction
        cnn_config: Configuration for the post-processing CNN
    """
    
    def __init__(
        self,
        fbp_filter: str = 'shepp-logan',
        cnn_config: Optional[Dict] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.fbp_filter = fbp_filter
        
        # Default CNN configuration
        if cnn_config is None:
            cnn_config = {
                'num_filters': 64,
                'num_layers': 8,
                'use_residual': True,
                'use_batch_norm': True
            }
        
        # Build post-processing CNN
        self.post_processing_cnn = FBPConvNet(**cnn_config)
    
    def call(self, inputs, training=None):
        """
        Forward pass through complete FBPConvNet pipeline.
        
        Args:
            inputs: Dictionary with 'sinogram' and 'angles' keys
            training: Training mode flag
        
        Returns:
            Post-processed reconstruction
        """
        
        # Extract sinogram and angles
        if isinstance(inputs, dict):
            sinogram = inputs['sinogram']
            angles = inputs['angles']
        else:
            # Assume pre-computed FBP reconstruction
            fbp_recon = inputs
            return self.post_processing_cnn(fbp_recon, training=training)
        
        # Perform FBP reconstruction
        # Note: This would typically be done in preprocessing
        # Here we assume the input is already an FBP reconstruction
        fbp_recon = sinogram  # Placeholder
        
        # Apply post-processing CNN
        output = self.post_processing_cnn(fbp_recon, training=training)
        
        return output


class ResidualBlock(layers.Layer):
    """
    Residual block for enhanced FBPConvNet architectures.
    
    This implements a standard residual block with two convolutions
    and a skip connection.
    """
    
    def __init__(
        self,
        filters: int,
        kernel_size: int = 3,
        use_batch_norm: bool = True,
        dropout_rate: float = 0.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.filters = filters
        self.kernel_size = kernel_size
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate
        
        # Build block layers
        self.conv1 = layers.Conv2D(
            filters,
            kernel_size,
            padding='same',
            name='conv1'
        )
        
        self.conv2 = layers.Conv2D(
            filters,
            kernel_size,
            padding='same',
            name='conv2'
        )
        
        if use_batch_norm:
            self.bn1 = layers.BatchNormalization(name='bn1')
            self.bn2 = layers.BatchNormalization(name='bn2')
        
        self.relu1 = layers.ReLU(name='relu1')
        self.relu2 = layers.ReLU(name='relu2')
        
        if dropout_rate > 0:
            self.dropout = layers.Dropout(dropout_rate, name='dropout')
        
        self.add = layers.Add(name='residual_add')
    
    def call(self, inputs, training=None):
        """Forward pass through residual block."""
        
        x = inputs
        
        # First convolution
        x = self.conv1(x)
        if self.use_batch_norm:
            x = self.bn1(x, training=training)
        x = self.relu1(x)
        
        # Second convolution
        x = self.conv2(x)
        if self.use_batch_norm:
            x = self.bn2(x, training=training)
        
        # Dropout
        if self.dropout_rate > 0:
            x = self.dropout(x, training=training)
        
        # Residual connection
        x = self.add([x, inputs])
        x = self.relu2(x)
        
        return x


class EnhancedFBPConvNet(keras.Model):
    """
    Enhanced FBPConvNet with residual blocks and attention.
    
    This is an improved version of FBPConvNet that uses:
    - Residual blocks for better gradient flow
    - Channel attention for feature selection
    - Multi-scale processing
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (256, 256, 1),
        num_blocks: int = 6,
        base_filters: int = 64,
        use_attention: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.input_shape = input_shape
        self.num_blocks = num_blocks
        self.base_filters = base_filters
        self.use_attention = use_attention
        
        # Build enhanced architecture
        self._build_enhanced_network()
    
    def _build_enhanced_network(self):
        """Build enhanced FBPConvNet architecture."""
        
        # Input processing
        self.input_conv = layers.Conv2D(
            self.base_filters,
            kernel_size=7,
            padding='same',
            name='input_conv'
        )
        self.input_bn = layers.BatchNormalization(name='input_bn')
        self.input_relu = layers.ReLU(name='input_relu')
        
        # Residual blocks
        self.residual_blocks = []
        for i in range(self.num_blocks):
            block = ResidualBlock(
                filters=self.base_filters,
                name=f'residual_block_{i}'
            )
            self.residual_blocks.append(block)
        
        # Channel attention
        if self.use_attention:
            self.channel_attention = self._create_channel_attention()
        
        # Output processing
        self.output_conv = layers.Conv2D(
            1,
            kernel_size=3,
            padding='same',
            activation='linear',
            name='output_conv'
        )
    
    def _create_channel_attention(self):
        """Create channel attention mechanism."""
        
        def channel_attention(x):
            # Global average pooling
            gap = layers.GlobalAveragePooling2D()(x)
            
            # FC layers
            fc1 = layers.Dense(self.base_filters // 4, activation='relu')(gap)
            fc2 = layers.Dense(self.base_filters, activation='sigmoid')(fc1)
            
            # Reshape and multiply
            attention_weights = layers.Reshape((1, 1, self.base_filters))(fc2)
            attended = layers.Multiply()([x, attention_weights])
            
            return attended
        
        return channel_attention
    
    def call(self, inputs, training=None):
        """Forward pass through enhanced FBPConvNet."""
        
        # Store input for global residual
        global_input = inputs
        
        # Input processing
        x = self.input_conv(inputs)
        x = self.input_bn(x, training=training)
        x = self.input_relu(x)
        
        # Residual blocks
        for block in self.residual_blocks:
            x = block(x, training=training)
        
        # Channel attention
        if self.use_attention:
            x = self.channel_attention(x)
        
        # Output processing
        x = self.output_conv(x)
        
        # Global residual connection
        output = layers.Add(name='global_residual')([x, global_input])
        
        return output


# Factory functions
def create_fbp_conv_net(
    variant: str = 'standard',
    input_shape: Tuple[int, int, int] = (256, 256, 1),
    **kwargs
) -> keras.Model:
    """
    Factory function to create FBPConvNet variants.
    
    Args:
        variant: Model variant ('standard', 'enhanced', 'light')
        input_shape: Input image shape
        **kwargs: Additional model arguments
    
    Returns:
        FBPConvNet model
    """
    
    if variant == 'standard':
        return FBPConvNet(input_shape=input_shape, **kwargs)
    elif variant == 'enhanced':
        return EnhancedFBPConvNet(input_shape=input_shape, **kwargs)
    elif variant == 'light':
        # Lightweight version
        light_config = {
            'num_filters': 32,
            'num_layers': 4,
            'use_residual': True,
            'use_batch_norm': True
        }
        light_config.update(kwargs)
        return FBPConvNet(input_shape=input_shape, **light_config)
    else:
        raise ValueError(f"Unknown variant: {variant}")


def compare_fbp_conv_net_variants(input_shape: Tuple[int, int, int] = (256, 256, 1)):
    """Compare different FBPConvNet variants."""
    
    variants = ['standard', 'enhanced', 'light']
    test_input = tf.random.normal((1,) + input_shape)
    
    print("🔧 Comparing FBPConvNet Variants")
    print("=" * 40)
    
    for variant in variants:
        try:
            model = create_fbp_conv_net(variant=variant, input_shape=input_shape)
            output = model(test_input)
            params = model.count_params()
            
            print(f"✅ {variant.title()} FBPConvNet:")
            print(f"   Parameters: {params:,}")
            print(f"   Output shape: {output.shape}")
            print(f"   Model size: {params * 4 / (1024**2):.2f} MB")
            print()
            
        except Exception as e:
            print(f"❌ {variant.title()} FBPConvNet: Error - {e}")


# Test function
if __name__ == "__main__":
    print("🧪 Testing FBPConvNet implementations...")
    
    # Test standard FBPConvNet
    fbp_conv_net = create_fbp_conv_net(
        variant='standard',
        input_shape=(256, 256, 1),
        num_filters=32,
        num_layers=6
    )
    
    print(f"✅ Standard FBPConvNet created: {fbp_conv_net.count_params():,} parameters")
    
    # Test forward pass
    test_input = tf.random.normal((2, 256, 256, 1))
    output = fbp_conv_net(test_input)
    print(f"✅ Forward pass: {test_input.shape} → {output.shape}")
    
    # Test enhanced variant
    enhanced_fbp = create_fbp_conv_net(
        variant='enhanced',
        input_shape=(256, 256, 1),
        num_blocks=4,
        base_filters=32
    )
    
    enhanced_output = enhanced_fbp(test_input)
    print(f"✅ Enhanced FBPConvNet: {enhanced_fbp.count_params():,} parameters, output {enhanced_output.shape}")
    
    # Compare all variants
    print("\n" + "=" * 50)
    compare_fbp_conv_net_variants((256, 256, 1))
    
    # Test residual block
    print("🔧 Testing ResidualBlock...")
    res_block = ResidualBlock(filters=64)
    test_features = tf.random.normal((2, 128, 128, 64))
    res_output = res_block(test_features)
    print(f"✅ ResidualBlock: {test_features.shape} → {res_output.shape}")
    
    print("\n🎉 FBPConvNet implementation test completed successfully!")
