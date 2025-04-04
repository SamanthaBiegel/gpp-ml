import torch.nn as nn
import torch

class Model(nn.Module):
    def __init__(self, input_dim, conditional_dim=0, hidden_dim=64):
        """
        Initialize the Model class.
        Args:
            input_dim (int): The number of input features.
            conditional_dim (int): The number of conditional features. Default is 0.
            hidden_dim (int): The number of hidden units in the first layer. Default is 64.
        """
        super().__init__()

        self.conditional = conditional_dim > 0

        current_dim = hidden_dim

        # First layer maps from input_dim+conditional_dim to hidden_dim
        layers = [nn.Sequential(
            nn.Linear(input_dim+conditional_dim, hidden_dim),
            nn.ReLU()
        )]
        
        # Create additional layers that half the dimension each time until it reaches 16
        while current_dim > 16:
            next_dim = max(current_dim // 2, 16)
            layer = nn.Sequential(
                nn.Linear(current_dim, next_dim),
                nn.ReLU()
            )
            layers.append(layer)
            current_dim = next_dim
        
        # Save all layers in an nn.ModuleList
        self.layers = nn.ModuleList(layers)
        
        # Final linear layer for regression output to 1
        self.final_layer = nn.Linear(current_dim, 1)
        
    def forward(self, x, c=None):
        # If conditional tensor is provided, concatenate it to the input tensor
        if self.conditional:
            x = torch.cat([x, c], dim=2)

        # Pass the input tensor through all layers
        for layer in self.layers:
            x = layer(x)

        # Pass the output through the final layer
        x = self.final_layer(x)

        return x