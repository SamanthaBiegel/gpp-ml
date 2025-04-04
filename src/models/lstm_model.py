import torch.nn as nn
import torch

class LayerNormLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        """"
        "Initialize the LayerNormLSTMCell class.
        Args:
            input_dim (int): The number of input features.
            hidden_dim (int): The number of hidden units.
        """
        super(LayerNormLSTMCell, self).__init__()
        self.input_weights = nn.Linear(input_dim, 4 * hidden_dim)
        self.hidden_weights = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.layer_norm = nn.LayerNorm(4 * hidden_dim)

    def forward(self, input, hidden):
        h, c = hidden
        gates = self.input_weights(input) + self.hidden_weights(h)
        gates = self.layer_norm(gates)
        input_gate, forget_gate, cell_gate, output_gate = gates.chunk(4, 1)
        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate)
        cell_gate = torch.tanh(cell_gate)
        output_gate = torch.sigmoid(output_gate)
        c_next = forget_gate * c + input_gate * cell_gate
        h_next = output_gate * torch.tanh(c_next)
        return h_next, c_next

class LayerNormLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.0):
        """
        Initialize the LayerNormLSTM class.
        Args:
            input_size (int): The number of input features.
            hidden_size (int): The number of hidden units.
            num_layers (int): The number of LSTM layers.
            dropout (float): Dropout probability. Default is 0.0.
        """
        super(LayerNormLSTM, self).__init__()
        self.hidden_dim = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.cells = nn.ModuleList([LayerNormLSTMCell(input_size if i == 0 else hidden_size, hidden_size) for i in range(num_layers)])

    def forward(self, x):
        h = [torch.zeros(x.size(0), self.hidden_dim).to(x.device) for _ in range(self.num_layers)]
        c = [torch.zeros(x.size(0), self.hidden_dim).to(x.device) for _ in range(self.num_layers)]
        outputs = []
        for t in range(x.size(1)):
            input = x[:, t, :]
            for i, cell in enumerate(self.cells):
                h[i], c[i] = cell(input, (h[i], c[i]))
                input = h[i]
                if i < self.num_layers - 1:
                    input = self.dropout(input)
            outputs.append(h[-1].unsqueeze(1))
        outputs = torch.cat(outputs, dim=1)
        return outputs, (torch.stack(h), torch.stack(c))


class Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, conditional_dim=0, num_layers=2, dropout=0.3, num_heads=8, attention=False, layernorm=True):
        """
        Initialize the Model class.
        Args:
            input_dim (int): The number of input features.
            hidden_dim (int): The number of hidden units.
            conditional_dim (int): The number of conditional features. Default is 0.
            num_layers (int): The number of LSTM layers. Default is 2.
            dropout (float): Dropout probability. Default is 0.3.
            num_heads (int): Number of attention heads. Default is 8.
            attention (bool): Whether to use attention mechanism. Default is False.
            layernorm (bool): Whether to use LayerNorm in LSTM. Default is True.
        """
        super().__init__()

        self.conditional = conditional_dim > 0

        # LSTM layer for sequence processing
        if layernorm:
            self.lstm = LayerNormLSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, dropout=dropout)
        else:
            self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)

        self.attention = attention
        if attention:
            # Multi-head attention mechanism to capture dependencies
            self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, 
                                                num_heads=num_heads,
                                                batch_first=True, 
                                                dropout=dropout)
            
            self.final = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, 1),
            )
        else:
            if self.conditional:
                initial_dim = hidden_dim + conditional_dim
            else:
                initial_dim = hidden_dim

            current_dim = max(initial_dim // 2, 16)

            layers = [nn.Sequential(
                nn.Linear(initial_dim, current_dim),
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
    
    def generate_causal_mask(self, size):
        mask = torch.tril(torch.ones(size, size)) == 0
        return mask.to(torch.bool)
        
    def forward(self, x, c=None):
        # Forward pass through the LSTM layer
        out, _ = self.lstm(x)

        if self.attention:
            _, seq_length, _ = x.shape

            # Generate causal mask
            causal_mask = self.generate_causal_mask(seq_length).to(x.device)

            # Apply multi-head attention mechanism over LSTM outputs with causal mask
            attn_output, _ = self.attention(out, out, out, attn_mask=causal_mask)
            attn_output = attn_output + out  # Residual connection

            y = self.final(attn_output)
        else:
            if self.conditional:
                out = torch.cat([out, c], dim=2)
            
            # Pass the output through the rest of the layers
            for layer in self.layers:
                out = layer(out)
            y = self.final_layer(out)
        
        return y
