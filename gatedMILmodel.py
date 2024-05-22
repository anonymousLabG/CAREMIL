from torch.nn import Sequential, Module, Linear, ReLU, \
    Sigmoid, Tanh, Softmax, ELU, BatchNorm1d, Dropout, MultiheadAttention
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch

class GatedMIL(nn.Module):

    def __init__(self, embed_dim, mil_hl_size, mil_dim, output_dim, n_classes, classifier_hl_size=64, dropout=0.3) -> None:
        super().__init__()


        # Networks for the MIL layer
        self.tanh_layer = Sequential(
            Linear(embed_dim, mil_hl_size),
            ELU(),
            Dropout(dropout),
            Linear(mil_hl_size, mil_dim),
            Tanh()
        )

        self.sigmoid_layer = Sequential(
            Linear(embed_dim, mil_hl_size),
            ELU(),
            Dropout(dropout),
            Linear(mil_hl_size, mil_dim),
            Sigmoid()
        )

        self.w = Sequential(
            Linear(mil_dim, 32),
            ELU(),
            Linear(32, 1)
        )    

        self.value_layer = Sequential(
            Linear(embed_dim, mil_hl_size),
            ELU(),
            Dropout(dropout),
            Linear(mil_hl_size, output_dim)
        )

        # Network for the final classification layer
        self.classification_layer = Sequential(
            Linear(output_dim, classifier_hl_size),
            ELU(),
            Dropout(dropout),
            BatchNorm1d(classifier_hl_size),
            Linear(classifier_hl_size, 32),
            ELU(),
            Linear(32, n_classes), 
            Softmax(dim=1)
        )

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def forward(self, x, padding_lengths):
        # x is of shape B x T_max x E

        tanh_layer = self.tanh_layer(x)     # B x T_max x mil_dim
        sigm_layer = self.sigmoid_layer(x)  # B x T_max x mil_dim
        value_layer = self.value_layer(x)   # B x T_max x output_dim

        attn = self.w(tanh_layer * sigm_layer)  # B x T_max x 1
        attn = attn.squeeze(2)  # B x T_max

        padding_mask = self.generate_mask(padding_lengths, \
                                          T_max=x.shape[1]) # B x T_max

        attn = attn.masked_fill(padding_mask == 0, float('-inf'))  # B x T_max
        attn = F.softmax(attn, dim=1)   # B x T_max

        # weighted_bag_embeddings will be of shape B x output_dim
        # Reshape attn to B x 1 x T_max for batch matrix multiplication
        # Value layer is of shape B x T_max x output_dim
        # After matrix multiplication, we will get B x 1 x output_dim which we will squeeze to get B x output_dim
        weighted_bag_embeddings = torch.bmm(attn.unsqueeze(1), value_layer).squeeze(1)

        pred_probs = self.classification_layer(weighted_bag_embeddings)  # B x n_classes

        return pred_probs, attn.unsqueeze(1)   # B x n_classes, B x 1 x T_max

        

    def generate_mask(self, padding_lengths, T_max):
        # This function will generate a mask of shape B x T_max x 1, where every row 
        # will have 1s for the non-padded queries and 0s for the padded queries.
        # This mask will be used to mask the padded queries in the attention weights.
        # padding_lengths is of shape B

        mask = torch.zeros((len(padding_lengths), T_max), device=self.device)
        for i, length in enumerate(padding_lengths):
            mask[i, :length] = 1

        return mask

'''
# Example Usage
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gatedmodel = GatedMIL(embed_dim=1000, mil_hl_size=256, mil_dim=128, output_dim=128,\
                      classifier_hl_size=64, n_classes=3).to(device)

inp_tensor = torch.randn(32, 123, 1000).to(device)
padding_lengths = np.random.randint(80, 123, 32)

bag_probs, attn = gatedmodel(inp_tensor, padding_lengths)
print(attn.shape)
print(bag_embeddings.shape)
'''
