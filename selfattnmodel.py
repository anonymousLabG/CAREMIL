from torch.nn import Sequential, Module, Linear, ReLU, \
    Sigmoid, Tanh, Softmax, ELU, BatchNorm1d, Dropout, MultiheadAttention, Flatten
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import torch

# TODO: Complete this shit!
# AggregateConcatenate Network. This will create a aggregator network for all cells and an 
# adj_input network for all cells. Output from aggregator newtork will be aggregated
# concatenated with the output from the input network.
class AggregateConcatenate(Module):
    def __init__(self, init_embed_size, hidden_size, agg_out_size, agg_method='normal', dropout=0.2):
        super().__init__()

        # This dict will have the num of heads to use for each aggregation method.
        agg_dict = {'normal': 4,\
                    'gm': 3,\
                    'lse': 2}

        self.num_heads = agg_dict[agg_method]

        # We will flatten the input first so our final output will be (B x T_max) x agg_out_size
        self.aggregators = Sequential(
            Flatten(start_dim=0, end_dim=1),
            Linear(init_embed_size, hidden_size),
            ELU(),
            BatchNorm1d(hidden_size),
            Dropout(dropout),
            Linear(hidden_size, agg_out_size), 
            Tanh()
        )

        self.adj_input_network = Sequential(
            Flatten(start_dim=0, end_dim=1),
            Linear(init_embed_size, hidden_size),
            ELU(),
            BatchNorm1d(hidden_size),
            Dropout(dropout),
            Linear(hidden_size, agg_out_size),
            Tanh()
        )
    
    def normal_aggregation(self, queries, padding_lengths):
        # queries is of shape B x T_max x agg_out_size
        # We will use (n=4) aggregation functions, mean, max and min and standard deviation which will be applied on all the queries in a bag.
        # The aggregations will only be applied to the non-padded queries.
        # Every bag will be represented by a tensor of shape n x agg_out_size, where every row is the aggregation of the queries in the bag.
        # The batch will be represented by a tensor of shape B x n x agg_out_size, which will consist of the bag representations of every bag in the batch.

        batches = queries.shape[0]
        batch_representations = []

        for b in range(batches):
            bag_representation = []
            non_padded_queries = queries[b, :padding_lengths[b], :] # T_max x agg_out_size

            bag_representation.append(torch.mean(non_padded_queries, dim=0))
            bag_representation.append(torch.max(non_padded_queries, dim=0).values)
            bag_representation.append(torch.min(non_padded_queries, dim=0).values)
            bag_representation.append(torch.std(non_padded_queries, dim=0))

            bag_representation = torch.stack(bag_representation)
            batch_representations.append(bag_representation)

        batch_representations = torch.stack(batch_representations)
        return batch_representations # B x n x agg_out_size

    def gm_aggregation(self, batch_aggregators, padding_lengths):
        # aggregators is of shape B x T_max x agg_out_size
        # The aggregations will only be applied to the non-padded queries.
        # Every bag will be represented by a tensor of shape n x head_size, where every row is the aggregation of the queries in the bag.
        # The output will be represented by a tensor of shape B x n x agg_out_size. 

        batch_size = batch_aggregators.shape[0]
        batch_representations = []

        for bag in range(batch_size):
            bag_representation = []
            # Take the non-padded queries from the tensor which will have a shape of T x agg_out_size
            bag_non_padded_aggs = batch_aggregators[bag, :padding_lengths[bag], :] # T x agg_out_size

            # Aggregate the queries using different functions   
            bag_representation.append(self.generalized_mean(bag_non_padded_aggs, power=5.0))
            bag_representation.append(self.generalized_mean(bag_non_padded_aggs, power=2.5))
            bag_representation.append(self.generalized_mean(bag_non_padded_aggs, power=1.0))

            bag_representation = torch.stack(bag_representation)    # n x agg_out_size
            batch_representations.append(bag_representation)
        
        return torch.stack(batch_representations) # B x n x agg_out_size
    
    def lse_aggregation(self, queries, padding_lengths):
        # queries is of shape B x T_max x agg_out_size
        # The aggregations will only be applied to the non-padded queries.
        # Every bag will be represented by a tensor of shape n x agg_out_size, where every row is the aggregation of the queries in the bag.
        # The batch will be represented by a tensor of shape B x n x agg_out_size.

        batch_size = queries.shape[0]
        batch_representations = []

        for bag in range(batch_size):
            bag_representation = []
            # Take the non-padded queries from the tensor which will have a shape of T x head_size
            non_padded_queries = queries[bag, :padding_lengths[bag], :]

            # Aggregate the queries using different functions
            bag_representation.append(self.log_sum_exponentiation(non_padded_queries, power=5.0))
            bag_representation.append(self.log_sum_exponentiation(non_padded_queries, power=2.5))
            # bag_representation.append(self.log_sum_exponentiation(non_padded_queries, power=0.0))

            bag_representation = torch.stack(bag_representation)
            batch_representations.append(bag_representation)

        return torch.stack(batch_representations) # B x n x head_size

    def generalized_mean(self, inp_tensor, power=1.0):
        # inp_tensor is of shape T x agg_out_size
        # We will take the generalized mean of the tensor along the zeroth dimension.
        # The generalized mean is defined as (1/n * sum(x_i^p))^(1/p), where p is the power parameter.
        # For p=0, it is the geometric mean, for p=1, it is the arithmetic mean and for p=infinity, it is the max function.
        # Output shape: agg_out_size

        n = inp_tensor.shape[0]
        generalized_mean = (1/n * torch.sum(inp_tensor**power, dim=0))**(1/power)
        return generalized_mean
    
    def log_sum_exponentiation(self, inp_tensor, power=1.0):
        # inp_tensor is of shape T x agg_out_size
        # We will take the log sum exponentiation of the tensor along the zeroth dimension.
        # The log sum exponentiation is defined as (1/p)*log((1/n) * sum(e^(p*x_i)), where p is the power parameter.
        # The output shape is agg_out_size

        n = inp_tensor.shape[0]
        sum_exp = (1/n) * torch.sum(torch.exp(power*inp_tensor), dim=0)
        log_sum_exp = (1/power) * torch.log(sum_exp)
        return log_sum_exp

    def forward(self, x, padding_lengths):
        # x is of shape B x T_max x E
        # We will pass the bag through aggregator and the adjacency network.
        # The input will be flattened in these networks so that pytorch can properly apply the linear layers.
        # We reshape the outputs of the aggregator and the adjacency network to get a shape of B x T_max x agg_out_size.

        B = x.shape[0]
        T_max = x.shape[1]

        # Aggregate all the non padded features of the bags.
        aggregations = self.aggregators(x) # (B x T_max) x agg_out_size
        aggregations = aggregations.view(B, T_max, -1) # B x T_max x agg_out_size

        bag_aggregations = self.normal_aggregation(aggregations, padding_lengths)   # B x n x agg_out_size

        # Pass all cell embeddings through another network to transform its embedding shape.
        adjacent_inputs = self.adj_input_network(x) # (B x T_max) x agg_out_size
        adjacent_inputs = adjacent_inputs.view(B, T_max, -1) # B x T_max x agg_out_size

        # Concatenate the bag aggregations and the adjacent inputs to get a shape of B x (n+T_max) x agg_out_size
        concatenated = torch.cat((bag_aggregations, adjacent_inputs), dim=1) # B x (n+T_max) x agg_out_size

        return concatenated


# Multi head Self Attention Model that will create a key, query, value and apply attention mechanism.
class MultiheadAttention(nn.Module):
    def __init__(self, inp_embedding_size, attn_head_size, num_attn_heads, n, dropout=0.3):
        super(MultiheadAttention, self).__init__()

        # number of aggregation heads in the concatenated length of the sequence.
        self.num_agg_heads = n

        self.key = Sequential(
            Linear(inp_embedding_size, 128),
            ELU(),
            Dropout(dropout),
            Linear(128, attn_head_size),
            Tanh()
        )

        self.query = Sequential(
            Linear(inp_embedding_size, 128),
            ELU(),
            Dropout(dropout),
            Linear(128, attn_head_size),
            Sigmoid()
        )

        self.value = Sequential(
            Linear(inp_embedding_size, 128),
            ELU(),
            Dropout(dropout),
            Linear(128, attn_head_size),
        )

        self.multihead_attention = nn.MultiheadAttention(embed_dim=attn_head_size, \
                                        num_heads=num_attn_heads, dropout=dropout, batch_first=True)

    def forward(self, input_tensor):
        # input_tensor is of shape B x L x E    where L is the length of the sequence and E is the embedding size.
        # In our case L = (n + T_max) and E = head_size1

        key = self.key(input_tensor)    # B x L x head_size2
        query = self.query(input_tensor)    # B x L x head_size2
        value = self.value(input_tensor)    # B x L x outsize

        # Use the same input for key, query, and value
        full_output, attn = self.multihead_attention(query, key, value) # B x (n+T_max) x outsize, B x (n+T_max) x (n+T_max)

        # Extract the attn weights from the heads to the T_max cells.
        head_attn = attn[:, :self.num_agg_heads, self.num_agg_heads:] # B x n x T_max

        # Extract the head embeddings from the full output.
        head_output = full_output[:, :self.num_agg_heads, :] # B x n x outsize

        return head_output, full_output, attn, head_attn

# MIL with self attention model
class MILSelfAttention(Module):
    def __init__(self, init_mil_embed, mil_head, n_classes, attn_head_size,\
                 agg_method = 'normal'):
        super().__init__()

        # This dict will have the num of heads to use for each aggregation method.
        agg_dict = {'normal': 4,\
                    'gm': 3,\
                    'lse': 2}
        
        # The output size of the attention layer will be the same as the attn_head_size.
        self.outsize = attn_head_size   
        
        self.num_agg_heads = agg_dict[agg_method]
        self.aggregation = AggregateConcatenate(init_mil_embed, 256, mil_head, agg_method, dropout=0.2)   # B x (n+T_max) x mil_head

        self.attention1 = MultiheadAttention(inp_embedding_size=mil_head, attn_head_size=attn_head_size, \
                                                num_attn_heads=1, n=self.num_agg_heads,\
                                                dropout=0.2)
        
        self.attention2 = MultiheadAttention(inp_embedding_size=attn_head_size, attn_head_size=attn_head_size, \
                                                num_attn_heads=1, n=self.num_agg_heads,\
                                                dropout=0.2)
        
        self.classifier = Sequential(
            Linear(self.num_agg_heads*attn_head_size, 128),
            ELU(),
            BatchNorm1d(128),
            Dropout(0.3),

            # Linear(hl_size, hl_size),
            # ELU(),
            # BatchNorm1d(256),
            # Dropout(0.3),

            Linear(128, 64),
            ELU(),
            BatchNorm1d(64),
            Dropout(0.3),

            Linear(64, n_classes),
            Softmax(dim=1)
        )
        

    def forward(self, x, padding_lengths):
        # x is of shape B x T_max x E
        # print('Input shape: ', x.shape)
        concatenated = self.aggregation(x, padding_lengths) # B x (n+T_max) x agg_out_size
        # print('Concatenated Shape: ', concatenated.shape)

        head_out1, full_out1, attn1, head_attn1 = self.attention1(concatenated)
        # print('Output of out1: ', full_out1.shape)
        # print('Attn1: ', attn1.shape)
        # print('Head Attn1: ', head_attn1.shape)

        head_out2, full_out2, attn2, head_attn2 = self.attention2(full_out1)
        # print('Full_out2: ', full_out2.shape)   # B x (n+T_max) x outsie
        # print('Head Out2: ', head_out2.shape)   # B x n x outsize
        # print('Attn2: ', attn2.shape)           # B x (n+T_max) x (n+T_max)
        # print('Head Attn2: ', head_attn2.shape) # B x n x T_max

        # Flatten the head_out2 to get a shape of B x (n*outsize)
        B = head_out2.shape[0]
        head_out2 = torch.flatten(head_out2, start_dim=1) # B x (n*outsize)
        # print('Flattened Head Out2: ', head_out2.shape)

        # Pass the flattened head_out2 through the classifier
        pred_probs = self.classifier(head_out2)
        # print('Pred Probs: ', pred_probs.shape) # B x n_classes

        # Return the predictions and the attention weights with shapes B x n_classes and B x n x T_max
        return pred_probs, head_attn2



'''
# Example Usage:

# Model hyperparameters used in the paper
init_mil_embed = 1000   # The initial embedding size of the MIL cells
mil_head = 256          # The output size of the aggregation network
attn_head = 128         # The output size of the attention network
mil_aggregation_method = 'gm'   # The aggregation method to use for the MIL cells
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_classes = 2


model = MILSelfAttention(init_mil_embed, mil_head, n_classes, attn_head_size=attn_head, \
                            agg_method=mil_aggregation_method).to(device)

# Example input to the model
B = 32      # Batch size
E = 1000    # The embedding size of the cells
example_padding_lengths = torch.randint(200, 300, (B,)).to(device)  # The padding lengths of the bags in the batch, the data input \
                                                                    # should be zero padded after this length.
T_max = torch.max(example_padding_lengths).item() # The maximum number of cells in a bag, across all the bags in the batch

example_data_input = torch.randn(B, T_max, E).to(device)    # The data input to the model (just for reference, zero padding not done)

pred_probs, head_attn = model(example_data_input, example_padding_lengths)
print('Predictions: ', pred_probs.shape)
print('Attention Weights: ', head_attn.shape)
'''


