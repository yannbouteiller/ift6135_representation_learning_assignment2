
#%%
import torch
import torch.nn as nn

import numpy as np
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt


def clones(module, N):
    "A helper function for producing N identical layers (each with their own parameters)."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

# Problem 1
class RNN(nn.Module):

    def __init__(self, emb_size, hidden_size, seq_len, batch_size, vocab_size, num_layers, dp_keep_prob,save_hiddens=False):
        """
        emb_size:     The numvwe of units in the input embeddings
        hidden_size:  The number of hidden units per layer
        seq_len:      The length of the input sequences
        vocab_size:   The number of tokens in the vocabulary (10,000 for Penn TreeBank)
        num_layers:   The depth of the stack (i.e. the number of hidden layers at
                      each time-step)
        dp_keep_prob: The probability of *not* dropping out units in the
                      non-recurrent connections.
                      Do not apply dropout on recurrent connections.
        """
        super(RNN, self).__init__()

        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.dp_keep_prob = dp_keep_prob
        self.batch_size = batch_size
        self.hiddens = None
        self.save_hiddens = save_hiddens
        
        self.embeddings = nn.Embedding(self.vocab_size,self.emb_size)

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(emb_size + hidden_size, hidden_size))
        self.layers.extend(clones(nn.Linear(2*hidden_size, hidden_size),num_layers-1))
        
        self.drop_outs = clones(nn.Dropout(1 - self.dp_keep_prob),num_layers+1)
        self.out_layer = nn.Linear(hidden_size, vocab_size)

        self.init_weights_uniform()
        
    def init_weights_uniform(self):

        nn.init.uniform_(self.embeddings.weight, -0.1, 0.1)

        for i in range(self.num_layers):
            b = 1/math.sqrt(self.hidden_size)
            nn.init.uniform_(self.layers[i].weight,-b, b)
            nn.init.uniform_(self.layers[i].bias,-b, b)

        nn.init.uniform_(self.out_layer.weight,-0.1, 0.1)
        nn.init.zeros_(self.out_layer.bias)
        
    def init_hidden(self):
        """
        This is used for the first mini-batch in an epoch, only.
        """
        return torch.zeros(self.num_layers,self.batch_size,self.hidden_size) # a parameter tensor of shape (self.num_layers, self.batch_size, self.hidden_size)

    def forward(self, inputs, hidden):
        """
        Arguments:
            - inputs: A mini-batch of input sequences, composed of integers that
                        represent the index of the current token(s) in the vocabulary.
                            shape: (seq_len, batch_size)
            - hidden: The initial hidden states for every layer of the stacked RNN.
                            shape: (num_layers, batch_size, hidden_size)

        Returns:
            - Logits for the softmax over output tokens at every time-step.
                  **Do NOT apply softmax to the outputs!**
                  Pytorch's CrossEntropyLoss function (applied in ptb-lm.py) does
                  this computation implicitly.
                        shape: (seq_len, batch_size, vocab_size)
            - The final hidden states for every layer of the stacked RNN.
                  These will be used as the initial hidden states for all the
                  mini-batches in an epoch, except for the first, where the return
                  value of self.init_hidden will be used.
                  See the repackage_hiddens function in ptb-lm.py for more details,
                  if you are curious.
                        shape: (num_layers, batch_size, hidden_size)
        """

        if inputs.is_cuda:
         	device = inputs.get_device()
        else:
            device = torch.device("cpu")

        embed_out = self.embeddings(inputs)# shape (seq_len,batch_size,emb_size)
        
        logits = torch.zeros(self.seq_len, self.batch_size, self.vocab_size).to(device)
        
        if self.save_hiddens:
            self.hiddens = []
        
        for timestep in range(self.seq_len):
            input_ = self.drop_outs[-1](embed_out[timestep])
            new_hidden = torch.zeros( self.num_layers,self.batch_size,self.hidden_size).to(device)
            for layer in range(self.num_layers):
                new_hidden[layer] = torch.tanh(self.layers[layer](torch.cat([input_,hidden[layer]],1)))
                hidden[layer] = new_hidden[layer]
                input_ = self.drop_outs[layer](hidden[layer].clone())
            
            if self.save_hiddens:
                self.hiddens.append(new_hidden)

            logits[timestep] = self.out_layer(input_)

        return logits, hidden

    def generate(self, inputs, hidden, generated_seq_len,temp=1):
        """
        Arguments:
            - input: A mini-batch of input tokens (NOT sequences!)
                            shape: (batch_size)
            - hidden: The initial hidden states for every layer of the stacked RNN.
                            shape: (num_layers, batch_size, hidden_size)
            - generated_seq_len: The length of the sequence to generate.
                           Note that this can be different than the length used
                           for training (self.seq_len)
        Returns:
            - Sampled sequences of tokens
                        shape: (generated_seq_len, batch_size)
        """

        if inputs.is_cuda:
         	device = inputs.get_device()
        else:
            device = torch.device("cpu")

        results = torch.zeros(generated_seq_len, self.batch_size).to(device)
        
        input_tokens = inputs
        for timestep in range(generated_seq_len):
            input_ = self.embeddings(input_tokens)
            for layer in range(self.num_layers):
                hidden[layer] = torch.tanh(self.layers[layer](torch.cat([input_,hidden[layer]],1)))
                input_ = hidden[layer]
            input_tokens = torch.squeeze(torch.multinomial(torch.softmax(self.out_layer(input_)/temp,1),1))
            results[timestep,:] = input_tokens

        return results, hidden

# Problem 2
class GRU(nn.Module): # Implement a stacked GRU RNN
    """
    Follow the same instructions as for RNN (above), but use the equations for
    GRU, not Vanilla RNN.
    """
    def __init__(self, emb_size, hidden_size, seq_len, batch_size, vocab_size, num_layers, dp_keep_prob, save_hiddens=False):

        super(GRU, self).__init__()

        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.dp_keep_prob = dp_keep_prob
        self.batch_size = batch_size
        self.hiddens = None
        self.save_hiddens = save_hiddens
        
        self.embeddings = nn.Embedding(self.vocab_size,self.emb_size)

        self.r = nn.ModuleList()
        self.r.append(nn.Linear(emb_size + hidden_size, hidden_size))
        self.r.extend(clones(nn.Linear(2*hidden_size, hidden_size),num_layers-1))

        self.z = nn.ModuleList()
        self.z.append(nn.Linear(emb_size + hidden_size, hidden_size))
        self.z.extend(clones(nn.Linear(2*hidden_size, hidden_size),num_layers-1))

        self.h = nn.ModuleList()
        self.h.append(nn.Linear(emb_size + hidden_size, hidden_size))
        self.h.extend(clones(nn.Linear(2*hidden_size, hidden_size),num_layers-1))
    
        self.drop_outs = clones(nn.Dropout(1 - self.dp_keep_prob),num_layers+1)
        self.out_layer = nn.Linear(hidden_size, vocab_size)

        self.init_weights_uniform()
        # TODO ========================

    def init_weights_uniform(self):
        # TODO ========================
        nn.init.uniform_(self.embeddings.weight, -0.1, 0.1)

        for layers in [self.r,self.z,self.h]:
            for i in range(self.num_layers):
                b = 1/math.sqrt(self.hidden_size)
                nn.init.uniform_(layers[i].weight,-b, b)
                nn.init.uniform_(layers[i].bias,-b, b)

        nn.init.uniform_(self.out_layer.weight,-0.1, 0.1)
        nn.init.zeros_(self.out_layer.bias)
        
    def init_hidden(self):
        # TODO ========================
        return torch.zeros(self.num_layers,self.batch_size,self.hidden_size,requires_grad=True) # a parameter tensor of shape (self.num_layers, self.batch_size, self.hidden_size)

    def forward(self, inputs, hidden):
        # TODO ========================
        if inputs.is_cuda:
        	device = inputs.get_device()
        else:
            device = torch.device("cpu")

        embed_out = self.embeddings(inputs) # shape (seq_len,batch_size,emb_size)
        
        if self.save_hiddens:
            self.hiddens = []

        logits = torch.zeros(self.seq_len, self.batch_size, self.vocab_size).to(device)
        
        for timestep in range(self.seq_len):
            input_ = self.drop_outs[-1](embed_out[timestep])
            new_hidden = torch.zeros( self.num_layers,self.batch_size,self.hidden_size).to(device)
            for layer in range(self.num_layers):

                r_out = torch.sigmoid(self.r[layer](torch.cat([input_, hidden[layer]],1)))
                z_out = torch.sigmoid(self.z[layer](torch.cat([input_, hidden[layer]],1)))
                h_out = torch.tanh(self.h[layer](torch.cat([input_, r_out * hidden[layer]],1)))

                new_hidden[layer] = (1 - z_out) * hidden[layer] + z_out * h_out
                input_ = self.drop_outs[layer](new_hidden[layer].clone())
            
            if self.save_hiddens:
                self.hiddens.append(hidden)
            hidden = new_hidden
            logits[timestep] = self.out_layer(input_)

        return logits, hidden

    def generate(self, inputs, hidden, generated_seq_len,temp=1):
        # TODO ========================
        if inputs.is_cuda:
        	device = inputs.get_device()
        else:
            device = torch.device("cpu")

        results = torch.zeros(generated_seq_len, self.batch_size).to(device)

        input_tokens = inputs
        for timestep in range(self.seq_len):
            input_ = self.embeddings(input_tokens)
            hidden_states = []
            for layer in range(self.num_layers):

                r_out = torch.sigmoid(self.r[layer](torch.cat([input_, hidden[layer]],1)))
                z_out = torch.sigmoid(self.z[layer](torch.cat([input_, hidden[layer]],1)))
                h_out = torch.tanh(self.h[layer](torch.cat([input_, r_out * hidden[layer]],1)))

                hidden_states.append( (1 - z_out) * hidden[layer] + z_out * h_out)
                input_ = hidden_states[-1]
            input_tokens = torch.squeeze(torch.multinomial(torch.softmax(self.out_layer(input_)/temp,1),1))
            results[timestep] = input_tokens

            
            hidden = torch.stack(hidden_states)

        return results, hidden


# Problem 3
##############################################################################
#
# Code for the Transformer model
#
##############################################################################

"""
Implement the MultiHeadedAttention module of the transformer architecture.
All other necessary modules have already been implemented for you.

We're building a transfomer architecture for next-step prediction tasks, and
applying it to sequential language modelling. We use a binary "mask" to specify
which time-steps the model can use for the current prediction.
This ensures that the model only attends to previous time-steps.

The model first encodes inputs using the concatenation of a learned WordEmbedding
and a (in our case, hard-coded) PositionalEncoding.
The word embedding maps a word's one-hot encoding into a dense real vector.
The positional encoding 'tags' each element of an input sequence with a code that
identifies it's position (i.e. time-step).

These encodings of the inputs are then transformed repeatedly using multiple
copies of a TransformerBlock.
This block consists of an application of MultiHeadedAttention, followed by a
standard MLP; the MLP applies *the same* mapping at every position.
Both the attention and the MLP are applied with Resnet-style skip connections,
and layer normalization.

The complete model consists of the embeddings, the stacked transformer blocks,
and a linear layer followed by a softmax.
"""

#This code has been modified from an open-source project, by David Krueger.
#The original license is included below:
#MIT License
#
#Copyright (c) 2018 Alexander Rush
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.



#----------------------------------------------------------------------------------

# TODO: implement this class
class MultiHeadedAttention(nn.Module):
    def __init__(self, n_heads, n_units, dropout=0.1):
        """
        n_heads: the number of attention heads
        n_units: the number of input and output units
        dropout: probability of DROPPING units
        """        
        # TODO: create/initialize any necessary parameters or layers
        # Initialize all weights and biases uniformly in the range [-k, k],
        # where k is the square root of 1/n_units.
        # Note: the only Pytorch modules you are allowed to use are nn.Linear 
        # and nn.Dropout
        # ETA: you can also use softmax
        
        super(MultiHeadedAttention, self).__init__()
        
        # storing the number of heads to be used later in the forward
        self.n_heads = n_heads
        # This sets the size of the keys, values, and queries (self.d_k) to all
        # be equal to the number of output units divided by the number of heads.
        self.d_k = n_units // n_heads
        # This requires the number of n_heads to evenly divide n_units.
        assert n_units % n_heads == 0
        # storing the number of units to be used later in the forward
        self.n_units = n_units
        # defining the dropout probability
        self.prob_dropout = dropout
        # creating the layers        
        self.linears = clones(nn.Linear(self.n_units, self.n_units, bias=True), 4)
        # defining the initiation parameter
        k = 1/math.sqrt(self.n_units)
        # initializing weights and biases for each layer
        for i in range(4):
            nn.init.uniform_(self.linears[i].weight,-k, k)
            nn.init.uniform_(self.linears[i].bias,-k, k)
        # defining the dropout
        self.dropout = nn.Dropout(self.prob_dropout)
        
    
    def masked_softmax(self, x, mask):
        """ Implementing a masked version of the softmax function
        """
        # applying the mask on x
        if mask is not None:
            mask = mask.unsqueeze(1)
            x_tilde = x.masked_fill(mask == 0, -1e9) # 0 : element not allowed, 1 : element allowed
        else:
            x_tilde = x 
        # applying softmax on the masked output
        softmax_x =  F.softmax(x_tilde, dim = -1)
        
        return softmax_x
            
        
        
    def forward(self, query, key, value, mask=None):
        # TODO: implement the masked multi-head attention.
        # query, key, and value correspond to Q, K, and V in the latex, and 
        # they all have size: (batch_size, seq_len, self.n_units)
        # mask has size: (batch_size, seq_len, seq_len)
        # As described in the .tex, apply input masking to the softmax 
        # generating the "attention values" (i.e. A_i in the .tex)
        # Also apply dropout to the attention values.
        
        # retrieving the batch size
        batch_size = query.size(0)
        # can test with : seq_len = query.size(1) instead of -1
        # creating the query tensor by applying the first layer after reshaping 
        query = self.linears[0](query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)
        # creating the key tensor by applying the second layer after reshaping
        key = self.linears[1](key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)
        # creating the value tensor by applying the first layer after reshaping 
        value = self.linears[2](value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)
        
        # obtaining the score divided by sqrt(d_k)
        a = torch.matmul(query, key.transpose(-2,-1)) / math.sqrt(self.d_k)
        # applying the mask then the softmax function to obtain the attention values (concatenated)
        a = self.masked_softmax(a, mask)
        # applying the dropout to the attention values
        a = self.dropout(a)
        
        # concatenating the heads then applying the last layer to get the final output
        att_output = self.linears[3](torch.matmul(a,value).contiguous().transpose(1,2).view(batch_size,-1,self.n_units))


        return att_output # size: (batch_size, seq_len, self.n_units)







#----------------------------------------------------------------------------------
# The encodings of elements of the input sequence

class WordEmbedding(nn.Module):
    def __init__(self, n_units, vocab):
        super(WordEmbedding, self).__init__()
        self.lut = nn.Embedding(vocab, n_units)
        self.n_units = n_units

    def forward(self, x):
        #print (x)
        return self.lut(x) * math.sqrt(self.n_units)


class PositionalEncoding(nn.Module):
    def __init__(self, n_units, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, n_units)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, n_units, 2).float() *
                             -(math.log(10000.0) / n_units))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)



#----------------------------------------------------------------------------------
# The TransformerBlock and the full Transformer


class TransformerBlock(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(TransformerBlock, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(ResidualSkipConnectionWithLayerNorm(size, dropout), 2)

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask)) # apply the self-attention
        return self.sublayer[1](x, self.feed_forward) # apply the position-wise MLP


class TransformerStack(nn.Module):
    """
    This will be called on the TransformerBlock (above) to create a stack.
    """
    def __init__(self, layer, n_blocks): # layer will be TransformerBlock (below)
        super(TransformerStack, self).__init__()
        self.layers = clones(layer, n_blocks)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class FullTransformer(nn.Module):
    def __init__(self, transformer_stack, embedding, n_units, vocab_size):
        super(FullTransformer, self).__init__()
        self.transformer_stack = transformer_stack
        self.embedding = embedding
        self.output_layer = nn.Linear(n_units, vocab_size)

    def forward(self, input_sequence, mask):
        embeddings = self.embedding(input_sequence)
        return F.log_softmax(self.output_layer(self.transformer_stack(embeddings, mask)), dim=-1)


def make_model(vocab_size, n_blocks=6,
               n_units=512, n_heads=16, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(n_heads, n_units)
    ff = MLP(n_units, dropout)
    position = PositionalEncoding(n_units, dropout)
    model = FullTransformer(
        transformer_stack=TransformerStack(TransformerBlock(n_units, c(attn), c(ff), dropout), n_blocks),
        embedding=nn.Sequential(WordEmbedding(n_units, vocab_size), c(position)),
        n_units=n_units,
        vocab_size=vocab_size
        )

    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


#----------------------------------------------------------------------------------
# Data processing

def subsequent_mask(size):
    """ helper function for creating the masks. """
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, x, pad=0):
        self.data = x
        self.mask = self.make_mask(self.data, pad)

    @staticmethod
    def make_mask(data, pad):
        "Create a mask to hide future words."
        mask = (data != pad).unsqueeze(-2)
        mask = mask & Variable(
            subsequent_mask(data.size(-1)).type_as(mask.data))
        return mask


#----------------------------------------------------------------------------------
# Some standard modules

class LayerNorm(nn.Module):
    "layer normalization, as in: https://arxiv.org/abs/1607.06450"
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class ResidualSkipConnectionWithLayerNorm(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(ResidualSkipConnectionWithLayerNorm, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class MLP(nn.Module):
    """
    This is just an MLP with 1 hidden layer
    """
    def __init__(self, n_units, dropout=0.1):
        super(MLP, self).__init__()
        self.w_1 = nn.Linear(n_units, 2048)
        self.w_2 = nn.Linear(2048, n_units)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
