
import argparse
import time
import collections
import sys
import torch
import torch.nn
from torch.autograd import Variable
import torch.nn as nn
import numpy
import matplotlib.pyplot as plt
from glob import glob
np = numpy
import os
sys.path.append(os.getcwd())
# NOTE ==============================================
# This is where your models are imported
from models import RNN, GRU 
from models import make_model as TRANSFORMER


##############################################################################
#
# ARG PARSING AND EXPERIMENT SETUP
#
##############################################################################

parser = argparse.ArgumentParser(description='PyTorch Penn Treebank Language Modeling')

# Arguments you may need to set to run different experiments in 4.1 & 4.2.
parser.add_argument('--model', type=str, default='RNN',
                    help='location of the model weights')

parser.add_argument('--batch_size', type=int, default=None,
                    help='batch size')

parser.add_argument('--debug', action='store_true') 

# DO NOT CHANGE THIS (setting the random seed makes experiments deterministic, 
# which helps for reproducibility)
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')


class ExperimentArgs:
    def __init__(self,base,batch_size = None):
        for file in os.listdir(base):
            if file.endswith("pt"):
                self.weights = os.path.join(base,file)
            elif "exp_config" in file:
                with open(os.path.join(base,file),"r") as f:
                    for line in f:
                        a,b = line.strip().split("    ")
                        if a in ["dp_keep_prob","initial_lr"]:
                            setattr(self,a,float(b))
                        else:
                            setattr(self,a,int(b) if b.isdigit() else b)
        if batch_size is not None:
            self.batch_size = batch_size

args_parse = parser.parse_args()
base_folder = glob(f"4_1/{args_parse.model}*")[0]

args = ExperimentArgs(base_folder,args_parse.batch_size)

args.debug = args_parse.debug

# Set the random seed manually for reproducibility.
torch.manual_seed(args_parse.seed)

# Use the GPU if you have one
if torch.cuda.is_available():
    print("Using the GPU")
    device = torch.device("cuda") 
else:
    print("WARNING: You are about to run on cpu, and this will likely run out \
      of memory. \n You can try setting batch_size=1 to reduce memory usage")
    device = torch.device("cpu")


###############################################################################
#
# DATA LOADING & PROCESSING
#
###############################################################################

# HELPER FUNCTIONS
def _read_words(filename):
    with open(filename, "r") as f:
      return f.read().replace("\n", "<eos>").split()

def _build_vocab(filename):
    data = _read_words(filename)

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))
    id_to_word = dict((v, k) for k, v in word_to_id.items())

    return word_to_id, id_to_word

def _file_to_word_ids(filename, word_to_id):
    data = _read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]

# Processes the raw data from text files
def ptb_raw_data(data_path=None, prefix="ptb"):
    train_path = os.path.join(data_path, prefix + ".train.txt")
    valid_path = os.path.join(data_path, prefix + ".valid.txt")
    test_path = os.path.join(data_path, prefix + ".test.txt")

    word_to_id, id_2_word = _build_vocab(train_path)
    train_data = _file_to_word_ids(train_path, word_to_id)
    valid_data = _file_to_word_ids(valid_path, word_to_id)
    test_data = _file_to_word_ids(test_path, word_to_id)
    return train_data, valid_data, test_data, word_to_id, id_2_word

# Yields minibatches of data
def ptb_iterator(raw_data, batch_size, num_steps):
    raw_data = np.array(raw_data, dtype=np.int32)

    data_len = len(raw_data)
    batch_len = data_len // batch_size
    data = np.zeros([batch_size, batch_len], dtype=np.int32)
    for i in range(batch_size):
        data[i] = raw_data[batch_len * i:batch_len * (i + 1)]

    epoch_size = (batch_len - 1) // num_steps

    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    for i in range(epoch_size):
        x = data[:, i*num_steps:(i+1)*num_steps]
        y = data[:, i*num_steps+1:(i+1)*num_steps+1]
        yield (x, y)


class Batch:
    "Data processing for the transformer. This class adds a mask to the data."
    def __init__(self, x, pad=-1):
        self.data = x
        self.mask = self.make_mask(self.data, pad)
    
    @staticmethod
    def make_mask(data, pad):
        "Create a mask to hide future words."

        def subsequent_mask(size):
            """ helper function for creating the masks. """
            attn_shape = (1, size, size)
            subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
            return torch.from_numpy(subsequent_mask) == 0

        mask = (data != pad).unsqueeze(-2)
        mask = mask & Variable(
            subsequent_mask(data.size(-1)).type_as(mask.data))
        return mask

# LOAD DATA
print('Loading data from '+args.data)
raw_data = ptb_raw_data(data_path=args.data)
train_data, valid_data, test_data, word_to_id, id_2_word = raw_data
vocab_size = len(word_to_id)
print('  vocabulary size: {}'.format(vocab_size))


###############################################################################
# 
# MODEL SETUP
#
###############################################################################

# NOTE ==============================================
# This is where your model code will be called. You may modify this code
# if required for your implementation, but it should not typically be necessary,
# and you must let the TAs know if you do so.
if args.model == 'RNN':
    model = RNN(emb_size=args.emb_size, hidden_size=args.hidden_size, 
                seq_len=args.seq_len, batch_size=args.batch_size,
                vocab_size=vocab_size, num_layers=args.num_layers, 
                dp_keep_prob=args.dp_keep_prob) 
elif args.model == 'GRU':
    model = GRU(emb_size=args.emb_size, hidden_size=args.hidden_size, 
                seq_len=args.seq_len, batch_size=args.batch_size,
                vocab_size=vocab_size, num_layers=args.num_layers, 
                dp_keep_prob=args.dp_keep_prob)
elif args.model == 'TRANSFORMER':
    if args.debug:  # use a very small model
        model = TRANSFORMER(vocab_size=vocab_size, n_units=16, n_blocks=2)
    else:
        # Note that we're using num_layers and hidden_size to mean slightly 
        # different things here than in the RNNs.
        # Also, the Transformer also has other hyperparameters 
        # (such as the number of attention heads) which can change it's behavior.
        model = TRANSFORMER(vocab_size=vocab_size, n_units=args.hidden_size, 
                            n_blocks=args.num_layers, dropout=1.-args.dp_keep_prob) 
    # these 3 attributes don't affect the Transformer's computations; 
    # they are only used in run_epoch
    model.batch_size=args.batch_size
    model.seq_len=args.seq_len
    model.vocab_size=vocab_size
else:
  print("Model type not recognized.")

model = model.to(device)

# LOSS FUNCTION
loss_fn = nn.CrossEntropyLoss()

###############################################################################
# 
# DEFINE COMPUTATIONS FOR PROCESSING ONE EPOCH
#
###############################################################################

def repackage_hidden(h):
    """
    Wraps hidden states in new Tensors, to detach them from their history.
    
    This prevents Pytorch from trying to backpropagate into previous input 
    sequences when we use the final hidden states from one mini-batch as the 
    initial hidden states for the next mini-batch.
    
    Using the final hidden states in this way makes sense when the elements of 
    the mini-batches are actually successive subsequences in a set of longer sequences.
    This is the case with the way we've processed the Penn Treebank dataset.
    """
    if isinstance(h, Variable):
        return h.detach_()
    else:
        return tuple(repackage_hidden(v) for v in h)


def run_epoch(model, data):
    # put the model on inference mode
    model.eval()

    iters = 0
    losses = np.zeros((1,model.seq_len))
    # LOOP THROUGH MINIBATCHES
    for step, (x, y) in enumerate(ptb_iterator(data, model.batch_size, model.seq_len)):
        print(f"Current Step = {step}")
        if args.model == 'TRANSFORMER':
            batch = Batch(torch.from_numpy(x).long().to(device))
            model.zero_grad()
            outputs = model.forward(batch.data, batch.mask).transpose(1,0)
        else:
            # initialize the hidden state at the beginning of each mini batch
            hidden = model.init_hidden()
            hidden = hidden.to(device)
            
            inputs = torch.from_numpy(x.astype(np.int64)).transpose(0, 1).contiguous().to(device)#.cuda()
            model.zero_grad()
            hidden = repackage_hidden(hidden)
            outputs, hidden = model(inputs, hidden)

        targets = torch.from_numpy(y.astype(np.int64)).transpose(0, 1).contiguous().to(device)#.cuda()
        
        # LOSS COMPUTATION
        for t in range(model.seq_len):
            tt = torch.squeeze(targets[t,:].view(-1, model.batch_size))
            loss = loss_fn(outputs[t,:,:].contiguous().view(-1, model.vocab_size), tt)
            losses[0,t] += loss.data.item()
        iters += 1
    return losses/iters

  
# Load weights
if torch.cuda.is_available():
    model.load_state_dict(torch.load(args.weights))
else:
    model.load_state_dict(torch.load(args.weights,map_location='cpu'))

# calculate the loss
val_loss = run_epoch(model, valid_data)

# plot the loss
plt.plot(val_loss.flatten())
plt.title(f"The average loss for each timestep for {args.model}")
plt.xlabel("Timestep")
plt.ylabel("The average loss")

# save figures
plt.savefig(os.path.join(base_folder,'avg_losses.png'))