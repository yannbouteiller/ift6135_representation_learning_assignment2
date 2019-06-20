import argparse
import time
import collections
import sys
import torch
import torch.nn
from torch.autograd import Variable
import torch.nn as nn
import numpy
np = numpy
import os

import tqdm
import matplotlib.pyplot as plt
from glob import glob
from models_plots import RNN, GRU 
from models_plots import make_model as TRANSFORMER


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
                dp_keep_prob=args.dp_keep_prob,save_hiddens=True) 
elif args.model == 'GRU':
    model = GRU(emb_size=args.emb_size, hidden_size=args.hidden_size, 
                seq_len=args.seq_len, batch_size=args.batch_size,
                vocab_size=vocab_size, num_layers=args.num_layers, 
                dp_keep_prob=args.dp_keep_prob,save_hiddens=True)
elif args.model == 'TRANSFORMER':
    raise NotImplementedError("Not required!")
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


def run_epoch(model, data, is_train=False, lr=1.0):
    """
    One epoch of training/validation (depending on flag is_train).
    """
    if is_train:
        model.train()
    else:
        model.eval()
    epoch_size = ((len(data) // model.batch_size) - 1) // model.seq_len
    start_time = time.time()
    if args.model != 'TRANSFORMER':
        hidden = model.init_hidden()
        hidden = hidden.to(device)
    costs = 0.0
    iters = 0
    losses = []

    # LOOP THROUGH MINIBATCHES
    pbar = tqdm.tqdm(enumerate(ptb_iterator(data, model.batch_size, model.seq_len)),total=epoch_size)
    for step, (x, y) in pbar:
        if args.debug:
        	print(step,f"Input shape{x.shape}")
      # for step, (x, y) in enumerate(ptb_iterator(data, model.batch_size, model.seq_len)):
        if args.model == 'TRANSFORMER':
            batch = Batch(torch.from_numpy(x).long().to(device))
            model.zero_grad()
            foutputs = model.forward(batch.data, batch.mask).transpose(1,0)
            #print ("outputs.shape", outputs.shape)
        else:
            
            inputs = torch.from_numpy(x.astype(np.int64)).transpose(0, 1).contiguous().to(device)#.cuda()
            model.zero_grad()
            # print("this is before repackage hidden: ", hidden)
            # print("this is before repackage hidden: ", hidden.shape)
            # hidden = repackage_hidden(hidden)
            # print("this is repackage hidden: ", hidden)
            # print("this is repackage hidden: ", hidden.shape)
            foutputs, hidden = model(inputs, hidden)
		# targets = torch.from_numpy(y.astype(np.int64)).contiguous().to(device) #.cuda()
        
        targets = torch.from_numpy(y.astype(np.int64)).transpose(0, 1).contiguous().to(device)#.cuda()
        tt = targets[-1]
        print("tt: ", tt.shape)
        # tt = torch.squeeze(targets.view(-1, model.batch_size * model.seq_len))
        
        outputs = foutputs[-1]
        print("outputs shape: ", outputs.shape)
        # LOSS COMPUTATION
        # This line currently averages across all the sequences in a mini-batch 
        # and all time-steps of the sequences.
        # For problem 5.3, you will (instead) need to compute the average loss 
        #at each time-step separately.
        #print("output.shape : ", outputs.shape)
        
        grads = {}
        def save_grad(name):
            def hook(grad):
                grads[name] = grad
            return hook
        loss = loss_fn(outputs, tt) 
        # In here, save_grad('y') returns a hook (a function) that keeps 'y' as name
        for i in range(len(model.hiddens)):
            model.hiddens[i].register_hook(save_grad(f"hidden{i}"))

        loss.register_hook(save_grad("new_loss"))
        loss.backward()
        print("test: ", grads["new_loss"])
        norms = []
        for i in range(len(model.hiddens)):
            grad = grads[f"hidden{i}"].permute(1,0,2).contiguous().view(model.batch_size,-1)
            grad = torch.norm(grad, p=2, dim=1).mean()
            norms.append(grad.item())
        if args.debug:
            print(step, loss)
        break

    return norms

model.load_state_dict(torch.load(args.weights, map_location=device))

norms = np.array(run_epoch(model, train_data))
print("the array of norms: ", norms)

#norms_valid = np.array(run_epoch(model, valid_data))
#print("the array of norms for valid: ", norms_valid)


norms = (norms - norms.min()) / (norms.max() - norms.min())

plt.plot(range(1,args.seq_len+1),norms)
plt.savefig(os.path.join(base_folder,"gradient_plot.jpg"))
