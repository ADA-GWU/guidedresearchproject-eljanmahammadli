import os
import sys
import argparse
import time
import math
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

import pandas as pd

from dataset_utils import clean_and_train_test_split, CharacterDataset, ContinuousDataLoader
from model_helpers import ModelConfig, display_samples, create_tokens, evaluate


# ----------------------------------------------- Transformer -----------------------------------------------

# This class is used for applying the GELU activation function (https://arxiv.org/abs/1606.08415)
class NewGELU(nn.Module):
    """Applies the Gaussian Error Linear Unit (GELU) function element-wise:
       0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))
       Can be used as a drop-in replacement for ReLU.
    """

    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


# This class implements causal (unidirectional) self-attention mechanism
class CausalSelfAttention(nn.Module):
    """Performs multi-head self-attention, then applies an output linear transformation.
       Causal (unidirectional) self-attention uses masked attention mechanism.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)  # Linear layer to create queries, keys and values
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)  # Output linear transformation
        # We register a lower triangular matrix used for causal masking
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head  # Number of attention heads
        self.n_embd = config.n_embd  # Embedding dimension

    def forward(self, x):
        B, T, C = x.size()  # Batch size, sequence length, feature number

        # Splitting the last dimension of the input tensor into queries, keys and values
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        
        # Reshape and transpose the dimensions for q, k and v
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) 
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) 
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) 

        # Compute the attention scores
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))  # Apply the mask
        att = F.softmax(att, dim=-1)  # Compute the attention probabilities
        y = att @ v  # Compute the weighted sum of values
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # Reshape y

        y = self.c_proj(y)  # Apply the output linear transformation
        return y


# The Transformer block
class Block(nn.Module):
    """An implementation of a Transformer block. It applies, in sequence, self-attention, layer normalization,
       a feed-forward neural network, and another layer normalization.
    """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)  # Layer normalization before self-attention
        self.attn = CausalSelfAttention(config)  # Self-attention mechanism
        self.ln_2 = nn.LayerNorm(config.n_embd)  # Layer normalization before MLP
        # Multi-layer perceptron with GELU activation
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd),
            c_proj  = nn.Linear(4 * config.n_embd, config.n_embd),
            act     = NewGELU(),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.c_proj(m.act(m.c_fc(x)))  # An MLP as a function

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))  # Self-attention with residual connection
        x = x + self.mlpf(self.ln_2(x))  # MLP with residual connection
        return x


# The main Transformer class
class Transformer(nn.Module):
    """The main Transformer model, implementing embedding, positional encoding, Transformer blocks,
       layer normalization and a final linear layer for language modelling.
    """

    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size

        # Transformer parts
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),  # Token embedding
            wpe = nn.Embedding(config.block_size, config.n_embd),  # Positional encoding
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),  # The Transformer blocks
            ln_f = nn.LayerNorm(config.n_embd),  # Final layer normalization
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)  # The final linear layer

        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("number of parameters: %.2fM" % (n_params/1e6,))  # Print out the number of parameters

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)  # Position encoding

        tok_emb = self.transformer.wte(idx)  # Token embedding
        pos_emb = self.transformer.wpe(pos)  # Positional encoding
        x = tok_emb + pos_emb  # Add token and positional embeddings
        for block in self.transformer.h:
            x = block(x)  # Apply each Transformer block
        x = self.transformer.ln_f(x)  # Apply final layer normalization
        logits = self.lm_head(x)  # Compute final output logits
        
        # If targets are provided, compute the loss function (useful for training)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss


# ---------------------------- Recurrent Neural Network (either vanilla or GRU) -----------------------------------

class RNNCell(nn.Module):
    """
    A basic RNN cell.

    This class represents the basic building block of a Recurrent Neural Network (RNN). 
    An RNN cell takes the current input and the previous hidden state to produce the 
    new hidden state. This operation is performed for every element in the input sequence.

    Args:
        config (ModelConfig): The configuration object containing the model parameters.
    """
    def __init__(self, config):
        """
        Initialize the RNN cell with a linear layer.

        The linear layer transforms the concatenated input and hidden state to the 
        new hidden state.

        Args:
            config (ModelConfig): The configuration object containing the model parameters.
        """
        super().__init__()
        self.xh_to_h = nn.Linear(config.n_embd + config.n_embd2, config.n_embd2) # (128, 64)

    def forward(self, xt, hprev):
        """
        Perform the forward pass of the RNN cell.

        The forward pass involves concatenating the input and the previous hidden state, 
        and passing it through the linear layer. The output of the linear layer is 
        passed through a tanh activation function to produce the new hidden state.

        Args:
            xt (torch.Tensor): The input tensor at the current timestep.
            hprev (torch.Tensor): The hidden state at the previous timestep.

        Returns:
            ht (torch.Tensor): The hidden state at the current timestep.
        """
        # xt: input tensor
        # hprev: previous hidden state
        xh = torch.cat([xt, hprev], dim=1) # concat along y-axis
        ht = F.tanh(self.xh_to_h(xh)) # obtain new hidden state
        return ht


class GRUCell(nn.Module):
    """
    Gated Recurrent Unit (GRU) cell.

    The GRU cell is an improved version of the vanilla RNN cell that incorporates gating 
    mechanisms, specifically update and reset gates. These gates allow the GRU cell to 
    adaptively control the flow of information from one time step to the next, thereby 
    mitigating the vanishing gradient problem.

    Args:
        config (ModelConfig): The configuration object containing the model parameters.
    """
    def __init__(self, config):
    """
    Gated Recurrent Unit (GRU) cell.

    The GRU cell is an improved version of the vanilla RNN cell that incorporates gating 
    mechanisms, specifically update and reset gates. These gates allow the GRU cell to 
    adaptively control the flow of information from one time step to the next, thereby 
    mitigating the vanishing gradient problem.

    Args:
        config (ModelConfig): The configuration object containing the model parameters.
    """

    def __init__(self, config):
        """
        Initialize the GRU cell with three linear layers.

        The linear layers are used to compute the values of the update gate, reset gate, 
        and candidate hidden state.

        Args:
            config (ModelConfig): The configuration object containing the model parameters.
        """
        super().__init__()
        # input, forget, output, gate
        self.xh_to_z = nn.Linear(config.n_embd + config.n_embd2, config.n_embd2)
        self.xh_to_r = nn.Linear(config.n_embd + config.n_embd2, config.n_embd2)
        self.xh_to_hbar = nn.Linear(config.n_embd + config.n_embd2, config.n_embd2)

    def forward(self, xt, hprev):
        """
        Perform the forward pass of the GRU cell.

        The forward pass involves computing the update and reset gates, calculating the 
        candidate hidden state based on the reset gate, and then blending the previous 
        hidden state and the candidate hidden state using the update gate to produce 
        the new hidden state.

        Args:
            xt (torch.Tensor): The input tensor at the current timestep.
            hprev (torch.Tensor): The hidden state at the previous timestep.

        Returns:
            ht (torch.Tensor): The hidden state at the current timestep.
        """
        xh = torch.cat([xt, hprev], dim=1)
        r = F.sigmoid(self.xh_to_r(xh)) # reset gate
        hprev_reset = r * hprev
        xhr = torch.cat([xt, hprev_reset], dim=1)
        hbar = F.tanh(self.xh_to_hbar(xhr)) # candidate hidden state
        z = F.sigmoid(self.xh_to_z(xh)) # update gate
        ht = (1 - z) * hprev + z * hbar # new hidden state
        return ht


class RNN(nn.Module):
    """
    Recurrent Neural Network (RNN) for character-based language modeling.

    This class implements a full RNN model, using either a vanilla RNN cell or a Gated 
    Recurrent Unit (GRU) cell. The model transforms input character indices into 
    embeddings, applies the RNN cell in a recurrent manner to update a hidden state 
    for each character in the sequence, and transforms the final hidden states into 
    logits for each character in the vocabulary.

    Args:
        config (ModelConfig): The configuration object containing the model parameters.
        cell_type (str): The type of the RNN cell ('rnn' or 'gru').
    """
    def __init__(self, config, cell_type):
        """
        Initialize the RNN model with an embedding layer, an RNN cell, and a linear layer.

        Args:
            config (ModelConfig): The configuration object containing the model parameters.
            cell_type (str): The type of the RNN cell ('rnn' or 'gru').
        """
        super().__init__()
        # Initialize attributes using the configuration parameters
        self.block_size = config.block_size  # Maximum length of the input sequences
        self.vocab_size = config.vocab_size  # Total number of unique characters in the data

        # Initialize the starting hidden state as a trainable parameter
        self.start = nn.Parameter(torch.zeros(1, config.n_embd2))  

        # Initialize the character embedding layer
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)  

        # Depending on the cell_type, initialize the appropriate type of RNN cell
        if cell_type == 'rnn':
            self.cell = RNNCell(config)
        elif cell_type == 'gru':
            self.cell = GRUCell(config)

        # Initialize the final linear layer to transform the RNN cell's output into logits for each character
        self.lm_head = nn.Linear(config.n_embd2, self.vocab_size)

    def get_block_size(self):
        """
        Get the size of the block (sequence length) for the RNN.

        Returns:
            block_size (int): The size of the block (sequence length).
        """
        # Return the maximum length of the input sequences
        return self.block_size

    def forward(self, idx, targets=None):
        """
        Perform the forward pass of the RNN model.

        The forward pass involves transforming the input character indices into embeddings, 
        applying the RNN cell in a recurrent manner to update a hidden state for each 
        character in the sequence, transforming the final hidden states into logits for 
        each character in the vocabulary, and optionally computing a loss.

        Args:
            idx (torch.Tensor): The input character indices.
            targets (torch.Tensor, optional): The target character indices.

        Returns:
            logits (torch.Tensor): The logits for each character in the vocabulary.
            loss (torch.Tensor, optional): The loss comparing the logits to the targets, 
                if targets are provided.
        """
        # Get the device of the input tensors and the batch size and sequence length
        device = idx.device
        b, t = idx.size()  

        # Embed the input indices for each character
        emb = self.wte(idx)

        # Initialize the hidden state to the starting hidden state
        hprev = self.start.expand((b, -1))  

        # Sequentially apply the RNN cell to each input and update the hidden state
        hiddens = []
        for i in range(t):
            # Get the embedding for the i-th character in each sequence
            xt = emb[:, i, :]  
            # Update the hidden state using the RNN cell
            ht = self.cell(xt, hprev)  
            # Set the previous hidden state for the next iteration
            hprev = ht  
            # Store the hidden state
            hiddens.append(ht)

        # Stack the hidden states into a tensor
        hidden = torch.stack(hiddens, 1)

        # Apply the linear layer to transform the hidden states into logits
        logits = self.lm_head(hidden)

        # If targets are provided, compute the loss
        loss = None
        if targets is not None:
            # Compute the cross-entropy loss between the logits and the targets
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        # Return the logits and the loss
        return logits, loss


# ----------------------------------------------------- MLP -------------------------------------------------
class MLP(nn.Module):
    """
    Bengio et al. 2003 https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf
    """

    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size
        self.vocab_size = config.vocab_size
        self.wte = nn.Embedding(config.vocab_size + 1, config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(self.block_size * config.n_embd, config.n_embd2),
            nn.Tanh(),
            nn.Linear(config.n_embd2, self.vocab_size)
        )

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, targets=None):

        embs = []
        for k in range(self.block_size):
            tok_emb = self.wte(idx)
            idx = torch.roll(idx, 1, 1)
            idx[:, 0] = self.vocab_size
            embs.append(tok_emb)

        x = torch.cat(embs, -1)
        logits = self.mlp(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss


# ----------------------------------------------------- Bigram --------------------------------------------

class Bigram(nn.Module):
    """
    Bigram Language Model 'neural net', simply a lookup table of logits for the
    next character given a previous character.
    """

    def __init__(self, config):
        super().__init__()
        n = config.vocab_size
        self.logits = nn.Parameter(torch.zeros((n, n)))

    def get_block_size(self):
        return 1 # this model only needs one previous character to predict the next

    def forward(self, idx, targets=None):

         # 'forward pass', lol
        logits = self.logits[idx]

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss


if __name__ == '__main__':
    # command-line interface arguments
    parser = argparse.ArgumentParser(description="Character-Level Language Modeling")
    parser.add_argument('--input', '-i', type=str, default='companies.csv', help="input names of any colllection, one per line")
    parser.add_argument('--work-dir', '-o', type=str, default='out', help="output")
    parser.add_argument('--resume', action='store_true', help="resume training from the model exported to the output")
    parser.add_argument('--inference', action='store_true', help="sample from the model without training")
    parser.add_argument('--max-steps', type=int, default=-1, help="max number of training step, -1 for infinite")
    parser.add_argument('--device', type=str, default='cpu', help="device: cpu, cuda, etc")
    parser.add_argument('--top-k', type=int, default=-1, help="-1 means no top-k")
    parser.add_argument('--model', type=str, default='transformer', help="which model to use, bigram, mlp, rnn, gru, transformer")
    parser.add_argument('--n-layer', type=int, default=4, help="number of layers")
    parser.add_argument('--n-head', type=int, default=4, help="number of heads in the transformer class")
    parser.add_argument('--n-embd', type=int, default=64, help="number of feature channelsl")
    parser.add_argument('--n-embd2', type=int, default=64, help="number of feature channels")
    parser.add_argument('--batch-size', '-b', type=int, default=32, help="batch size")
    parser.add_argument('--learning-rate', '-l', type=float, default=5e-4, help="learning rate")
    parser.add_argument('--weight-decay', '-w', type=float, default=0.01, help="weight decay")
    args = parser.parse_args()
    print(vars(args))

    device = args.device
    top_k = args.top_k
    work_dir = args.work_dir
    input_file = args.input_file
    n_layer = args.n_layer
    n_head = args.n_head
    n_embd = args.n_embd
    n_embd2 = args.n_embd2
    model = args.model


    torch.manual_seed(10110609)
    torch.cuda.manual_seed_all(args.seed)
    os.makedirs(args.work_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.work_dir)

    # constructing the dataset
    train_dataset, test_dataset = clean_and_train_test_split()
    vocab_size = train_dataset.get_vocab_size() # 26 letter plus end token (".")
    block_size = train_dataset.get_output_length() # max lenght word + 1
    print(f"{vocab_size=}, {block_size=}")

    # chosing model and its configurations
    config = ModelConfig(vocab_size=vocab_size, block_size=block_size,
                       n_layer=args.n_layer, n_head=args.n_head,
                       n_embd=args.n_embd, n_embd2=args.n_embd2)
    if args.model == 'transformer':
        model = Transformer(config)
    elif args.model == 'bigram':
        model = Bigram(config)
    elif args.model == 'mlp':
        model = MLP(config)
    elif args.model == 'rnn':
        model = RNN(config, cell_type='rnn')
    elif args.model == 'gru':
        model = RNN(config, cell_type='gru')
    elif args.model == 'bow':
        model = BoW(config)
    else:
        raise ValueError(f'model type {args.model} is not recognized')
    model.to(args.device)
    print(f"Number of model parameters: {sum(p.numel() for p in model.parameters())}")
    if args.resume or args.sample_only:
        print("resuming the last model ")
        model.load_state_dict(torch.load(os.path.join(args.work_dir, 'model.pt')))
    if args.sample_only:
        print_samples(num=40)
        sys.exit()

    # optimizer and data loader
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.99), eps=1e-8)
    batch_loader = InfiniteDataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, num_workers=args.num_workers)

    # training the model
    best_loss = None
    step = 0

    while True:

        t0 = time.time()

        # batch loading
        batch = batch_loader.get_next()
        batch = [t.to(device) for t in batch]
        X, Y = batch

        # fitting into model
        logits, loss = model(X, Y)

        # parameter optimization
        model.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # Ensure that all CUDA operations are complete before measuring the time.
        if device.startswith('cuda'):
            torch.cuda.synchronize()
        t1 = time.time()

        # logging and tracking stats
        if step % 10 == 0:
            print(f"step {step} | loss {loss.item():.4f} | step time {(t1-t0)*1000:.2f}ms")

        # evaluate the model
        if step > 0 and step % 500 == 0:
            train_lossi = evaluate(model, train_dataset, device, batch_size=100, max_batches=10)
            test_lossi  = evaluate(model, test_dataset, device, batch_size=100, max_batches=10)
            writer.add_scalar("Loss/train", train_lossi, step)
            writer.add_scalar("Loss/test", test_lossi, step)
            writer.flush()
            print(f"step {step} train loss: {train_lossi} test loss: {test_lossi}")
            # save the model to disk if it has improved
            if best_loss is None or test_lossi < best_loss:
                out_path = os.path.join(work_dir, "model.pt")
                print(f"test loss {test_lossi} is the best so far, saving model to {out_path}")
                torch.save(model.state_dict(), out_path)
                best_loss = test_lossi

        # sample from the model
        if step > 0 and step % 200 == 0:
            display_samples(device, train_dataset, model, quantity=10)ContinuousDataLoader

        step += 1
        # termination conditions
        if max_steps >= 0 and step >= max_steps:
            break