from dataclasses import dataclass

import torch
from torch.nn import functional as F
from torch.utils.data.dataloader import DataLoader


@dataclass
class ModelConfig:
    block_size: int = None # input sequences length
    vocab_size: int = None # (0, vocab_size -1)
    # model parameters for different layers
    n_layer: int = 4
    n_embd: int = 64
    n_embd2: int = 64
    n_head: int = 4


@torch.no_grad()
def create_tokens(model, sequence_indices, max_token_creation, sampling=False, top_k=None):
    """
    Create new tokens from the model, starting from the provided sequence of indices.
    """
    sequence_limit = model.get_block_size()
    for _ in range(max_token_creation):
        # If the sequence context grows too large, it must be trimmed to sequence_limit
        sequence_condition = sequence_indices if sequence_indices.size(1) <= sequence_limit else sequence_indices[:, -sequence_limit:]
        # Pass the model forward to get the logits for the index in the sequence
        logits, _ = model(sequence_condition)
        logits = logits[:, -1, :]
        # Optionally trim the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float('Inf')
        # Apply softmax to convert logits to (normalized) probabilities
        probabilities = F.softmax(logits, dim=-1)
        # Either sample from the distribution or take the most likely element
        if sampling:
            next_index = torch.multinomial(probabilities, num_samples=1)
        else:
            _, next_index = torch.topk(probabilities, k=1, dim=-1)
        # Append sampled index to the ongoing sequence and continue
        sequence_indices = torch.cat((sequence_indices, next_index), dim=1)

    return sequence_indices


def display_samples(device, train_dataset, model, quantity=10):
    """ Function for model inference: sampling some words from the model """
    starting_input = torch.zeros(quantity, 1, dtype=torch.long).to(device)
    generation_steps = train_dataset.get_output_length() - 1 # -1 due to initial <START> token (index 0)
    sampled_input = create_tokens(model, starting_input, generation_steps, top_k=None, sampling=True).to(device)
    training_words, testing_words, novel_words = [], [], []
    for i in range(sampled_input.size(0)):
        # Obtain the i'th row of sampled integers, as python list
        sequence_row = sampled_input[i, 1:].tolist() # Remove the <START> token
        # Token 0 is the <STOP> token, thus we truncate the output sequence at that point
        stop_index = sequence_row.index(0) if 0 in sequence_row else len(sequence_row)
        sequence_row = sequence_row[:stop_index]
        sample_word = train_dataset.decode(sequence_row)
        # Check which words are in the training/testing set and which are new
        if train_dataset.contains(sample_word):
            training_words.append(sample_word)
        elif train_dataset.contains(sample_word):
            testing_words.append(sample_word)
        else:
            novel_words.append(sample_word)
    print('-'*50)
    for word_list, descriptor in [(training_words, 'in training'), (testing_words, 'in testing'), (novel_words, 'new')]:
        print(f"{len(word_list)} samples that are {descriptor}:")
        for word in word_list:
            print(word)
    print('-'*50)


@torch.inference_mode()
def evaluate(model, dataset, device, batch_size=50, max_batches=None):
    model.eval() # evaluation mode
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=0)
    losses = []
    for i, batch in enumerate(loader):
        batch = [t.to(device) for t in batch]
        X, Y = batch
        logits, loss = model(X, Y)
        losses.append(loss.item())
        if max_batches is not None and i >= max_batches:
            break
    mean_loss = torch.tensor(losses).mean().item()
    model.train()
    return mean_loss