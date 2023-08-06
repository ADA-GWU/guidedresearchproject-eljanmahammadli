import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

import pandas as pd


seed = 10110609
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def clean_and_train_test_split():
    df = pd.read_csv(
      "../data/cleansed_layer/companies_usa_size_over_10.csv", usecols=["name"]
    )
    
    # calling "words" instead of names as input data can be any collection of words
    words = df.name.to_list()

    # cleaning the data, removing and leading or ending spaces and deleting empty words
    words = [w.strip() for w in words] 
    words = [w for w in words if w]
    alphabet = sorted(list(set(''.join(words)))) # constructing the alphabets
    max_length = max(len(w) for w in words)
    print(f"word size in the data: {len(words)}")
    print(f"word with the maximum length: {max_length}")
    print(f"number of characters in the alphabet: {len(alphabet)}")
    print("alphabet: ", ''.join(alphabet))

    # train/test split (we'll use the test set to evaluate the model)
    test_set_size = min(1000, int(len(words) * 0.1))
    randp = torch.randperm(len(words)).tolist()
    train = [words[i] for i in randp[:-test_set_size]]
    test = [words[i] for i in randp[-test_set_size:]]
    print(f"train set size: {len(train)}, test set size: {len(test)}")

    train_dataset = CharacterDataset(train, alphabet, max_length)
    test_dataset = CharacterDataset(test, alphabet, max_length)

    return train_dataset, test_dataset


class CharacterDataset(Dataset):

    def __init__(self, words, alphabet, max_word_length):
        self.words = words
        self.alphabet = alphabet
        self.max_word_length = max_word_length
        self.stoi = {ch:i+1 for i,ch in enumerate(alphabet)} # string to index encoding
        self.itos = {i:s for s,i in self.stoi.items()} # index to string decoding

    def __len__(self):
        return len(self.words)

    def contains(self, word):
        return word in self.words

    def get_vocab_size(self):
        return len(self.alphabet) + 1 # all the possible characters and special 0 token

    def get_output_length(self):
        return self.max_word_length + 1 # the longest word + 1 for the SOS token

    def encode(self, word):
        ix = torch.tensor([self.stoi[w] for w in word], dtype=torch.long)
        return ix

    def decode(self, ix):
        word = ''.join(self.itos[i] for i in ix)
        return word

    def __getitem__(self, idx):
        word = self.words[idx]
        ix = self.encode(word)
        x = torch.zeros(self.max_word_length + 1, dtype=torch.long)
        y = torch.zeros(self.max_word_length + 1, dtype=torch.long)
        x[1:1+len(ix)] = ix
        y[:len(ix)] = ix
        y[len(ix)+1:] = -1
        return x, y
    
    
class ContinuousDataLoader:

    def __init__(self, data_source, **loader_args):
        infinite_sampler = torch.utils.data.RandomSampler(data_source, replacement=True, num_samples=int(1e10))
        self.infinite_loader = DataLoader(data_source, sampler=infinite_sampler, **loader_args)
        self.data_iterator = iter(self.infinite_loader)

    def get_next(self):
        try:
            data_batch = next(self.data_iterator)
        except StopIteration:
            self.data_iterator = iter(self.infinite_loader)
            data_batch = next(self.data_iterator)
        return data_batch