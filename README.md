# Name Generation with Autoregressive Character-level Language Modeling

The aim of this research project is to develop an AI-powered system that can generate new and unique names. The system leverages character-level language techniques, ranging from simple n-grams to advanced models like RNNs and Transformers.

## Architecture
![System architecture](https://github.com/ADA-GWU/guidedresearchproject-eljanmahammadli/blob/main/images/architecture.png "System architecture")

## Folder Structure
```
.
├── code_notebooks/         # Folder containing notebooks
├── data/                   # Datasets used in experiments and analysis
│   ├── raw_layer/          # Folder for unprocessed data
│   ├── cleansed_layer/     # Folder for cleaned data
│   ├── app_layer/          # Folder for data prepared for application
│   └── model_layer/        # Folder for data formatted for model training and evaluation
├── papers/                 # Papers that have been analyzed
├── presentations/          # Presentations stored here
├── reports/                # Reports that have been written
└── reviews/                # Given peer reviews
```
## Usage
Command-line interface includes lots of arguments to choose from. Refer to the `src/main.py` for more arguments. Below are two example commands.

In order to train transformer model on dinosaurs.csv with 4 number of head which outputs model to the output folder you can use following command
```
python3 main.py -i dinosaurs.csv -o output -model transformer -n-head 4
```

Or if you want to only sample from the model without stopping or starting training you can use:
```
python3 main.py -i names.txt -o output --inference
```

## Research Progress
Find weekly progress in the **code_notebooks/** folder, which includes Jupyter Notebook files with code and explanations. 
- Week 1 - preprocessing.ipynb
- Week 2 - bigram_count_based_model.ipynb
- Week 3 - trigram_count_based_model.ipynb
- Week 4 - trigram_neural_net_model.ipynb
- Week 5 - MLP_model.ipynb
- Week 6 - WaveNet_model.ipynb
- Week 7 - Vanilla_RNN_and_GRU.ipynb + (model_helpers.py, dataset_utils.py)
- Week 8 - Transformer_GPT2.ipynb


## Results for different models
#### Untrained Model
- **Design:** Basically randomly initialized model.
- **Loss:** 3.784
- **Model Inference:** `xaxljywzqyuuarun, uonnwjekiwouhly, ywxxeklvr, uzpzcqohmccy, gvrvlccdvixprjb`

#### Bigram count-based model
- **Design:** A bigram language model predicts the next word in a sequence based on the probability distribution of pairs of consecutive words in the training data.
- **Loss:** 2.725
- **Model Inference:** `paruis, joa, ftrtx, ts, halloum`

#### Trigram count-based model
- **Design:** A trigram language model predicts the next word in a sequence based on the probability distribution of triplets of consecutive words in the training data.
- **Loss:** 2.496
- **Model Inference:** `tics, nutelamic, prel, tovil, reelesto`

#### Trigram neural-net based model
- **Design:** A trigram neural network-based language model optimizes a 729 by 29 matrix to predict the next word in a sequence, using the probabilities of triplets of consecutive words and leveraging neural network techniques.
- **Loss:** 2.496
- **Model Inference:** `llitekra, merchr, trettravers, alspep, agica`
- **Bottleneck**: the reason why the neural net-based version did not improve loss is that n-gram models themselves are super simple.

#### MLP
- **Design:** The MLP architecture takes a sequence of characters as input, applies character embedding, feeds it through a hidden layer with ReLU activation, and generates logits for each character in the output layer. The architecture learns to capture the relationships between characters in the input sequence and predicts the probabilities of the next character.
- **Spec:** 10-dimensional feature vector, 200 hidden neurons, 3 block size, 11,897 total parameters.
- **Loss:** 2.363
- **Model Inference:** `rid, forcend, welluma, cloudson, rantown`
- **Bottleneck** Scaling up the model does not improve the result because the bottleneck of the architecture is 3 block size.

#### WaveNet
- **Infor:** It is a deep generative model known for its ability to generate high-quality audio waveforms. While WaveNet is primarily designed for audio tasks, I adapted it for character-level language model.
- **Design:** WaveNet for character-level language modeling is a deep neural network architecture featuring dilated convolutions, residual and skip connections, and gated activation functions. It efficiently captures long-range dependencies, handles vanishing gradients, and generates high-quality text by modeling the conditional probabilities of characters in sequences.
- **Spec** - 24-dimensional character embedding space, 128 neurons in each hidden layer, 76579 total parameters
- **Loss**: 2.213
- **Model Inference:** `balloub, socience, opmus, tranvideo, homeline, keyibas, redfir, intellavids, opeq, niflywel, carrage,alphars`

