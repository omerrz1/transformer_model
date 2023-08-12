## PyTransformers 

PyTransformers is a powerful library for data processing and implementing Transformer-based models using Keras and TensorFlow. This library simplifies the data preprocessing steps and allows you to build and train Transformer models for various natural language processing tasks.

### Installation

To install the pytransformers library, you can use pip:

pip install pytransformers

### DataProcessor Class

The `DataProcessor` class in pytransformers is designed for data preprocessing and tokenization. It prepares the data for training and evaluation by cleaning the input and target sentences and creating TextVectorization objects for inputs and targets.

#### Constructor

| Parameter           | Description                                                                                                  |
|---------------------|--------------------------------------------------------------------------------------------------------------|
| `inputs`            | List of input sentences.                                                                                    |
| `targets`           | List of target sentences.                                                                                   |
| `maxlen` (optional) | Maximum length of input and target sentences. If not provided, it will be set to the maximum sentence length in the data.  |
| `remove_target_punc`| Boolean value to indicate whether to remove punctuation from the target sentences during data processing. |
| `remove_input_punc` | Boolean value to indicate whether to remove punctuation from the input sentences during data processing.  |


#### Methods

| Method                 | Description                                                                        |
|------------------------|------------------------------------------------------------------------------------|
| `get_Dataset()`        | Returns a preprocessed TensorFlow Dataset ready for training.                      |
| `save_input_vectoriser(name)`  | Saves the input TextVectorization object to a pickle file with the given name.      |
| `save_target_vectoriser(name)` | Saves the target TextVectorization object to a pickle file with the given name.     |
| `load_vectoriser(name)` | Loads a TextVectorization object from a pickle file with the given name.             |

### Transformer Class

The `Transformer` class combines the encoder and decoder layers to create the Transformer model. It takes in sequence length, vocabulary size, latent dimension, embedding dimension, and the number of heads as its parameters.

#### Constructor

| Parameter       | Description                            |
|-----------------|----------------------------------------|
| `seq_length`    | Maximum sequence length for inputs and targets. |
| `vocab_size`    | Vocabulary size (number of unique tokens). |
| `latent_dim`    | Latent dimension for the model.        |
| `embd_dim`      | Embedding dimension for the model.     |
| `num_heads`     | Number of attention heads in the model.|
| `EncoderUnits` | Number of encoder layers in the model.|
| `DecoderUnits`  | Number of deocder layers in the model.|

#### Methods

| Method                    | Description                                                                   |
|---------------------------|-------------------------------------------------------------------------------|
| `model()`                 | Returns the Keras model for the Transformer.                                  |
| `save_transformer(name)`  | Saves the trained Transformer model to an h5 file with the given name.       |
| `answer()`                | Performs prediction for a given input sentence using the trained model.       |
| `Chat()`                  | Allows interactive chat with the trained model for question-answer tasks.     |
| `load_transformer(name)`  | Loads the trained Transformer model from an h5 file with the given name.      |
| `train()`                 | used to fine tune the Transformer model with new data and saves the updated model.     |

### Usage

```python
# Example usage for DataProcessor and Transformer

import pandas as pd
from pytransformer import Transformer, DataProcessor

# Example data
data = pd.DataFrame({
    'text': ['this is the first example', 'and here comes the second example'],
    'code': ['print("hello")', 'print("world")']
})

inputs = data['text'].tolist()
targets = data['code'].tolist()

dp = DataProcessor(inputs=inputs, targets=targets, maxlen=100, remove_input_punc=True, remove_target_punc=True)
dataset = dp.get_Dataset(batch_size=24)

seq_len = dp.maxlen
embd_dim = 512
dense_dim = 8000
vocab_size = dp.vocab_size
encoder_units = 6
decoder_units = 12
num_heads = 16

transformer =  Transformer(vocab_size=vocab_size,embd_dim=embd_dim,seq_length=seq_len,latent_dim=dense_dim,num_heads=num_heads,DecoderUnits=decoder_units,EncoderUnits=encoder_units)
model = transformer.model()
model.summary()
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(dataset, epochs=1)

# Saving the model and vocabulary
transformer.save_transformer(name='transformer_model')
dp.save_input_vectoriser('transformer_inp_vec')
dp.save_target_vectoriser('transformer_tar_vec')

# To use the trained model for prediction, you can load the model and vectorizers and call the Transformer.Chat method.

input_vocab = DataProcessor.load_vectoriser('transformer_inp_vec.pkl')
tar_vocab = DataProcessor.load_vectoriser('transformer_tar_vec.pkl')

model = keras.models.load_model('transformer_model.h5')

# Run prediction with the correct 'max_len' value
max_len = 100

Transformer.Chat(input_vectoriser=input_vocab, target_vectoriser=tar_vocab, model=model, maxlen=max_len)


# fine-tuning the Model

# Load example data
data = pd.DataFrame({
    'text': ['this is the first example', 'and here comes the second example'],
    'code': ['print("hello")', 'print("world")']
})

inputs = data['text'].tolist()
targets = data['code'].tolist()

# Load model and vectorizers
input_vec = DataProcessor.load_vectoriser('transformer_inp_vec.pkl')
target_vec = DataProcessor.load_vectoriser('transformer_tar_vec.pkl')
model = Transformer.load_transformer('transformer_model.h5')

# Train the model with new data
Transformer.train(model=model, input_vectoriser=input_vec, target_vectoriser=target_vec, batch_size=128, epochs=5, inputs=inputs, targets=targets, name='transformer_model')
# Train method will train the model and save it to the local directory

```
# News:
### OBert model coming soon !!
OBERT is a model that closely resembles the BERT architecture but incorporates a few modifications. It is designed to serve the purposes of classification and predicting the next token in a sequence.

## Contributing
If you want to contribute to the pytransformers library, feel free to email me omermustafacontact@gmail.com