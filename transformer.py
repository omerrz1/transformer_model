import re
import pickle
import tensorflow as tf
import keras
import numpy as np
from keras import layers
from keras.layers import TextVectorization
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer



# create a class for data  proccessing
class DataProcessor():
    def __init__(self,inputs,targets,maxlen = None, remove_input_punc= False, remove_target_punc = False ):
        
        if maxlen is None:
            inpmaxlen = max([len(item) for item in inputs])
            tarmaxlen = max([len(item) for item in targets])
            self.maxlen = max(inpmaxlen,tarmaxlen)

        else:
            self.maxlen = maxlen

        # cleaning data: 
        self.inputs = self.custom_standardization(inputs) if remove_input_punc else inputs
        self.targets = self.custom_standardization(targets) if remove_target_punc else targets
        
        # creating vvectorisers 
        self.input_vectoriser = TextVectorization(output_mode='int',output_sequence_length=self.maxlen)
        self.targets_vectoriser = TextVectorization(output_mode='int',output_sequence_length=self.maxlen)
        
        # padding the taining data to the vectoriser to create tokens 
        self.input_vectoriser.adapt(self.inputs)
        self.targets_vectoriser.adapt(self.targets)

    def format_dataset(self,inputs, targets):

        inputs = self.input_vectoriser(inputs)
        targets = self.targets_vectoriser(targets)

        return ({"encoder_inputs": inputs, "decoder_inputs": targets[:, :-1],}, targets[:, 1:])


    def get_Dataset(self,batch_size):
        dataset = tf.data.Dataset.from_tensor_slices((self.inputs,self.targets))
        dataset = dataset.batch(batch_size)
        dataset = dataset.map(self.format_dataset)
        dataset = dataset.shuffle(1024).prefetch(8).cache()
        return dataset
    
    def custom_standardization(self,input_string):
        # Define a regular expression pattern to match only special characters
        pattern = r"[^\w\[\] ]"
        # Use regex.sub to remove all special characters except [ ], and space
        return re.sub(pattern, "", input_string.lower())

    def save_input_vectoriser(self,name):
        config = {
            'config' : self.input_vectoriser.get_config(),
            'weights': self.input_vectoriser.get_weights()
        }
        pickle.dump(config,open(f'{name}.pkl','wb'))
    
    def save_target_vectoriser(self,name):
        config = {
            'config':self.targets_vectoriser.get_config(),
            'weights': self.targets_vectoriser.get_weights()
        }

        pickle.dump(config,open(f'{name}.pkl','wb'))
    

    @classmethod
    def load_vectoriser(cls,name):
        # loading config
        config = pickle.load(open(f'{name}','rb'))

        # setting  intial config
        vectoriser = TextVectorization().from_config(config['config'])
        # setting initial weights
        vectoriser.set_weights(config['weights'])
        
        return vectoriser






# positinal aware word embedding layer 
@keras.saving.register_keras_serializable()
class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim
        )
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=embed_dim
        )
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)

    def get_config(self):
        config = super().get_config()
        config.update({
            'sequence_length': self.sequence_length,
            'vocab_size': self.vocab_size,
            'embed_dim':self.embed_dim
        })

        return config
    

# transformer encoder layer
@keras.saving.register_keras_serializable()
class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.fully_connected = keras.Sequential(
            [layers.Dense(dense_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        # masking is used to handle variable length inputs 
        self.supports_masking = True

    def call(self, inputs, mask=None):
        padding_mask = None 
        if mask is not None:
            padding_mask = tf.cast(mask[:, tf.newaxis, :], dtype="int32")
        attention_output = self.attention(
            query=inputs, value=inputs, key=inputs, attention_mask=padding_mask
        )
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.fully_connected(proj_input)
        return self.layernorm_2(proj_input + proj_output)
    
    # for saving the layer when saving the model 
    def get_config(self):
          config = super().get_config()
          config.update(
              {
                'dense_dim': self.dense_dim,
                'embed_dim':self.embed_dim,
                'num_heads':self.num_heads

              }
                )
          return config



@keras.saving.register_keras_serializable()
class TransformerDecoder(layers.Layer):
    def __init__(self, embed_dim, latent_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.attention_1 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.attention_2 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.dense_proj = keras.Sequential(
            [layers.Dense(latent_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, encoder_outputs, mask=None):
        causal_mask = self.get_causal_attention_mask(inputs)
        if mask is not None:
            padding_mask = tf.cast(mask[:, tf.newaxis, :], dtype="int32")
            padding_mask = tf.minimum(padding_mask, causal_mask)

        attention_output_1 = self.attention_1(
            query=inputs, value=inputs, key=inputs, attention_mask=causal_mask
        )
        out_1 = self.layernorm_1(inputs + attention_output_1)

        attention_output_2 = self.attention_2(
            query=out_1,
            value=encoder_outputs,
            key=encoder_outputs,
            attention_mask=padding_mask,
        )
        out_2 = self.layernorm_2(out_1 + attention_output_2)

        proj_output = self.dense_proj(out_2)
        return self.layernorm_3(out_2 + proj_output)

    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)],
            axis=0,
        )
        return tf.tile(mask, mult)

    def get_config(self):
        config = super().get_config()
        config.update({
            'embed_dim':self.embed_dim,
            'latent_dim': self.latent_dim,
            'num_heads':self.num_heads
        })
        return config

class Transformer():
    def __init__(self,seq_length,vocab_size,latent_dim,embd_dim,num_heads):

        encoder_inputs = keras.Input(shape=(None,), dtype="int64", name="encoder_inputs")
        x = PositionalEmbedding(seq_length, vocab_size, embd_dim)(encoder_inputs)
        encoder_outputs = TransformerEncoder(embd_dim, latent_dim, num_heads)(x)
        encoder = keras.Model(encoder_inputs, encoder_outputs)

        decoder_inputs = keras.Input(shape=(None,), dtype="int64", name="decoder_inputs")
        encoded_seq_inputs = keras.Input(shape=(None, embd_dim), name="decoder_state_inputs")
        x = PositionalEmbedding(seq_length, vocab_size, embd_dim)(decoder_inputs)
        x = TransformerDecoder(embd_dim, latent_dim, num_heads)(x, encoded_seq_inputs)
        x = layers.Dropout(0.5)(x)
        decoder_outputs = layers.Dense(vocab_size, activation="softmax")(x)
        decoder = keras.Model([decoder_inputs, encoded_seq_inputs], decoder_outputs)

        decoder_outputs = decoder([decoder_inputs, encoder_outputs])
        
        self.Transormer_model = keras.Model(
            [encoder_inputs, decoder_inputs], decoder_outputs, name="transformer"
        )

    def model(self):
        return self.Transormer_model

    
    def save_transformer(self,name):
        self.model().save(f'{name}.h5')


# methods 
    @classmethod
    def answer(cls,inpu_seq,model,inp_vocab,tar_vocab,max_len):
        vectorised_input = inpu_seq.split(' ')
        print('split inp:',vectorised_input)
        vectorised_input = [inp_vocab.get(word,0) for word in vectorised_input if word]
        print(vectorised_input)
        vectorised_input = pad_sequences([vectorised_input],maxlen=max_len,padding='post')
        print(vectorised_input)
        reversed_tar_vocab = {i:w for w,i in tar_vocab.items()}
        decoded_sentence = '[start]'
        
        for i in range(max_len):
            vectorised_target_sentence = decoded_sentence.split(' ')
            vectorised_target_sentence = [tar_vocab.get(word,0) for word in vectorised_target_sentence if word]
            print('decoded',vectorised_target_sentence)
            vectorised_target_sentence = pad_sequences([vectorised_target_sentence],max_len,padding='post')

            predictions = model([vectorised_input,vectorised_target_sentence])
            print('predictions: ', predictions)
            predicted_token = np.argmax(predictions[0, i, :])
            print(predicted_token)
            word = reversed_tar_vocab.get(predicted_token, 'UNK')
            decoded_sentence += " "+ word

            if word =='[end]':
                break
        return(decoded_sentence)

    @classmethod
    def Chat(cls,model,inp_vocab,tar_vocab,max_len):
        while True:
            user_message = input('------>')
            print('model==>',cls.answer(user_message,model,inp_vocab,tar_vocab,max_len))

    @classmethod
    def load_transformer(cls,name):
        model = keras.models.load_model(name)
        return model
    
    

   
