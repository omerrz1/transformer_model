import re
import pickle
import numpy as np
import keras
import tensorflow as tf
from keras import layers
from keras.layers import TextVectorization




# create a class for data  proccessing
class DataProcessor():
    def __init__(self,inputs,targets,maxlen = None, remove_target_punc = False ,remove_input_punc =False,  ):
        
        if maxlen is None:
            inpmaxlen = max([len(item) for item in inputs])
            tarmaxlen = max([len(item) for item in targets])
            self.maxlen = max(inpmaxlen,tarmaxlen)
        else:
            self.maxlen = maxlen

        # cleaning data: 
        self.inputs = [self.custom_standardization(text) for text in inputs if text] if remove_input_punc else inputs
        self.targets = [f'[start] {sentence} [end]' for sentence in targets]
        self.targets = [self.custom_standardization(text) for text in self.targets] if remove_target_punc else self.targets
        
        # creating vvectorisers 
        # check if the user wants onlyy one vectoriser 

        self.input_vectoriser = TextVectorization(output_mode='int',output_sequence_length=self.maxlen,standardize='lower')
        self.targets_vectoriser = TextVectorization(output_mode='int',output_sequence_length=self.maxlen+1,standardize='lower')
        
        # passing the taining data to the vectoriser to create tokens 

        self.input_vectoriser.adapt(self.inputs)
        self.targets_vectoriser.adapt(self.targets)

        # getting vocab_size
        self.vocab_size = len(self.targets_vectoriser.get_vocabulary())
        
    def format_dataset(self,inputs, targets):
        inputs = self.input_vectoriser(inputs)
        targets = self.targets_vectoriser(targets)
        return ({"encoder_inputs": inputs, "decoder_inputs": targets[:, :-1],}, targets[:, 1:])


    def get_Dataset(self,batch_size):
        dataset = tf.data.Dataset.from_tensor_slices((self.inputs,self.targets))
        dataset = dataset.batch(batch_size)
        dataset = dataset.map(self.format_dataset)
        dataset = dataset.shuffle(1024).prefetch(16).cache()
        return dataset
    
    def custom_standardization(self,input_string):

        pattern = r"[^\w\[\] ]"
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
    def answer(cls,input_sentence,maxlen,input_vectoriser,target_vectoriser,model):
        
        targ_vocab = target_vectoriser.get_vocabulary()
        targ_index_lookup = dict(zip(range(len(targ_vocab)), targ_vocab))

        tokenized_input_sentence = input_vectoriser([input_sentence])
        decoded_sentence = "[start]"

        for i in range(maxlen):
            tokenized_target_sentence = target_vectoriser([decoded_sentence])[:, :-1]

            predictions = model([tokenized_input_sentence, tokenized_target_sentence])

            sampled_token_index = np.argmax(predictions[0, i, :])
            sampled_token = targ_index_lookup[sampled_token_index]
            decoded_sentence += " " + sampled_token

            if sampled_token == "[end]":
                break

        return decoded_sentence

    @classmethod
    def Chat(cls,maxlen,input_vectoriser,target__vectoriser,model):
        while True:
            user_message = input('------>')
            print('model ==>',cls.answer(user_message,maxlen,input_vectoriser,target__vectoriser,model))

    @classmethod
    def load_transformer(cls,name):
        model = keras.models.load_model(name)
        return model
    @classmethod
    def train(cls,model,input_vectoriser,target_vectoriser,epochs,batch_size,name,inputs,targets):
        
        def format_dataset(inputs, targets):
            inputs = input_vectoriser(inputs)
            targets = target_vectoriser(targets)
            return ({"encoder_inputs": inputs, "decoder_inputs": targets[:, :-1],}, targets[:, 1:])


        def get_Dataset():
            dataset = tf.data.Dataset.from_tensor_slices((inputs,targets))
            dataset = dataset.batch(batch_size)
            dataset = dataset.map(format_dataset)
            dataset = dataset.shuffle(1024).prefetch(16).cache()
            return dataset
        
        model.fit(get_Dataset(),epochs=epochs)
        model.save(f'{name}.h5')
    

   
