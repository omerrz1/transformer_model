import tensorflow as tf
import keras
import numpy as np




# create a class for data  proccessing
class DataProcessor():
# fix the eos and sos tokens after padding 
    def __init__(self, inputs, eos_sos_token=False, max_len=None):
        self.inputs = inputs
        self.eos = '[end]'
        self.sos ='[start]'
        self.eos_sos_token = eos_sos_token
        self.max_len = max_len
        self.sequences = None

    def gen_tokens(self):
        inputs = self.inputs
        if self.eos_sos_token:
            inputs = [f'{self.sos} {inp} {self.eos}' for inp in inputs]
        inp_tokens = [sentence.split(' ') for sentence in inputs if sentence]
        self.tokens = inp_tokens
        return inp_tokens

    def create_vocab(self):
        words = set()
        for seq in self.tokens:
            for word in seq:
                if word not in words:
                    words.add(word)
        self.vocab = {w:i+1 for i,w in enumerate(words)}
        self.vocab['[PAD]'] = 0
        self.vocab_size = len(self.vocab.items())
        return self.vocab
    
    def gen_sequences(self,):
        self.sequences = [[self.vocab[token] for token in seq] for seq in self.tokens]
        return self.sequences
    
    def pad_seq(self):
        if self.max_len is None:
            self.max_len = max([len(seq)for seq in self.sequences])
        padded_seqs = []
        for seq in self.sequences:
            padded_array = np.zeros(self.max_len)
            padded_array[:len(seq)] = seq[:self.max_len]
            padded_seqs.append(padded_array)
        return np.array(padded_seqs)
    
    def fullyProccess(self):
        self.gen_tokens()
        self.create_vocab()
        self.gen_sequences()
        return self.pad_seq()








# positinal aware word embedding layer 
class Positional_embedding(keras.layers.Layer):
    # attributes = sequence length , embedded dim and vocab size
    def __init__(self,seqeunce_length , embd_dim , vocab_sie, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_sie
        self.embd_dim = embd_dim
        self.sequence_length = seqeunce_length
        
        # defining embedding layer 
        self.token_embedding_layer = keras.layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embd_dim
        )
        # positiona encoding layer 
        self.positional_encoding = keras.layers.Embedding(
            input_dim=self.sequence_length,
            output_dim=self.embd_dim
        )

    #  the call function makaes use of all the layers we dfined above
    def call(self,inputs):
        length = tf.shape(inputs) [-1] # takes the sequence length of the input for example (10 , 60) 60 = seqeunce length 
        positions = tf.range(start=0,limit=length-1,delta=1) # creates a tensor with the same length as the sequence length we extacted above , with delta = 1 (step size = 1) example [1,2,3,..,length-1]
        embedded_tokens = self.token_embedding_layer(inputs) # passing the inputs to an embedding layer to generate embedding representations 
        embedded_positions = self.positional_encoding(positions)  # passing the positions tensor we generated above into the positinonal encoding layer 
        positions_aware_embeddings = embedded_tokens + embedded_positions #adding positional encoding with the embedding representations 
        return positions_aware_embeddings

# transformer encoder layer
class Transformer_encoder(keras.layers.Layer): # we inherit from keras layer class to make it a layer 

    # first we deine all the layers anad then we use them in the call function attributes: embeddedd dim, dense layer dim , number off heads
    def __init__(self,embd_dim,dense_dim,num_heads,**kwargs):
        super().__init__(**kwargs)
        # hyper parameters
        self.embd_dim = embd_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads

        # defining multiheaded self attention 
        self.attention = keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.embd_dim
        )

        # defining the encoder's fully connected layer 
        self.fully_connected = keras.Sequential(
            [keras.layers.Dense(self.dense_dim,activation='relu'),keras.layers.Dense(embd_dim)]
        )
        
        # defining the encoder add_and norm layers (1 and 2)
        self.add_and_norm_1 = keras.layers.LayerNormalization()
        self.add_and_norm_2 = keras.layers.LayerNormalization()

        self.supports_masking =True

    # connecting all the layers together here 
    def call(self,inputs,mask= None):
        if mask is not None:
            padding_mask = tf.cast(mask[:,tf.newaxis,:],dtype='int32')
        
        # passing the input to the attention key, query anad value 
        attention_output = self.attention(query = inputs, key = inputs , value = inputs, attention_mask = padding_mask)
        
        # passing the attention and the input to an add and norm layer 
        fully_connected_input = self.add_and_norm_1(inputs+attention_output) #add and norm '1' ouput 
        
        # passing the add and norm ouput to a fully connected layer  
        fully_connected_output = self.fully_connected(fully_connected_input)

        # passing the fully connected layer output and the add and norm '1' output to a adad and norm and then thats the encoder output 
        encoder_ouput = self.add_and_norm_2(fully_connected_input + fully_connected_output)
        
        return encoder_ouput
    



class Transformer_Decoder(keras.layers.Layer):
    def __init__(self,latent_dim,embd_dim,num_heads,**kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.embd_dim = embd_dim
        self.numm_heads = num_heads

        self.attention_1 = keras.layers.MultiHeadAttention(
            num_heads=self.numm_heads,
            key_dim=embd_dim)
        
        self.attention_2 = keras.layers.MultiHeadAttention(
            num_heads=self.numm_heads,
            key_dim=self.embd_dim
        )

        self.fully_conncted  = keras.Sequential(
            [
                keras.layers.Dense(units=self.latent_dim,activation='relu'),
                keras.layers.Dense(units=self.embd_dim)
            ]
        )

        self.add_and_norm_1 = keras.layers.LayerNormalization()
        self.add_and_norm_2 = keras.layers.LayerNormalization()
        self.add_and_norm_3 = keras.layers.LayerNormalization()

        self.supports_masking = True


        def call(self,inputs,encoder_ouput,mask=None):
            causal_mask = self.get_causal_attention_mask(inputs)
            if mask is not None:
                padding_mask = tf.cast(mask[:,tf.newaxis,:])
                padding_mask = tf.minimum(padding_mask,causal_mask)
            
            attention_1_output = self.attention_1(key=inputs , query=inputs , value=inputs , mask = causal_mask)

            addAndNormOut_1 = self.add_and_norm_1(inputs+attention_1_output)
            
            attention_2_output = self.attention_2(query = addAndNormOut_1, key = encoder_ouput, value = encoder_ouput, mask = padding_mask)

            addAndNormOut_2 = self.add_and_norm_2(addAndNormOut_1,attention_2_output)

            fully_connected_ouput = self.fully_conncted(addAndNormOut_2)

            decoder_ouput = self.add_and_norm_3(addAndNormOut_2+fully_connected_ouput)

            return decoder_ouput
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
        




inputs = ['ehloewbf dbbwe hhw fbbewfj weifne we','ehloewbf dbbwe hhw fbbewfj weifne we','ehloewbf dbbwe hhw fbbewfj weifne webthis sould now be thee longest one out of all of them','ehloewbf dbbwe hhw fbbewfj weifne we',]

dp = DataProcessor(inputs=inputs,eos_sos_token=True,max_len=5)
print(dp.fullyProccess())
print(dp.gen_sequences())
