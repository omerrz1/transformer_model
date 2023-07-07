import tensorflow as tf
import keras







# positinal aware word embedding layer 
class Positional_embedding(keras.layers.Layer):
    def __init__(self,seqeunce_length , embd_dim , vocab_sie, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_sie
        self.embd_dim = embd_dim
        self.sequence_length = seqeunce_length

        # define embeeding layer and the posetional encoding layer 
        


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
    
