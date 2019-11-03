import tensorflow as tf
import numpy as np
import math
from tensorflow.contrib.layers import fully_connected
class model(object):
    def __init__( self, vocab_size, l2_reg_lambda=0.0, mode=tf.estimator.ModeKeys.TRAIN):
        
        self.sequence_length = tf.flags.FLAGS.max_seq_len
        self.embedding_size = tf.flags.FLAGS.embedding_dim
        self.tokens = tf.placeholder(tf.int32, [None, self.sequence_length], name="tokens")
        self.surf_features = tf.placeholder(tf.float32, [None, tf.flags.FLAGS.surf_features_dim], name="surf_features")
        
        print('---- tokens:', self.tokens)
        # Embedding layer
        #with tf.device('/cpu:0'),  tf.variable_scope("embedding", reuse=tf.AUTO_REUSE) :
        with  tf.variable_scope("embedding", reuse=tf.AUTO_REUSE) :
            # emb_dict, emb_size, emb_mat = load_embed_txt(FLAGS.glove_file, vocab)
            # embed_table is embedding matrix
            self.embed_table = tf.get_variable("emb_mat", shape=[vocab_size, tf.flags.FLAGS.embedding_dim ] )
            print('---- embed table:', self.embed_table)
            self.embedded_words = tf.nn.embedding_lookup(self.embed_table, self.tokens)
            # result shape:  [None, sequence_length, embedding_size]
        
        print('---- embedded_words', self.embedded_words)

        if tf.flags.FLAGS.model == 'bicnn':
            self.semantic_embed = self.bicnn(self.embedded_words)
        if tf.flags.FLAGS.model == 'cnn':
            self.semantic_embed = self.cnn(self.embedded_words)
        elif tf.flags.FLAGS.model == 'lstm':
            self.semantic_embed = self.lstm(self.embedded_words,  mode) 
        elif tf.flags.FLAGS.model == 'gru':
            self.semantic_embed = self.gru(self.embedded_words,  mode) 
        elif tf.flags.FLAGS.model == 'atten_bicnn':
            self.semantic_embed = self.atten_bicnn(self.embedded_words) 
        
        print('---- self.semantic_embed', self.semantic_embed)
        # add surface features
        if tf.flags.FLAGS.use_surface_features:
            self.full_embedding = tf.concat([self.semantic_embed, self.surf_features], axis=1, name='full_embed')
        else: 
            self.full_embedding = self.semantic_embed

        print('---- self.semantic + surf ', self.full_embedding)
        # add context features
        if tf.flags.FLAGS.use_context_features:
            def context_builder(x):
                loss = tf.losses.cosine_distance(x[0], x[1], axis=0)
                return loss
            normalized_embedding = tf.nn.l2_normalize( self.full_embedding, axis = 1)
            self.batchsize = tf.placeholder(tf.int32, [], name="batchsize")


            def embed_context(normalized_embedding, full_embedding):
                for count in range(1, tf.flags.FLAGS.context_size+1):
                    context = tf.concat([normalized_embedding[count:, :], tf.zeros([count, normalized_embedding.shape[1]])], axis=0)
                    mapped = tf.map_fn (context_builder, [context, normalized_embedding], dtype=tf.float32)
                    cosine =  tf.expand_dims(mapped, axis=1)
                    full_embedding = tf.concat([full_embedding, cosine], axis=1)
                for count in range(1, tf.flags.FLAGS.context_size+1):
                    context = tf.concat([tf.zeros([count, normalized_embedding.shape[1]]), normalized_embedding[:-count, :]], axis=0)
                    cosine =  tf.expand_dims(tf.map_fn (context_builder, [context, normalized_embedding], dtype=tf.float32) ,  axis=1)
                    full_embedding = tf.concat([full_embedding, cosine], axis=1)
                return full_embedding

            
            def embed_context_zero(normalized_embedding, full_embedding):
                full_embedding = tf.concat([full_embedding, tf.zeros([ self.batchsize, 2 * tf.flags.FLAGS.context_size] )], axis=1)
                return full_embedding
    
            self.full_embedding = tf.cond(self.batchsize > tf.flags.FLAGS.context_size,
                                        lambda: embed_context(normalized_embedding, self.full_embedding),
                                        lambda: embed_context_zero(normalized_embedding, self.full_embedding))

        self.sent_embed = tf.reshape(self.full_embedding,
                                 [self.batchsize, 2* tf.flags.FLAGS.context_size + self.semantic_embed.shape[-1] + tf.flags.FLAGS.surf_features_dim])
        print('---- embedding with context size:', self.full_embedding)
        # Final (unnormalized) scores and predictions by a Multi Layer Perceptron
        with tf.variable_scope(tf.flags.FLAGS.model + "_output", reuse=tf.AUTO_REUSE):
            n_hidden1 = 100
            n_hidden2 = 50
            n_outputs = 1
            hidden1 = fully_connected(self.sent_embed, n_hidden1, scope="hidden1", activation_fn=tf.nn.sigmoid)
            hidden2 = fully_connected(hidden1, n_hidden2, scope="hidden2", activation_fn=tf.nn.sigmoid)
            output = fully_connected(hidden2, n_outputs, scope="outputs", activation_fn=tf.nn.sigmoid)
            self.scores = tf.squeeze(output)
        if mode != tf.estimator.ModeKeys.PREDICT:
            with tf.variable_scope("loss", reuse=tf.AUTO_REUSE):
                self.input_y = tf.placeholder(tf.float32, [None], name="scores")
                print('---- input_y', self.input_y)
                # Calculate mean cross-entropy loss
                self.loss = tf.losses.mean_squared_error(labels=self.input_y, predictions=self.scores)

    # Create a convolution + maxpool layer for different filter sizes
    def cnn(self, tokens):
        print('---- multi filter cnn')
        pooled_outputs = []
        # add one dimension to be suitable as inputfor conv2d
        self.embedded_expanded_words = tf.expand_dims(self.embedded_words, -1)
        filter_sizes = list(map(int, tf.flags.FLAGS.filter_sizes.split(",")))
        for i, filter_size in enumerate(filter_sizes):
            with tf.variable_scope("conv-maxpool-%s" % filter_size, reuse=tf.AUTO_REUSE):
                # Convolution Layer
                filter_shape = [filter_size, tf.flags.FLAGS.embedding_dim, 1, tf.flags.FLAGS.num_filters]
                W = tf.get_variable("W", shape=filter_shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
                b = tf.get_variable("b", shape=[tf.flags.FLAGS.num_filters], initializer=tf.constant_initializer(0.1))
                conv = tf.nn.conv2d(
                    self.embedded_expanded_words,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, tf.flags.FLAGS.max_seq_len - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = tf.flags.FLAGS. num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total]) 
        print('---- cnn output', self.h_pool_flat)
        return self.h_pool_flat
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def bicnn(self, tokens):
        with tf.variable_scope("bi_cnn", reuse=tf.AUTO_REUSE):
            # make the bigram representation of the embeddings
            embed_head_less = self.embedded_words[:, 1:, :]
            print('---- embed head less: ', embed_head_less)
            embed_prime = tf.concat([embed_head_less, tf.expand_dims(self.embedded_words[:, 0, :], axis=1)], axis=1 )
            print('---- embed prime: ', embed_prime)
            bigram_embed = tf.concat([self.embedded_words , embed_prime], axis=-1, name = 'bigram_embed')
            print('---- bigram shape embed: ', bigram_embed)
            # bigram embed shape:  [None, sequence_length, embedding_size]
            # Convolution Layer
            filter_shape = [2 * self.embedding_size, self.embedding_size]
            W_conv = tf.get_variable("W_conv", shape=filter_shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
            b_conv = tf.get_variable("b_conv", shape=[self.embedding_size], initializer=tf.constant_initializer(0.1))
            print('---- b:', b_conv)
            #  vbi(i,i+1 ) = f(W*b(i,i+1)+b)
            Vbi = tf.map_fn(lambda x: tf.tanh(tf.add(tf.matmul(x, W_conv), b_conv)), bigram_embed, name="conv")
            print('---- conv shape:', Vbi)
            # each element is a bigram vector of embedding_size
            # shape of Vbi = [?, sequence_length, embedding_size ]
            # VSt = max vbi 0<i<|St|
            VSt = tf.map_fn(lambda x: tf.math.reduce_max(x, axis=0), Vbi, name='VSt')
            # shape of VSt = [?, embedding_size] 
            print('---- VSt:', VSt)
            sent_embed = VSt
        return sent_embed

    """
    Attention based bi-cnn
    """
    def atten_bicnn(self, tokens):
        with tf.variable_scope("bi_cnn", reuse=tf.AUTO_REUSE):
            # make the bigram representation of the embeddings
            embed_head_less = self.embedded_words[:, 1:, :]
            print('---- embed head less: ', embed_head_less)
            embed_prime = tf.concat([embed_head_less, tf.expand_dims(self.embedded_words[:, 0, :], axis=1)], axis=1 )
            print('---- embed prime: ', embed_prime)
            bigram_embed = tf.concat([self.embedded_words , embed_prime], axis=-1, name = 'bigram_embed')
            print('---- bigram shape embed: ', bigram_embed)
            # bigram embed shape:  [None, sequence_length, embedding_size]
            # Convolution Layer
            filter_shape = [2 * self.embedding_size, self.embedding_size]
            W_conv = tf.get_variable("W_conv", shape=filter_shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
            b_conv = tf.get_variable("b_conv", shape=[self.embedding_size], initializer=tf.constant_initializer(0.1))
            print('---- b:', b_conv)
            #  vbi(i,i+1 ) = f(W*b(i,i+1)+b)
            Vbi = tf.map_fn(lambda x: tf.tanh(tf.add(tf.matmul(x, W_conv), b_conv)), bigram_embed, name="conv")
            print('---- Vbi conv shape:', Vbi)
            # each element is a bigram vector of embedding_size
            # shape of Vbi = [?, sequence_length, embedding_size ]
            
            # VSt = max vbi 0<i<|St|
            VSt = tf.map_fn(lambda x: tf.math.reduce_max(x, axis=0), Vbi, name='VSt')
            # shape of VSt = [?, embedding_size] 

            # get the attention wieghts
            #W_atten = tf.get_variable("W_atten", shape=[self.sequence_length], initializer=tf.truncated_normal_initializer(stddev=0.1))
            #W_atten = tf.map_fn(lambda x: tf.reduce_sum(tf.multiply(x[0],x[1]), 1),(Vbi,VSt) , name='attention_weight')
            def mult(a):
                a1norm = tf.norm(a[1], ord=2)# scalar norm of the VSt
                print('---- a1norm:', a1norm)
                a0norm = tf.map_fn(lambda x: tf.norm(x, ord=2), a[0], dtype = tf.float32)
                print('---- a0norm:', a0norm)
                out = tf.divide ( tf.divide (tf.reduce_sum(tf.multiply(a[0],a[1]), 1), a0norm), a1norm)
                print('---- out:', out)
                return out
            W_atten = tf.map_fn(mult, [Vbi,VSt] , dtype=tf.float32, name='attention_weight')
            print('---- atten:', W_atten)
            # W_atten shape: [?, sequence_length]
            #VSC = tf.map_fn(lambda x: tf.multiply(), (Vbi, W_atten), dtype=float32,name='atten_mult')
            print('---- VSt:', VSt)
            sent_embed = VSt
        return sent_embed


    """
    A LSTM model for sentence embedding
    """
    def lstm(self, tokens, mode):
        def last_relevant(seq):
            batch_size = tf.shape(seq)[0]
            max_length = int(seq.get_shape()[1])
            input_size = int(seq.get_shape()[2])
            index = tf.range(0, batch_size) * max_length + (max_length - 1)
            print('----index:', index)
            flat = tf.reshape(seq, [-1, input_size])
            print('----flat:', flat)
            return tf.gather(flat, index)

        with tf.variable_scope("rnn", reuse=tf.AUTO_REUSE):
            def make_cell():
                cell = tf.contrib.rnn.BasicLSTMCell(tf.flags.FLAGS.hidden_size, forget_bias=0.0, state_is_tuple=True)# , reuse=not is_training)
                if mode!=tf.estimator.ModeKeys.PREDICT and tf.flags.FLAGS.dropout_keep_prob < 1:
                    cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=tf.flags.FLAGS.dropout_keep_prob)
                return cell
            #cell = tf.contrib.rnn.MultiRNNCell([make_cell() for _ in range(3)], state_is_tuple=True)
            cell = make_cell()
            print('---- tokens shape:', tokens)
            # We discard the following 'state' because every time we look at a new sequence, the state becomes irrelevant.
            all_outputs, state = tf.nn.dynamic_rnn(cell=cell,
                                               inputs=tokens,
                                               dtype=tf.float32)
            print('---- all output', all_outputs)
            sent_embed = last_relevant(all_outputs)
            return sent_embed
    def gru(self, tokens, mode):
        def last_relevant(seq):
            batch_size = tf.shape(seq)[0]
            max_length = int(seq.get_shape()[1])
            input_size = int(seq.get_shape()[2])
            index = tf.range(0, batch_size) * max_length + (max_length - 1)
            print('----index:', index)
            flat = tf.reshape(seq, [-1, input_size])
            print('----flat:', flat)
            return tf.gather(flat, index)

        with tf.variable_scope("rnn", reuse=tf.AUTO_REUSE):
            def make_cell():
                cell = tf.nn.rnn_cell.GRUCell(tf.flags.FLAGS.hidden_size) 
                if mode!=tf.estimator.ModeKeys.PREDICT and tf.flags.FLAGS.dropout_keep_prob < 1:
                    cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=tf.flags.FLAGS.dropout_keep_prob)
                return cell
            #cell = tf.contrib.rnn.MultiRNNCell([make_cell() for _ in range(3)], state_is_tuple=True)
            cell = make_cell()
            print('---- tokens shape:', tokens)
            # We discard the following 'state' because every time we look at a new sequence, the state becomes irrelevant.
            all_outputs, state = tf.nn.dynamic_rnn(cell=cell,
                                               inputs=tokens,
                                               dtype=tf.float32)
            print('---- all output', all_outputs)
            sent_embed = last_relevant(all_outputs)
            return sent_embed
        
