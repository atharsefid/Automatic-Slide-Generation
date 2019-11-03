import numpy as np
import re
from rouge import Rouge
from nltk.tree import *
from nltk.parse import CoreNLPParser
from nltk.tokenize import sent_tokenize
import tensorflow as tf
import nltk
import collections
import math
from glob import glob
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

def print_args(flags):
    """Print arguments."""
    print("\nParameters:")
    for attr in flags:
        value = flags[attr].value
        print("{}={}".format(attr, value))
    print("")


def load_embedding(embed_file, vocab):
    emb_dict = dict()
    emb_size = tf.flags.FLAGS.embedding_dim
    with open(embed_file, 'r', encoding='utf8') as f:
        for line in f:
            tokens = line.strip().split(" ")
            word = tokens[0]
            vec = list(map(float, tokens[1:]))
            emb_dict[word] = vec
            if emb_size:
                assert emb_size == len(vec), "All embedding size should be same."
            else:
                emb_size = len(vec)
    oov_counter = 0
    for token in vocab:
        if token not in emb_dict:
            emb_dict[token] = [0.0] * emb_size
            oov_counter +=1
    print('oove:', oov_counter, 'total dic:', len(emb_dict))
    with tf.device('/cpu:0'), tf.name_scope("embedding"):
    #with tf.variable_scope("pretrain_embeddings", dtype=dtype):
        emb_table = np.array([emb_dict[token] for token in vocab], dtype=np.float32)
        emb_table = tf.convert_to_tensor(emb_table, dtype=tf.float32)
        print('---- emb_table:', emb_table)

    return emb_dict, emb_size, emb_table

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()



def load_vocab(vocab_file):
    """load vocab from vocab file.
    Args:
        vocab_file: vocab file path
    Returns:
        vocab_table, vocab, vocab_size
    """

    vocab_table = tf.contrib.lookup.index_table_from_file( # Returns a lookup table that converts a string tensor into int64 IDs.
        vocabulary_file=vocab_file, default_value=0)
    vocab = []
    with open(vocab_file, "rb") as f:
        vocab_size = 0
        for word in f:
            vocab_size += 1
            vocab.append(word.strip())
    return vocab_table, vocab, vocab_size



def load_model(sess, ckpt):
    with sess.as_default(): 
        with sess.graph.as_default(): 
            init_ops = [tf.global_variables_initializer(),
                        tf.local_variables_initializer(), tf.tables_initializer()]
            sess.run(init_ops)
            ckpt_path = tf.train.latest_checkpoint(ckpt)
            print("Loading saved model: " + ckpt_path)
            saver = tf.train.Saver()
            saver.restore(sess, ckpt_path)

# The code for batch iterator:
def _parse_infer_csv(line):
    cols_types = [['']] * 3
    columns = tf.decode_csv(line, record_defaults=cols_types, field_delim='\t')
    return columns

def _parse_infer_test_csv(line):
    cols_types = [['']] * 2
    columns = tf.decode_csv(line, record_defaults=cols_types, field_delim='\t')
    return columns

def _truncate(tensor):
    dim = tf.size(t)
    return tf.cond( tf.greater(dim, k), lambda: tf.slice(t, [0], [k]))

def _split_string(tensor):
    results = np.array(re.split('\[|\]|, |,', tensor.decode("utf-8") ))
    results = [float(result) for result in results if result!='']
    return np.array(results).astype(np.float32)



def get_iterator(data_file, vocab_table, batch_size, max_seq_len, padding=True,):
    """Iterator for train and eval.
    Args:
        data_file: data file, each line contains a sentence that must be ranked
        vocab_table: tf look-up table
        max_seq_len: sentence max length
        padding: Bool
            set True for cnn or attention based model to pad all samples into same length, must set seq_max_len
    Returns:
        (batch, size)
    """
    # interleave is very important to process multiple files at the same time
    dataset = tf.data.TextLineDataset(data_file) # reads the file with each line correspoding to one sample
    dataset = dataset.map(_parse_infer_csv)
    dataset = dataset.map(lambda score, sent, feats: (tf.string_to_number(score, tf.float32), tf.string_split([sent]).values,\
                          tf.py_func(_split_string, inp=[feats], Tout=tf.float32)))
    #                        tf.string_split([feats], delimiter=',] ' ).values)) # you can set num_parallel_calls 
    dataset = dataset.map(lambda score, sent_tokens, feats: (score, tf.cond(tf.greater(tf.size(sent_tokens),tf.flags.FLAGS.max_seq_len), 
                                                                     lambda: tf.slice(sent_tokens, [0], [tf.flags.FLAGS.max_seq_len]), 
                                                                     lambda: sent_tokens), feats)) # truncate to max_seq_length
    # Convert the word strings to ids.  Word strings that are not in the
    # vocab get the lookup table's default_value integer.
    dataset = dataset.map(lambda score, sent_tokens, feats:{'scores':score, 'tokens': tf.cast(vocab_table.lookup(sent_tokens), tf.int32), 'features': feats})
    if padding:
        batch_dataset = dataset.padded_batch(batch_size, padded_shapes={'scores':[],'tokens':[tf.flags.FLAGS.max_seq_len], 'features':[tf.flags.FLAGS.surf_features_dim]},
                                        padding_values=None,
                                        drop_remainder=False)
    else:
        batch_dataset = dataset.padded_batch(batch_size,padded_shapes={'scores':[],'tokens':[tf.flags.FLAGS.max_seq_len], 'features':[tf.flags.FLAGS.surf_features_dim]}, drop_remainder=False)
    batched_iter = batch_dataset.make_initializable_iterator()
    next_batch = batched_iter.get_next()

    return batched_iter, next_batch

'''
def _pad_up_to(tensor):
    constant_values = 'None'
    s = tf.shape(tensor)
    paddings = [[0,tf.flags.FLAGS.max_seq_len - tensor.shape[0]]]  
    return tf.pad(tensor, paddings, 'CONSTANT', constant_values=constant_values)

def get_dev_data(data_file, vocab_table, batch_size, max_seq_len, padding=True,):
    dataset = tf.data.TextLineDataset(data_file) # reads the file with each line correspoding to one sample
    dataset = dataset.map(_parse_infer_csv)
    dataset = dataset.map(lambda score, sent, feats: (tf.string_to_number(score, tf.float32), tf.string_split([sent]).values,\
                          tf.py_func(_split_string, inp=[feats], Tout=tf.float32)))
    dataset = dataset.map(lambda score, sent_tokens, feats: (score, tf.cond(tf.greater(tf.size(sent_tokens),tf.flags.FLAGS.max_seq_len), 
                                                                     lambda: tf.slice(sent_tokens, [0], [tf.flags.FLAGS.max_seq_len]), 
                                                                     lambda: sent_tokens), feats)) # truncate to max_seq_length
    dataset = dataset.map(lambda score, sent_tokens, feats:(score,tf.py_function(_pad_up_to, inp=[sent_tokens], Tout=tf.string),feats))
    dataset = dataset.map(lambda score, sent_tokens, feats:{'scores':score, 'tokens': tf.cast(vocab_table.lookup(sent_tokens), tf.int32), 'features': feats})
    iter = dataset.make_initializable_iterator()
    next_batch = iter.get_next()
    return iter, next_batch
'''


def get_test_iterator(data_file, 
                 vocab_table,
                 batch_size,
                 max_seq_len,
                 padding=True,):

    # interleave is very important to process multiple files at the same time
    dataset = tf.data.TextLineDataset(data_file) # reads the file with each line correspoding to one sample
    dataset = dataset.map(_parse_infer_test_csv)
    dataset = dataset.map(lambda  sent, feats: (tf.string_split([sent]).values,\
                          tf.py_func(_split_string, inp=[feats], Tout=tf.float32)))
    #                        tf.string_split([feats], delimiter=',] ' ).values)) # you can set num_parallel_calls 
    dataset = dataset.map(lambda sent_tokens, feats: ( tf.cond(tf.greater(tf.size(sent_tokens),tf.flags.FLAGS.max_seq_len), 
                                                                     lambda: tf.slice(sent_tokens, [0], [tf.flags.FLAGS.max_seq_len]), 
                                                                     lambda: sent_tokens), feats)) # truncate to max_seq_length
    # Convert the word strings to ids.  Word strings that are not in the
    # vocab get the lookup table's default_value integer.
    dataset = dataset.map(lambda sent_tokens, feats:{'tokens': tf.cast(vocab_table.lookup(sent_tokens), tf.int32), 'features': feats})
    if padding:
        batch_dataset = dataset.padded_batch(batch_size, padded_shapes={'tokens':[tf.flags.FLAGS.max_seq_len], 'features':[tf.flags.FLAGS.surf_features_dim]},
                                        padding_values=None,
                                        drop_remainder=False)
    else:
        batch_dataset = dataset.padded_batch(batch_size,padded_shapes={'tokens':[tf.flags.FLAGS.max_seq_len], 'features':[tf.flags.FLAGS.surf_features_dim]}, drop_remainder=False)
    batched_iter = batch_dataset.make_initializable_iterator()
    next_batch = batched_iter.get_next()

    return batched_iter, next_batch


