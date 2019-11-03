import tensorflow as tf
from data_helpers import get_test_iterator, load_model, print_args, load_vocab
from config import FLAGS
from model import model
import time 
from glob import glob
import numpy

def predict(test_File):
    
    outfile = test_file[:-len('sent_features.txt')] +tf.flags.FLAGS.model +'_predicted_scores.txt'
    writer = open(outfile, 'w')

    sess.run(iterator.initializer)
    while True:
        try:
            [batch] = sess.run([test_next_batch])
            feed_dict = { 
              mymodel.tokens: batch['tokens'],
              mymodel.surf_features:batch['features'],
              mymodel.batchsize: batch['tokens'].shape[0]
            } 
            [scores] = sess.run([mymodel.scores], feed_dict)
            if type(scores) == numpy.float32:
                writer.write(str(scores)+'\n') 
            else:
                for score in scores:
                    writer.write(str(score)+'\n')
        except tf.errors.OutOfRangeError:
            break
    print("Done. Write output into {}".format(outfile))
    writer.close()

if __name__ == '__main__':
    vocab_table, _, vocab_size = load_vocab(FLAGS.vocab_file)
    mode = tf.estimator.ModeKeys.PREDICT
    mymodel = model(vocab_size, l2_reg_lambda=FLAGS.l2_reg_lambda, mode=mode)
    #FLAGS.batch_size = 1 # for testing batch size must be 
    init_ops = [tf.global_variables_initializer(),                                                                                                                         
                tf.local_variables_initializer(), tf.tables_initializer()] 
    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        sess.run([init_ops])
        for i in range(1000, 1198):
            if i==1163:
                continue
            test_file = glob('slide_generator_data/data/' + str(i) + '/grobid/sent_features.txt')[0]
            iterator, test_next_batch = get_test_iterator(test_file, vocab_table, FLAGS.batch_size, FLAGS.max_seq_len, padding=True)
            predict(test_file) 
