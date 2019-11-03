#! /usr/bin/env python
from rouge import Rouge
import tensorflow as tf
import numpy as np
import os
import time
import datetime
from data_helpers import get_iterator, load_vocab, load_embedding
from model import model
from config import FLAGS
import math

def train(): 

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # load the vocab and embedding files
            vocab_table, vocab, vocab_size = load_vocab(FLAGS.vocab_file)
            embeddings =  load_embedding(FLAGS.embed_file, vocab)
            train_iterator, train_next_batch = get_iterator(FLAGS.train_data_file, vocab_table, FLAGS.batch_size, FLAGS.max_seq_len, padding=True)
            dev_iterator, dev_next_batch = get_iterator(FLAGS.dev_data_file, vocab_table, 10000000, FLAGS.max_seq_len, padding=True)
            
            mode = tf.estimator.ModeKeys.TRAIN
            mymodel = model(vocab_size, l2_reg_lambda=FLAGS.l2_reg_lambda, mode=mode)

            global_step = tf.Variable(0, name="global_step", trainable=False)

            learning_rate = 0.001
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            grads_and_vars = optimizer.compute_gradients(mymodel.loss)
            # clip the gradient norms:
            cliped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in grads_and_vars]
            train_op = optimizer.apply_gradients(cliped_gvs, global_step=global_step)
            
            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            # timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, tf.flags.FLAGS.model +"_runs"))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss
            loss_summary = tf.summary.scalar("loss", mymodel.loss)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
            

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)
            def train_step(): 
                """
                A single training step
                """
                [batch] = sess.run([train_next_batch])
                feed_dict = {
                  mymodel.tokens: batch['tokens'],
                  mymodel.surf_features:batch['features'] ,
                  mymodel.input_y: batch['scores'],
                  mymodel.batchsize: batch['tokens'].shape[0]
                }
                _, step, summaries, loss = sess.run(
                    [train_op, global_step, train_summary_op, mymodel.loss], feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}".format(time_str, step, loss))
                train_summary_writer.add_summary(summaries, step)

            def dev_step(step, writer=None):
                """
                Evaluates model on a dev set
                """
                sess.run(dev_iterator.initializer)
                while True:
                    try:
                        [batch] = sess.run([dev_next_batch])
                        feed_dict = {
                          mymodel.tokens: batch['tokens'],
                          mymodel.surf_features:batch['features'],
                          mymodel.input_y: batch['scores'],
                          mymodel.batchsize: batch['tokens'].shape[0] 
                        }
                        summaries, loss = sess.run(
                            [ dev_summary_op, mymodel.loss], feed_dict)
                        print('--- dev loss: ', loss)
                        if writer:
                            writer.add_summary(summaries, step)
                    except  tf.errors.OutOfRangeError:
                        print("End of dataset")
                        break
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}".format(time_str, step, loss))
                if writer:
                    writer.add_summary(summaries, step)

            # Initialize all variables
            init_ops = [tf.global_variables_initializer(),
                    tf.local_variables_initializer(), tf.tables_initializer()]
            sess.run(init_ops)
            for epoch in range(FLAGS.num_epochs): 
                # initialize going through dataset
                sess.run(train_iterator.initializer)
                while True:
                    try:
                        train_step()
                        current_step = tf.train.global_step(sess, global_step)
                        # evaluate on dev set 
                        if current_step % FLAGS.evaluate_every == 0:
                            print("\nEvaluation:")
                            dev_step(current_step, writer=dev_summary_writer)
                            print("")
                        
                        if current_step % FLAGS.checkpoint_every == 0:
                            path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                            print("Saved model checkpoint to {}\n".format(path))
                    except tf.errors.OutOfRangeError:
                        print("End of dataset")
                        break
                print('-'*100)
def main(argv=None):
    start = time.time()
    train()
    end = time.time()
    print('RUNNING TIME IS: ', end-start)

if __name__ == '__main__':
    tf.app.run()

