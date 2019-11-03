import tensorflow as tf

# Parameters
# ==================================================

# Data loading params
# tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("embed_file", "./glove.6B/glove.6B.50d.txt", "Pretrained word embedding GLOVE.")
tf.flags.DEFINE_string("vocab_file", "./slide_generator_data/vocab.txt", "Vocabulary File.")
tf.flags.DEFINE_string("train_data_file", "./slide_generator_data/train_data.txt", "Train data file.")
tf.flags.DEFINE_string("dev_data_file", "./slide_generator_data/dev_data.txt", "Train data file.")
tf.flags.DEFINE_string("test_data_file", "./slide_generator_data/test_data.txt", "Test datat file.")
tf.flags.DEFINE_string("model", "gru", "model ['lstm', 'gru', 'cnn', 'atten_bicnn']")
tf.flags.DEFINE_string("model_dir", "./"+tf.flags.FLAGS.model+"_runs/checkpoints", "Directory to save the model.")
# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 50, "Dimensionality of character embedding (default: 50)")
if tf.flags.FLAGS.model=='lstm':
    # lower keep probability should improve the performance. But, it does not decrease the training time because of random selection of neurons.  
    tf.flags.DEFINE_float("dropout_keep_prob", 0.3, "Dropout keep probability for lstm (lstm default: 0.3)")
else:
    tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size",100, "Batch Size (default: 100)")
tf.flags.DEFINE_integer("num_epochs", 100, "Number of training epochs (default: 50")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 10, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_integer("max_seq_len", 200, "Maximum sequence length  (default: 1000)")
tf.flags.DEFINE_integer("surf_features_dim",18 , "Number of surface features for each sentence  (default: 8)")
tf.flags.DEFINE_integer("lstm_num_layers", 3, "number of layers in LSTM model (default: 3)")
tf.flags.DEFINE_integer("hidden_size", 50, "Hidden size of the LSTM model (default: 50)")
tf.flags.DEFINE_integer("context_size", 4, "Number of senetences to consider as context (default: 4)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement") # this will allow to run on CPU rather GPU
tf.flags.DEFINE_boolean("log_device_placement", False, "log placement of ops on devices")
tf.flags.DEFINE_boolean("use_surface_features", True, "Allow the model to use surface features. ")
tf.flags.DEFINE_boolean("use_context_features", True, "Allow the model to use context features. ")



# CNN parameters:
tf.flags.DEFINE_string("filter_sizes", "2,3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")


FLAGS = tf.flags.FLAGS
