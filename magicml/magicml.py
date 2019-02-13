import tensorflow as tf

class TextClassification:
    """ Applying of the text classification"""
    def __init__(self, dataset):
        self._dataset = dataset
        self.train_df = load_dataset(os.path.join(os.path.dirname(dataset),"aclImdb", "train"))
        self.test_df = load_dataset(os.path.join(os.path.dirname(dataset),"aclImdb", "test"))
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss