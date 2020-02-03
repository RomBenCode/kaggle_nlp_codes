import tensorflow as tf


class F1Score(tf.keras.metrics.Metric):

    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.p = tf.keras.metrics.Precision()
        self.r = tf.keras.metrics.Recall()

    def update_state(self, *args, **kwargs):
        self.p.update_state(*args, **kwargs)
        self.r.update_state(*args, **kwargs)

    def reset_states(self):
        self.p.reset_states()
        self.r.reset_states()

    def result(self):
        p_res, r_res = self.p.result(), self.r.result()
        return (2 * p_res * r_res) / (p_res + r_res)