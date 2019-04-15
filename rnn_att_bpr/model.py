import tensorflow as tf

from tensorflow.python.ops.rnn_cell import GRUCell
from tensorflow.python.ops.rnn_cell import LSTMCell
from tensorflow.python.ops.rnn_cell import MultiRNNCell

class Model(object):

  def __init__(self,user_count, config):

    self.u = tf.placeholder(tf.int32, [None,]) # [B]
    self.i = tf.placeholder(tf.int32, [None,]) # [B]
    self.i_week = tf.placeholder(tf.int32, [None,]) # [B]
    self.i_daygap = tf.placeholder(tf.int32, [None,]) # [B]
    self.j = tf.placeholder(tf.int32, [None,]) # [B]
    self.j_week = tf.placeholder(tf.int32, [None,]) # [B]
    self.j_daygap = tf.placeholder(tf.int32, [None,]) # [B]
    self.y = tf.placeholder(tf.float32, [None,]) # [B]
    self.hist_i = tf.placeholder(tf.int32, [None, None]) # [B, T]
    self.hist_i_week = tf.placeholder(tf.int32, [None, None]) # [B, T]
    self.hist_i_daygap = tf.placeholder(tf.int32, [None, None]) # [B, T]
    self.sl = tf.placeholder(tf.int32, [None,]) # [B]
    self.lr = tf.placeholder(tf.float64, [])

    hidden_units = 128
    self.config = config
    item_count = self.config['embedding_len'][0]
    week_count = self.config['embedding_len'][1]
    daygap_count = self.config['embedding_len'][2]

    user_emb_w = tf.get_variable("user_emb_w", [user_count, hidden_units])
    item_emb_w = tf.get_variable("item_emb_w", [item_count, hidden_units // 2])
    item_b = tf.get_variable("item_b", [item_count],
                             initializer=tf.constant_initializer(0.0))
    week_emb_w = tf.get_variable("week_emb_w", [week_count, hidden_units // 4])
    daygap_emb_w = tf.get_variable("daygap_emb_w", [daygap_count, hidden_units // 4])

    u_emb = tf.nn.embedding_lookup(user_emb_w, self.u)

    i_emb = tf.concat([
        tf.nn.embedding_lookup(item_emb_w, self.i),
        tf.nn.embedding_lookup(week_emb_w, self.i_week),
        tf.nn.embedding_lookup(daygap_emb_w, self.i_daygap),
    ], 1)
    i_b = tf.gather(item_b, self.i)

    # jc = tf.gather(cate_list, self.j)
    j_emb = tf.concat([
        tf.nn.embedding_lookup(item_emb_w, self.j),
        tf.nn.embedding_lookup(week_emb_w, self.j_week),
        tf.nn.embedding_lookup(daygap_emb_w, self.j_daygap),
    ], 1)
    j_b = tf.gather(item_b, self.j)

    # hc = tf.gather(cate_list, self.hist_i)
    h_emb = tf.concat([
        tf.nn.embedding_lookup(item_emb_w, self.hist_i),
        tf.nn.embedding_lookup(week_emb_w, self.hist_i_week),
        tf.nn.embedding_lookup(daygap_emb_w, self.hist_i_daygap),
    ], 2)

    cell_fw = build_cell(hidden_units)
    cell_bw = build_cell(hidden_units)
    rnn_output, _ = tf.nn.bidirectional_dynamic_rnn(
        cell_fw, cell_bw, h_emb, self.sl, dtype=tf.float32)
    rnn_output = tf.concat(rnn_output, 2)

    hist = vanilla_attention(i_emb, rnn_output, self.sl)
    hist = tf.reshape(hist, [-1, hidden_units * 2])
    hist = tf.layers.dense(hist, hidden_units)

    u_emb = hist

    x = i_b - j_b + tf.reduce_sum(tf.multiply(u_emb, (i_emb - j_emb)), 1) # [B]
    self.logits = i_b + tf.reduce_sum(tf.multiply(u_emb, i_emb), 1)
    self.mf_auc = tf.reduce_mean(tf.to_float(x > 0))

    # logits for all item:
    # all_emb = tf.concat([
    #     item_emb_w,
    #     tf.nn.embedding_lookup(cate_emb_w, cate_list)
    #     ], axis=1)
    # self.logits_all = tf.sigmoid(
    #     item_b + tf.matmul(u_emb, all_emb, transpose_b=True))

    # Step variable
    self.global_step = tf.Variable(0, trainable=False, name='global_step')
    self.global_epoch_step = \
        tf.Variable(0, trainable=False, name='global_epoch_step')
    self.global_epoch_step_op = \
        tf.assign(self.global_epoch_step, self.global_epoch_step+1)

    # Loss
    l2_norm = tf.add_n([
        tf.nn.l2_loss(u_emb),
        tf.nn.l2_loss(i_emb),
        ])

    regulation_rate = 0.00005
    self.loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.logits,
            labels=self.y)
        ) + regulation_rate * l2_norm

    trainable_params = tf.trainable_variables()
    self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
    gradients = tf.gradients(self.loss, trainable_params)
    clip_gradients, _ = tf.clip_by_global_norm(gradients, 5)
    self.train_op = self.opt.apply_gradients(
        zip(clip_gradients, trainable_params), global_step=self.global_step)


  def train(self, sess, uij, l):
      loss, _ = sess.run([self.loss, self.train_op], feed_dict={
          self.u: uij[0],
          self.i: uij[1],
          self.i_week: uij[2],
          self.i_daygap: uij[3],
          self.y: uij[4],
          self.hist_i: uij[5],
          self.hist_i_week: uij[6],
          self.hist_i_daygap: uij[7],
          self.sl: uij[8],
          self.lr: l,
      })
      return loss

  def eval(self, sess, uij):
      u_auc = sess.run(self.mf_auc, feed_dict={
          self.u: uij[0],
          self.i: uij[1],
          self.i_week: uij[2],
          self.i_daygap: uij[3],
          self.j: uij[4],
          self.j_week: uij[5],
          self.j_daygap: uij[6],
          self.hist_i: uij[7],
          self.hist_i_week: uij[8],
          self.hist_i_daygap: uij[9],
          self.sl: uij[10],
      })
      return u_auc

  # def test(self, sess, uid, hist_i, sl):
  #   return sess.run(self.logits_all, feed_dict={
  #       self.u: uid,
  #       self.hist_i: hist_i,
  #       self.sl: sl,
  #       })

  def save(self, sess, path):
    pass
    # saver = tf.train.Saver()
    # saver.save(sess, save_path=path)

  def restore(self, sess, path):
    saver = tf.train.Saver()
    saver.restore(sess, save_path=path)

def extract_axis_1(data, ind):
  batch_range = tf.range(tf.shape(data)[0])
  indices = tf.stack([batch_range, ind], axis=1)
  res = tf.gather_nd(data, indices)
  return res

def build_single_cell(hidden_units):
  cell_type = LSTMCell
  # cell_type = GRUCell
  cell = cell_type(hidden_units)
  return cell

def build_cell(hidden_units, depth=1):
  cell_list = [build_single_cell(hidden_units) for i in range(depth)]
  return MultiRNNCell(cell_list)

def vanilla_attention(queries, keys, keys_length):
  '''
    queries:     [B, H]
    keys:        [B, T, H]
    keys_length: [B]
  '''
  queries = tf.tile(queries, [1, 2])
  queries = tf.expand_dims(queries, 1) # [B, 1, H]
  # Multiplication
  outputs = tf.matmul(queries, tf.transpose(keys, [0, 2, 1])) # [B, 1, T]

  # Mask
  key_masks = tf.sequence_mask(keys_length, tf.shape(keys)[1])   # [B, T]
  key_masks = tf.expand_dims(key_masks, 1) # [B, 1, T]
  paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
  outputs = tf.where(key_masks, outputs, paddings)  # [B, 1, T]

  # Scale
  outputs = outputs / (keys.get_shape().as_list()[-1] ** 0.5)

  # Activation
  outputs = tf.nn.softmax(outputs)  # [B, 1, T]

  # Weighted sum
  outputs = tf.matmul(outputs, keys)  # [B, 1, H]

  return outputs
