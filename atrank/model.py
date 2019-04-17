import os
import json
import numpy as np
import tensorflow as tf

class Model(object):
  def __init__(self, config):
    self.config = config

    # Summary Writer
    self.train_writer = tf.summary.FileWriter(config['model_dir'].value + '/train')
    self.eval_writer = tf.summary.FileWriter(config['model_dir'].value + '/eval')

    # Building network
    self.init_placeholders()
    self.build_model()
    self.init_optimizer()


  def init_placeholders(self):
    # [B] user id
    self.u = tf.placeholder(tf.int32, [None,],name='u')

    # [B] item id
    self.i = tf.placeholder(tf.int32, [None,],name='i')
    self.i_week = tf.placeholder(tf.int32, [None,],name='i_week')
    self.i_daygap = tf.placeholder(tf.int32, [None,],name='i_daygap')

    # [B] item label
    self.y = tf.placeholder(tf.float32, [None,],name='y')

    # [B, T] user's history item id
    self.hist_i = tf.placeholder(tf.int32, [None, None],name='hist_i')
    self.hist_i_week = tf.placeholder(tf.int32, [None, None],name='hist_i_week')
    self.hist_i_daygap = tf.placeholder(tf.int32, [None, None],name='hist_i_daygap')

    # [B, T] user's history item purchase time
    # self.hist_t = tf.placeholder(tf.int32, [None, None])

    # [B] valid length of `hist_i`
    self.sl = tf.placeholder(tf.int32, [None,],name='sl')

    # learning rate
    self.lr = tf.placeholder(tf.float64, [],name='lr')

    # whether it's training or not
    self.is_training = tf.placeholder(tf.bool, [],name='is_training')


  def build_model(self):
    item_count = self.config['embedding_len'][0]
    week_count = self.config['embedding_len'][1]
    daygap_count = self.config['embedding_len'][2]

    item_emb_w = tf.get_variable(
        "item_emb_w",
        [item_count, self.config['itemid_embedding_size'].value])
    item_b = tf.get_variable(
        "item_b",
        [item_count,],
        initializer=tf.constant_initializer(0.0))
    week_emb_w = tf.get_variable(
        "week_emb_w",
        [week_count, self.config['weekid_embedding_size'].value])
    daygap_emb_w = tf.get_variable(
        "daygap_emb_w",
        [daygap_count, self.config['daygapid_embedding_size'].value])
    # cate_list = tf.convert_to_tensor(cate_list, dtype=tf.int64)

    i_emb = tf.concat([
        tf.nn.embedding_lookup(item_emb_w, self.i),
        tf.nn.embedding_lookup(week_emb_w, self.i_week),
        tf.nn.embedding_lookup(daygap_emb_w, self.i_daygap),
        ], 1)
    i_b = tf.gather(item_b, self.i)

    h_emb = tf.concat([
        tf.nn.embedding_lookup(item_emb_w, self.hist_i),
        tf.nn.embedding_lookup(week_emb_w, self.hist_i_week),
        tf.nn.embedding_lookup(daygap_emb_w, self.hist_i_daygap),
        ], 2)

    # if self.config['concat_time_emb'].value == True:
    #   t_emb = tf.one_hot(self.hist_t, 12, dtype=tf.float32)
    #   h_emb = tf.concat([h_emb, t_emb], -1)
    #   h_emb = tf.layers.dense(h_emb, self.config['hidden_units'].value)
    # else:
    #   t_emb = tf.layers.dense(tf.expand_dims(self.hist_t, -1),
    #                           self.config['hidden_units'].value,
    #                           activation=tf.nn.tanh)
    #   h_emb += t_emb


    num_blocks = self.config['num_blocks'].value
    num_heads = self.config['num_heads'].value
    dropout_rate = self.config['dropout'].value
    num_units = h_emb.get_shape().as_list()[-1]

    u_emb, self.att, self.stt = attention_net(
        h_emb,
        self.sl,
        i_emb,
        num_units,
        num_heads,
        num_blocks,
        dropout_rate,
        self.is_training,
        False)

    self.logits = tf.identity(i_b + tf.reduce_sum(tf.multiply(u_emb, i_emb), 1),name='logits')

    # ============== Eval ===============
    self.eval_logits = self.logits
  
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

    self.loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.logits,
            labels=self.y)
        ) + self.config['regulation_rate'].value * l2_norm

    self.train_summary = tf.summary.merge([
        tf.summary.histogram('embedding/1_item_emb', item_emb_w),
        tf.summary.histogram('embedding/2_week_emb', week_emb_w),
        tf.summary.histogram('embedding/3_daygap_emb', daygap_emb_w),
        # tf.summary.histogram('embedding/4_time_raw', self.hist_t),
        # tf.summary.histogram('embedding/3_time_dense', t_emb),
        tf.summary.histogram('embedding/4_final', h_emb),
        tf.summary.histogram('attention_output', u_emb),
        tf.summary.scalar('L2_norm Loss', l2_norm),
        tf.summary.scalar('Training Loss', self.loss),
        ])


  def init_optimizer(self):
    # Gradients and SGD update operation for training the model
    trainable_params = tf.trainable_variables()
    if self.config['optimizer'].value == 'adadelta':
      self.opt = tf.train.AdadeltaOptimizer(learning_rate=self.lr)
    elif self.config['optimizer'].value == 'adam':
      self.opt = tf.train.AdamOptimizer(learning_rate=self.lr)
    elif self.config['optimizer'].value == 'rmsprop':
      self.opt = tf.train.RMSPropOptimizer(learning_rate=self.lr)
    else:
      self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr)

    # Compute gradients of loss w.r.t. all trainable variables
    gradients = tf.gradients(self.loss, trainable_params)

    # Clip gradients by a given maximum_gradient_norm
    clip_gradients, _ = tf.clip_by_global_norm(
        gradients, self.config['max_gradient_norm'].value)

    # Update the model
    self.train_op = self.opt.apply_gradients(
        zip(clip_gradients, trainable_params), global_step=self.global_step)



  def train(self, sess, uij, l, add_summary=True):

    input_feed = {
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
        self.is_training: True,
        }

    output_feed = [self.loss, self.train_op]

    if add_summary:
      output_feed.append(self.train_summary)

    outputs = sess.run(output_feed, input_feed)

    if add_summary:
      self.train_writer.add_summary(
          outputs[2], global_step=self.global_step.eval())
    # print(outputs)

    return outputs[0]

  def eval(self, sess, uij):
    res1 = sess.run(self.eval_logits, feed_dict={
        self.u: uij[0],
        self.i: uij[1],
        self.i_week: uij[2],
        self.i_daygap: uij[3],
        self.hist_i: uij[7],
        self.hist_i_week: uij[8],
        self.hist_i_daygap: uij[9],
        self.sl: uij[10],
        self.is_training: False,
        })
    res2 = sess.run(self.eval_logits, feed_dict={
        self.u: uij[0],
        self.i: uij[4],
        self.i_week: uij[5],
        self.i_daygap: uij[6],
        self.hist_i: uij[7],
        self.hist_i_week: uij[8],
        self.hist_i_daygap: uij[9],
        self.sl: uij[10],
        self.is_training: False,
        })
    return np.mean(res1 - res2 > 0)

  # def test(self, sess, uij):
  #   res1, att_1, stt_1 = sess.run([self.eval_logits, self.att, self.stt], feed_dict={
  #       self.u: uij[0],
  #       self.i: uij[1],
  #       self.hist_i: uij[3],
  #       self.hist_t: uij[4],
  #       self.sl: uij[5],
  #       self.is_training: False,
  #       })
  #   res2, att_2, stt_2 = sess.run([self.eval_logits, self.att, self.stt], feed_dict={
  #       self.u: uij[0],
  #       self.i: uij[2],
  #       self.hist_i: uij[3],
  #       self.hist_t: uij[4],
  #       self.sl: uij[5],
  #       self.is_training: False,
  #       })
  #   return res1, res2, att_1, stt_1, att_2, stt_1


     
  def save(self, sess):
      if os.path.exists(self.config['model_dir'].value):
          os.rmdir(self.config['model_dir'].value)
      with sess.graph.as_default():
          builder = tf.saved_model.builder.SavedModelBuilder(self.config['model_dir'].value)
          # signature_def_map = self._build_signature_def()
          builder.add_meta_graph_and_variables(
              sess, [tf.saved_model.tag_constants.SERVING])
          builder.save()


    # checkpoint_path = os.path.join(self.config['model_dir'].value, 'atrank')
    # saver = tf.train.Saver()
    # save_path = saver.save(
    #     sess, save_path=checkpoint_path, global_step=self.global_step.eval())
    # # json.dump(self.config,
    # #           open('%s-%d.json' % (checkpoint_path, self.global_step.eval()), 'w'),
    # #           indent=2)
    # print('model saved at',save_path, flush=True)

  def restore(self, sess, path):
    saver = tf.train.Saver()
    saver.restore(sess, save_path=path)
    print('model restored from %s' % path, flush=True)


def attention_net(enc, sl, dec, num_units, num_heads, num_blocks, dropout_rate, is_training, reuse):
  with tf.variable_scope("all", reuse=reuse):
    with tf.variable_scope("user_hist_group"):
      for i in range(num_blocks):
        with tf.variable_scope("num_blocks_{}".format(i)):
          ### Multihead Attention
          enc, stt_vec = multihead_attention(queries=enc,
              queries_length=sl,
              keys=enc,
              keys_length=sl,
              num_units=num_units,
              num_heads=num_heads,
              dropout_rate=dropout_rate,
              is_training=is_training,
              scope="self_attention"
              )

          ### Feed Forward
          enc = feedforward(enc,
              num_units=[num_units // 4, num_units],
              scope="feed_forward", reuse=reuse)

    dec = tf.expand_dims(dec, 1)
    with tf.variable_scope("item_feature_group"):
      for i in range(num_blocks):
        with tf.variable_scope("num_blocks_{}".format(i)):
          ## Multihead Attention ( vanilla attention)
          dec, att_vec = multihead_attention(queries=dec,
              queries_length=tf.ones_like(dec[:, 0, 0], dtype=tf.int32),
              keys=enc,
              keys_length=sl,
              num_units=num_units,
              num_heads=num_heads,
              dropout_rate=dropout_rate,
              is_training=is_training,
              scope="vanilla_attention")

          ## Feed Forward
          dec = feedforward(dec,
              num_units=[num_units // 4, num_units],
              scope="feed_forward", reuse=reuse)

    dec = tf.reshape(dec, [-1, num_units])
    return dec, att_vec, stt_vec


def multihead_attention(queries,
            queries_length,
            keys,
            keys_length,
            num_units=None,
            num_heads=8,
            dropout_rate=0,
            is_training=True,
            scope="multihead_attention",
            reuse=None):
  '''Applies multihead attention.

  Args:
    queries: A 3d tensor with shape of [N, T_q, C_q].
    queries_length: A 1d tensor with shape of [N].
    keys: A 3d tensor with shape of [N, T_k, C_k].
    keys_length:  A 1d tensor with shape of [N].
    num_units: A scalar. Attention size.
    dropout_rate: A floating point number.
    is_training: Boolean. Controller of mechanism for dropout.
    num_heads: An int. Number of heads.
    scope: Optional scope for `variable_scope`.
    reuse: Boolean, whether to reuse the weights of a previous layer
    by the same name.

  Returns
    A 3d tensor with shape of (N, T_q, C)
  '''
  with tf.variable_scope(scope, reuse=reuse):
    # Set the fall back option for num_units
    if num_units is None:
      num_units = queries.get_shape().as_list[-1]

    # Linear projections, C = # dim or column, T_x = # vectors or actions
    Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu)  # (N, T_q, C)
    K = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # (N, T_k, C)
    V = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # (N, T_k, C)

    # Split and concat
    Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
    K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
    V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

    # Multiplication
    # query-key score matrix
    # each big score matrix is then split into h score matrix with same size
    # w.r.t. different part of the feature
    outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)

    # Scale
    outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

    # Key Masking
    key_masks = tf.sequence_mask(keys_length, tf.shape(keys)[1])   # (N, T_k)
    key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
    key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)

    paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
    outputs = tf.where(key_masks, outputs, paddings)  # (h*N, T_q, T_k)

    # Causality = Future blinding: No use, removed

    # Activation
    outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

    # Query Masking
    query_masks = tf.sequence_mask(queries_length, tf.shape(queries)[1], dtype=tf.float32)   # (N, T_q)
    query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
    query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
    outputs *= query_masks  # broadcasting. (h*N, T_q, T_k)

    # Attention vector
    att_vec = outputs

    # Dropouts
    outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

    # Weighted sum
    outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)

    # Restore shape
    outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

    # Residual connection
    outputs += queries

    # Normalize
    outputs = normalize(outputs)  # (N, T_q, C)

  return outputs, att_vec

def feedforward(inputs,
        num_units=[2048, 512],
        scope="feedforward",
        reuse=None):
  '''Point-wise feed forward net.

  Args:
    inputs: A 3d tensor with shape of [N, T, C].
    num_units: A list of two integers.
    scope: Optional scope for `variable_scope`.
    reuse: Boolean, whether to reuse the weights of a previous layer
    by the same name.

  Returns:
    A 3d tensor with the same shape and dtype as inputs
  '''
  with tf.variable_scope(scope, reuse=reuse):
    # Inner layer
    params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
          "activation": tf.nn.relu, "use_bias": True}
    outputs = tf.layers.conv1d(**params)

    # Readout layer
    params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
          "activation": None, "use_bias": True}
    outputs = tf.layers.conv1d(**params)

    # Residual connection
    outputs += inputs

    # Normalize
    outputs = normalize(outputs)

  return outputs

def normalize(inputs,
        epsilon=1e-8,
        scope="ln",
        reuse=None):
  '''Applies layer normalization.

  Args:
    inputs: A tensor with 2 or more dimensions, where the first dimension has
    `batch_size`.
    epsilon: A floating number. A very small number for preventing ZeroDivision Error.
    scope: Optional scope for `variable_scope`.
    reuse: Boolean, whether to reuse the weights of a previous layer
    by the same name.

  Returns:
    A tensor with the same shape and data dtype as `inputs`.
  '''
  with tf.variable_scope(scope, reuse=reuse):
    inputs_shape = inputs.get_shape()
    params_shape = inputs_shape[-1:]

    mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
    beta = tf.Variable(tf.zeros(params_shape))
    gamma = tf.Variable(tf.ones(params_shape))
    normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
    outputs = gamma * normalized + beta

  return outputs


def extract_axis_1(data, ind):
  batch_range = tf.range(tf.shape(data)[0])
  indices = tf.stack([batch_range, ind], axis=1)
  res = tf.gather_nd(data, indices)
  return res

