import os
import time
import pickle
import random
import numpy as np
import tensorflow as tf

from input import DataInput, DataInputTest
from model import Model
if __name__ == '__main__':

  config={}
  # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
  random.seed(1234)
  np.random.seed(1234)
  tf.set_random_seed(1234)

  train_batch_size = 32
  test_batch_size = 512

  with open('dataset.pkl', 'rb') as f:
    train_set = pickle.load(f)
    test_set = pickle.load(f)
    # cate_list = pickle.load(f)
    user_count, embedding_len = pickle.load(f)
  config['user_count'] = user_count
  config['embedding_len'] = embedding_len

  best_auc = 0.0

  def _eval(sess, model):
    auc_sum = 0.0
    for _, uij in DataInputTest(test_set, test_batch_size):
      auc_sum += model.eval(sess, uij) * len(uij[0])
    test_auc = auc_sum / len(test_set)
    global best_auc
    if best_auc < test_auc:
      best_auc = test_auc
      # model.save(sess, 'save_path/ckpt')
    return test_auc


  # gpu_options = tf.GPUOptions(allow_growth=True)
  # with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
  with tf.Session() as sess:

    model = Model(user_count, config)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # print('test_auc: %.4f' % _eval(sess, model))

    lr = 1.0
    start_time = time.time()
    for _ in range(50):
      num = 0
      random.shuffle(train_set)

      epoch_size = round(len(train_set) / train_batch_size)
      loss_sum = 0.0
      for _, uij in DataInput(train_set, train_batch_size):
        loss = model.train(sess, uij, lr)
        loss_sum += loss
        num = num+1
        print(num,'loss_avg',loss_sum/num)

        if model.global_step.eval() % 1000 == 0:
          test_auc = _eval(sess, model)
          print('Epoch %d Global_step %d\tTrain_loss: %.4f\tEval_AUC: %.4f' %
                (model.global_epoch_step.eval(), model.global_step.eval(),
                 loss_sum / 1000, test_auc),
                flush=True)


        # if model.global_step.eval() == 50000:
        #   lr = 0.1
        if model.global_step.eval() == 336000:
          lr = 0.1

      print('Epoch %d DONE\tCost time: %.2f' %
            (model.global_epoch_step.eval(), time.time()-start_time),
            flush=True)
      model.global_epoch_step_op.eval()

    print('best test_auc:', best_auc)
