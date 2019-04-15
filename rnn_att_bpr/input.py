import numpy as np

class DataInput:
  def __init__(self, data, batch_size):

    self.batch_size = batch_size
    self.data = data
    self.epoch_size = len(self.data) // self.batch_size
    if self.epoch_size * self.batch_size < len(self.data):
      self.epoch_size += 1
    self.i = 0

  def __iter__(self):
    return self

  def __next__(self):

    if self.i == self.epoch_size:
      raise StopIteration

    ts = self.data[self.i * self.batch_size : min((self.i+1) * self.batch_size,
                                                  len(self.data))]
    self.i += 1

    u, i, y, sl = [], [], [], []
    for t in ts:
      u.append(t[0])
      i.append(t[2])
      y.append(t[3])
      sl.append(len(t[1]))
    max_sl = max(sl)

    hist_i = np.zeros([len(ts), max_sl], np.int64)
    hist_i_week = np.zeros([len(ts), max_sl], np.int64)
    hist_i_daygap = np.zeros([len(ts), max_sl], np.int64)

    k = 0
    for t in ts:
      for l in range(len(t[1])):
        data= t[1][l].split(':')
        hist_i[k][l] = data[0]
        hist_i_week[k][l] =   data[1]
        hist_i_daygap[k][l] = int((int(t[2].split(':')[2]) - int(data[2]))/3600.0/24)
        # hist_t[k][l] = t[2][l]
      k += 1

    return self.i, (u, [x.split(':')[0] for x in i], [x.split(':')[1] for x in i], [0 for x in i], y, hist_i, hist_i_week, hist_i_daygap, sl)


class DataInputTest:
  def __init__(self, data, batch_size):

    self.batch_size = batch_size
    self.data = data
    self.epoch_size = len(self.data) // self.batch_size
    if self.epoch_size * self.batch_size < len(self.data):
      self.epoch_size += 1
    self.i = 0

  def __iter__(self):
    return self

  def __next__(self):

    if self.i == self.epoch_size:
      raise StopIteration

    ts = self.data[self.i * self.batch_size : min((self.i+1) * self.batch_size,
                                                  len(self.data))]
    self.i += 1

    u, i, j, sl = [], [], [], []
    for t in ts:
      u.append(t[0])
      i.append(t[2][0])
      j.append(t[2][1])
      sl.append(len(t[1]))
    max_sl = max(sl)

    hist_i = np.zeros([len(ts), max_sl], np.int64)
    hist_i_week = np.zeros([len(ts), max_sl], np.int64)
    hist_i_daygap = np.zeros([len(ts), max_sl], np.int64)

    k = 0
    for t in ts:
      for l in range(len(t[1])):
        data= t[1][l].split(':')
        hist_i[k][l] = data[0]
        hist_i_week[k][l] = data[1]
        hist_i_daygap[k][l] = int((int(t[2][0].split(':')[2]) - int(data[2]))/3600.0/24)
      k += 1

    return self.i, (u, [x.split(':')[0] for x in i], [x.split(':')[1] for x in i], [0 for x in i], [x.split(':')[0] for x in j], [x.split(':')[1] for x in j], [0 for x in j], hist_i, hist_i_week, hist_i_daygap, sl)

