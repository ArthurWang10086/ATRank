import random
import pickle
import numpy as np
import pandas as pd

if __name__ == '__main__':

    random.seed(1234)
    filename='../raw_data/nsh_trainset_2019-05-16.txt'
    reviews_df = open(filename,'r').read().split('\n')[1:]
    # reviews_df = pd.read_csv(filename,sep=' ')
    user_count=len(set([x.split(' ')[0] for x in reviews_df]))
    cate_count=item_count=66
    example_count=len(reviews_df)
    print(user_count,item_count,example_count)


    train_set = []
    test_set = []
    for i in range(len(reviews_df)):
        data=reviews_df[i].replace(', ',',').split(' ')
        if int(data[0])%11>3:
            train_set.append((data[0], data[1].split(','), data[2], data[3]))
        else:
            if int(data[3])>0 and int(data[0])%11<=3:
                pos = data[2].split(':')
                neg = str(random.randint(0, item_count-1)+1)
                test_set.append((data[0], data[1].split(','), (data[2],':'.join([neg,pos[1],pos[2]]))))

    # # meta_df = pd.read_csv('../raw_data/meta.txt')
    # # meta_df = meta_df[['asin', 'categories']]
    # # cate_list = meta_df
    # # [1, 2) = 0, [2, 4) = 1, [4, 8) = 2, [8, 16) = 3...  need len(gap) hot
    # gap = np.array([2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096])
    # # gap = [2, 7, 15, 30, 60,]
    # # gap.extend( range(90, 4000, 200) )
    # # gap = np.array(gap)
    # print(gap.shape[0])
    #
    # def proc_time_emb(hist_t, cur_t):
    #   hist_t = [cur_t - i + 1 for i in hist_t]
    #   hist_t = [np.sum(i >= gap) for i in hist_t]
    #   return hist_t
    #
    # train_set = []
    # test_set = []
    # for reviewerID, hist in reviews_df.groupby('reviewerID'):
    #   pos_list = hist['asin'].tolist()
    #   tim_list = hist['unixReviewTime'].tolist()
    #   tim_list = [i // 3600 // 24 for i in tim_list]
    #   def gen_neg():
    #     neg = pos_list[0]
    #     while neg in pos_list:
    #       neg = random.randint(0, item_count-1)
    #     return neg
    #   neg_list = [gen_neg() for i in range(len(pos_list))]
    #
    #   for i in range(1, len(pos_list)):
    #     hist_i = pos_list[:i]
    #     hist_t = proc_time_emb(tim_list[:i], tim_list[i])
    #     if i != len(pos_list) - 1:
    #       train_set.append((reviewerID, hist_i, hist_t, pos_list[i], 1))
    #       train_set.append((reviewerID, hist_i, hist_t, neg_list[i], 0))
    #     else:
    #       label = (pos_list[i], neg_list[i])
    #       test_set.append((reviewerID, hist_i, hist_t, label))

    random.shuffle(train_set)
    random.shuffle(test_set)

    # assert len(test_set) == user_count
    # assert(len(test_set) + len(train_set) // 2 == reviews_df.shape[0])
    embedding_item_len = 67 #1-62
    embedding_week_len = 7  #0-6
    embedding_daygap_len = 15  #1-14


    with open('dataset.pkl', 'wb') as f:
      pickle.dump(train_set, f, pickle.HIGHEST_PROTOCOL)
      pickle.dump(test_set, f, pickle.HIGHEST_PROTOCOL)
      # pickle.dump(cate_list, f, pickle.HIGHEST_PROTOCOL)
      pickle.dump((user_count, [embedding_item_len, embedding_week_len,embedding_daygap_len]), f, pickle.HIGHEST_PROTOCOL)
