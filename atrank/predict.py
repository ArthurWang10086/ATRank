import tensorflow as tf
import sys
if __name__ == '__main__':
    filename ='../raw_data/nsh_task_predictset_%s.txt3'%(sys.argv[1])
    outputname = 'predict_%s.txt'%(sys.argv[1])
    modelname ='save_path1'
    items_count = 62
    f_out = open(outputname,'w')
    config = tf.ConfigProto(device_count={"CPU": 1},
                            inter_op_parallelism_threads = 1,
                            intra_op_parallelism_threads = 1,
                            log_device_placement=True)
    with tf.Session(config=config) as sess:
        tf.saved_model.loader.load(sess,[tf.saved_model.tag_constants.SERVING],export_dir=modelname)
        u_t=tf.get_default_graph().get_tensor_by_name('u:0')
        i_t=tf.get_default_graph().get_tensor_by_name('i:0')
        i_week_t=tf.get_default_graph().get_tensor_by_name('i_week:0')
        i_daygap_t=tf.get_default_graph().get_tensor_by_name('i_daygap:0')
        hist_i_t=tf.get_default_graph().get_tensor_by_name('hist_i:0')
        hist_i_week_t=tf.get_default_graph().get_tensor_by_name('hist_i_week:0')
        hist_i_daygap_t=tf.get_default_graph().get_tensor_by_name('hist_i_daygap:0')
        sl_t=tf.get_default_graph().get_tensor_by_name('sl:0')
        is_training_t=tf.get_default_graph().get_tensor_by_name('is_training:0')
        logits_t=tf.get_default_graph().get_tensor_by_name('logits:0')
        datas = open(filename,'r').read().split('\n')
        for data in datas[1:-1] :
            data=data.replace(', ',',')
            u = [data.split(' ')[0]]*items_count
            i = list(range(1,items_count+1))
            week = data.split(' ')[2].split(':')[1]
            i_week = [week]*items_count
            i_daygap = [0]*items_count
            hist = data.split(' ')[1].split(',')
            hist_i = [[x.split(':')[0] for x in hist]]*items_count
            hist_i_week = [[x.split(':')[1] for x in hist]]*items_count
            hist_i_daygap = [[int((int(data.split(' ')[2].split(':')[2]) - int(x.split(':')[2]))/3600.0/24) for x in hist]]*items_count
            sl = [len(hist)]*items_count
        
            res = sess.run(logits_t, feed_dict={
                u_t: u,
                i_t: i,
                i_week_t: i_week,
                i_daygap_t: i_daygap,
                hist_i_t: hist_i,
                hist_i_week_t: hist_i_week,
                hist_i_daygap_t: hist_i_daygap,
                sl_t: sl,
                is_training_t: False,
            })
            import numpy as np
            score = [np.exp(z)/(np.exp(z)+1) for z in res]
            print(str(u[0])+' '+week+' '+','.join(map(str,score)),file=f_out)