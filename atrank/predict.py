import tensorflow as tf
if __name__ == '__main__':
    filename =''
    modelname =''
    datas = open(filename,'r').read().split('\n')
    items_count = 62

    with tf.Session() as sess:
        tf.saved_model.loader.load(sess,[tf.saved_model.tag_constants.SERVING],export_dir=modelname)
        u_t=tf.get_default_graph().get_tensor_by_name('u')
        i_t=tf.get_default_graph().get_tensor_by_name('i')
        i_week_t=tf.get_default_graph().get_tensor_by_name('i_week')
        i_daygap_t=tf.get_default_graph().get_tensor_by_name('i_daygap')
        hist_i_t=tf.get_default_graph().get_tensor_by_name('hist_i')
        hist_i_week_t=tf.get_default_graph().get_tensor_by_name('hist_i_week')
        hist_i_daygap_t=tf.get_default_graph().get_tensor_by_name('hist_i_daygap')
        sl_t=tf.get_default_graph().get_tensor_by_name('sl')
        is_training_t=tf.get_default_graph().get_tensor_by_name('is_training')
        logits_t=tf.get_default_graph().get_tensor_by_name('logits')

        for data in datas :
            u = [data.split(' ')[0]]*items_count
            i = list(range(1,items_count+1))
            i_week = [data.split(' ')[2].split(':')[1]]*items_count
            i_daygap = [data.split(' ')[2].split(':')[2]]*items_count
            hist = data.split(' ')[1].split(',')
            hist_i = [[x.split(':')[0] for x in hist]]*items_count
            hist_i_week = [[x.split(':')[1] for x in hist]]*items_count
            hist_i_daygap = [[x.split(':')[1] for x in hist]]*items_count
            sl = len(hist)
        
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
            print(u,res)