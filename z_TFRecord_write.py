import os
import tensorflow as tf
import multiprocessing
import numpy as np
stack_num=10
check_path = r'D:\Desktop\DEMO_FILE\UCF101pic';root='ApplyEyeMakeup';name='v_ApplyEyeMakeup_g01_c01'
pic_path = os.path.join(check_path, root, name)  # UCF101pic\ApplyLipstick\v_ApplyLipstick_g25_c04
of_path = os.path.join(check_path + '_of', root, name)  # UCF101pic_of\ApplyLipstick\v_ApplyLipstick_g25_c04
#writer = tf.python_io.TFRecordWriter(tf_name)
order = sorted(os.listdir(pic_path), key=lambda x: int(x.split('.')[0]))  # ['1.jpg','2.jpg',...,'198.jpg']
#for j in order[:-stack_num]:  # '1.jpg'

of_dict_file = os.path.join(check_path + '_of_dict',root+'.txt')
f = open(of_dict_file,'r')
dict_str = f.read()
of_dict = eval('{'+dict_str+'}')
f.close()    

print(of_dict)


j='3.jpg'
pic_file = os.path.join(pic_path, j)  # UCF101pic\ApplyLipstick\v_ApplyLipstick_g25_c04\1.jpg
image_raw = tf.gfile.FastGFile(pic_file, 'rb').read()
idex = os.path.splitext(j)[0]
of_bytes_list = [['' for xy in range(2)] for k in range(stack_num)]
of_min_max_arr = np.empty([stack_num, 2, 2], dtype='float32')
of_stack_eval_str=''
for n in range(stack_num):
    of_idx = int(idex) + n
    of_ext = ['_x', '_y']
    of_pic_path_base = os.path.join(of_path, str(of_idx))
    for xy in range(2):
        of_raw_from_file = of_pic_path_base + of_ext[xy] + '.jpg'
        print(of_raw_from_file)
        of_bytes_list[n][xy] = tf.gfile.FastGFile(of_raw_from_file, 'rb').read()
        of_name = 'of_'+str(n)+of_ext[xy]
        of_astr = 'of_bytes_list[%d][%d]' % (n, xy)
        of_eval_str = """'%s': _bytes_feature(%s),""" % (of_name, of_astr)
        of_stack_eval_str = of_stack_eval_str + of_eval_str
        min_max_ext = ['min','max']
        for min_max in range(2):
        	word = name+'/'+str(of_idx)+of_ext[xy] + '.jpg'
        	of_min_max_arr[n][xy][min_max]=of_dict[word][min_max_ext[min_max]]
        	#print(word,min_max_ext[min_max],of_dict[word][min_max_ext[min_max]])
of_min_max_raw=of_min_max_arr.tostring()
write_tfrecord_base = """tf.train.Example(features=tf.train.Features(
    feature={'root': _bytes_feature(bytes(root, encoding='utf-8')),
             'name': _bytes_feature(bytes(name, encoding='utf-8')),
             'idex': _bytes_feature(bytes(idex, encoding='utf-8')),
             'label': _int64_feature(label),
             'image_raw': _bytes_feature(image_raw),
             'of_min_max_raw': _bytes_feature(of_min_max_raw),
             }))"""
write_tfrecord=write_tfrecord_base[:-3]+of_stack_eval_str+'}))'             
#print(of_bytes_list)
print(of_min_max_arr)
print(write_tfrecord)
# print(of_bytes_list)



#  'image_raw': _bytes_feature(image_raw)
    

    # test = """tf.train.Example(features=tf.train.Features(
    #     feature={'root': _bytes_feature(bytes(root, encoding='utf-8')),
    #              'name': _bytes_feature(bytes(name, encoding='utf-8')),
    #              'idex': _bytes_feature(bytes(idex, encoding='utf-8')),
    #              'label': _int64_feature(label),
    #              'image_raw': _bytes_feature(image_raw)
    #              }))"""

    # # example = tf.train.Example(features=tf.train.Features(
    # #     feature={'root': _bytes_feature(bytes(root, encoding='utf-8')),
    # #              'name': _bytes_feature(bytes(name, encoding='utf-8')),
    # #              'idex': _bytes_feature(bytes(idex, encoding='utf-8')),
    # #              'label': _int64_feature(label)}))

    # example = eval(test)
    
    # writer.write(example.SerializeToString())
    # print('%d,bytes.' % sys.getsizeof(example))
    # print('%d,bytes.' % sys.getsizeof(image_raw))
#writer.close()