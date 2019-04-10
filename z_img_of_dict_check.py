import platform
#import tensorflow as tf
import os
import time
#import datasets_reader as dr
import multiprocessing
import sys

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.001
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 4000
MOVING_AVERAGE_DECAY = 0.99
DEMO = True
KEY = 'UCF101'
OF_STACK_NUMS = 2
split = 1

def check_or_create_path(path_list, create=True):
    for a_path in path_list:
        if not os.path.exists(a_path) and create:
            os.makedirs(a_path)
        elif not os.path.exists(a_path) and not create:
            exit('ZBH: %s does not exist.' % a_path)
def of_dict_reader(of_dict_file):
    f = open(of_dict_file, 'r')
    dict_str = f.read()
    of_dict = eval('{' + dict_str + '}')
    f.close()
    return of_dict

if platform.system() == 'Windows':
    N_GPU = 4
    DATA_FRAMES_PATH = r'D:\Desktop\DEMO_FILE\UCF101pic'
    DATA_OF_PATH = DATA_FRAMES_PATH + '_of'
    DATA_OF_DIC_PATH = DATA_OF_PATH + '_dict'
    ROOT_PATH = os.path.split(DATA_FRAMES_PATH)[0]
    DATA_TF_FILE_PATH = os.path.join(ROOT_PATH, '%s_tfrecords_%sof' % (KEY, OF_STACK_NUMS))
    if KEY == 'UCF101':
        FILE_LIST_PATH = os.path.join(ROOT_PATH, 'ucfTrainTestlist')

class_list=[['1','ApplyEyeMakeup'],['2', 'ApplyLipstick']]

for i in class_list:
    clas=i[1]
    im_sub_path=os.path.join(DATA_FRAMES_PATH,clas)
    of_sub_path=os.path.join(DATA_OF_PATH,clas)
    check_or_create_path([im_sub_path, of_sub_path], create=False)
    sub_path=os.listdir(im_sub_path)
    if len(sub_path) != len(os.listdir(of_sub_path)):
        exit('ZBH: sub_path doed not equal between \n\t%s \t\tand \n\t%s.' % (im_sub_path,of_sub_path))
    of_dict_num=0
    for j in sub_path:
        im_sample_path=os.path.join(im_sub_path,j)
        of_samplt_path=os.path.join(of_sub_path,j)
        im_list=sorted(os.listdir(im_sample_path), key=lambda x: int(x[:-4]))     
        of_list=sorted(os.listdir(of_samplt_path), key=lambda x: int(x[:-6]))
        im_num=len(im_list)
        of_num=len(of_list)
        if 2*im_num != of_num+2 or im_list[0] != '1.jpg' or of_list[0] != '1_x.jpg' or im_list[-1] != ('%d.jpg' % im_num) or of_list[-1] != ('%d_y.jpg' % (im_num-1)):
            exit('ZBH: something wrong at %s.\n\tkey values: %d, %d, %s, %s, %s, %s' % (j,im_num,of_num,im_list[0],of_list[0],im_list[-1],of_list[-1]))
        of_dict_num=of_dict_num+of_num
    of_dict_file=os.path.join(DATA_OF_DIC_PATH,clas+'.txt')
    of_dict=of_dict_reader(of_dict_file)
    if of_dict_num != len(of_dict):        
        # print('ZBH: %s:\tdict_num should be %d but %d in %s' % (clas,of_dict_num,len(of_dict),of_dict_file))
        exit('ZBH: %s dict_num should be %d but %d in %s' % (clas,of_dict_num,len(of_dict),of_dict_file))
