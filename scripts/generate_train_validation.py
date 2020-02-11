# -*- coding: utf-8 -*-
#########################################################################
# Author: mao
# Created Time: 04,28,Apr,2018
# File Name: generate_train_validation.py
# Description:
#            generate train_set and validation_set
#########################################################################

import os
import shutil
import random
from os import getcwd

#return how many picturs in pics_path
def count(pics_path):
    count_ = 0
    for _, _, files in os.walk(pics_path):
        for file in files:
            count_ = count_+1
    return count_

#train validation genrate and split
def generate_train_validation(pics_path, train_path, validation_path, validation_size=0.2):

    if os.path.exists(train_path):
        os.remove(train_path)
    if os.path.exists(validation_path):
        os.remove(validation_path)

    count_ = count(pics_path)
    all_indexs = range(0, count_)
    test_nums = int(count_ * validation_size)
    validation_indexs = random.sample(all_indexs, test_nums)
    validation_indexs.sort()
    
    train_txt = open(train_path, 'w')
    validation_txt = open(validation_path, 'w')
    
    index = 0
    cur_count = 0
    for _, _, files in os.walk(pics_path):
        for file in files:
            if index<len(validation_indexs) and validation_indexs[index]==cur_count:
                validation_txt.write(pics_path+'/%s\n'%(str(cur_count).zfill(6)+'.jpg'))
                index = index + 1
                print(pics_path+'/%s\n'%(str(cur_count).zfill(6)+'.jpg -> test'))
            else:
                train_txt.write(pics_path+'/%s\n'%(str(cur_count).zfill(6)+'.jpg'))
                print(pics_path+'/%s\n'%(str(cur_count).zfill(6)+'.jpg -> train'))
            cur_count = cur_count+1
    
    train_txt.close()
    validation_txt.close()
    
    print('---------- split train and validation finished ----------')
    print('all pic nums = ' + str(count_))
    print('train pic nums = ' + str(count_-index))
    print('validation pic nums = ' + str(index))
    print('finish')

wd = getcwd()
pics_path = wd + '/JPEGImages'
train_path = wd + '/2018_train.txt'
validation_path = wd + '/2018_validation.txt'
generate_train_validation(pics_path, train_path, validation_path)
target_path = wd + '/ImageSets/Main'
if os.path.exists(target_path+'2018_train.txt'):
    os.remove(target_path+'/2018_train.txt')
if os.path.exists(target_path+'2018_validation.txt'):
    os.remove(target_path+'/2018_validation.txt')
os.system("mv " + train_path + ' ' + target_path+'/2018_train.txt')
os.system("mv " + validation_path + ' ' + target_path+'/2018_validation.txt')
