import sys
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import PIL as Image
import numpy as np
import math
import os
import copy
import shutil
from lxml.etree import Element, SubElement, tostring, ElementTree

#SRC_WIDTH = 1224
#SRC_HIGH = 370
DST_WIDTH = 1216
DST_HIGH = 352

#Source folder in which images will be resized
src_image_dir = '/home/sunny/soft_proj/ai/traffic/object_detection/dataset/kitti/data_object_image_2/training/image_2/'
#Destine folder in which images resized
dst_image_dir = '/home/sunny/soft_proj/ai/traffic/object_detection/dataset/kitti/data_object_image_2/training/image_2_resize/'

#Annotation txt files folder 
txt_file_dir = '/home/sunny/soft_proj/ai/traffic/object_detection/dataset/kitti/label_training/label_2/'
#Annotation xml files folder
xml_file_dir = '/home/sunny/soft_proj/ai/traffic/object_detection/dataset/kitti/label_training/label_2_xml/'
#Template xml files path
template_file = '/home/sunny/soft_proj/ai/traffic/object_detection/dataset/kitti/label_training/demo.xml'

#The folder of source image which will be add a frame
raw_image_dir = '/home/sunny/soft_proj/ai/traffic/object_detection/dataset/kitti/data_object_image_2/training/image_2_resize/'
#The folder of target image copy from source image
tgt_image_dir = '/home/sunny/soft_proj/ai/traffic/object_detection/dataset/kitti/data_object_image_2/training/image_2_frame/'

#Resize picture,1224*370 -> 1216*352
def resize_pic(file_name):
    image_array = cv2.imread(src_image_dir + file_name)
    #print(str(np.shape(image_array)[0]) + ' ' + str(np.shape(image_array)[1]))
    #if(np.shape(image_array)[0] != 370 or np.shape(image_array)[1] != 1224):
    #    print('This image is not a kitti image')
    #    return
    image_resize = cv2.resize(image_array,(DST_WIDTH,DST_HIGH),None,0,0,cv2.INTER_LINEAR)
    cv2.imwrite(dst_image_dir + file_name,image_resize,[int(cv2.IMWRITE_JPEG_QUALITY),100])

#Images in kitti dataset do not have the same size
def resize_annot(xmin,ymin,xmax,ymax,src_width,src_high):
    new_xmin = math.floor(float(xmin)*DST_WIDTH/src_width)
    new_ymin = math.floor(float(ymin)*DST_HIGH/src_high)
    new_xmax = math.ceil(float(xmax)*DST_WIDTH/src_width)
    new_ymax = math.ceil(float(ymax)*DST_HIGH/src_high)
    return new_xmin,new_ymin,new_xmax,new_ymax

#Change annotation format from txt to xml
def txt2xml(txt_file_name,xml_file_name):
    #Get raw image size
    png_file_name = txt_file_name.split('.')[0] + '.png'
    image_array = cv2.imread(src_image_dir + png_file_name)
    image_high = int(image_array.shape[0])
    image_width = int(image_array.shape[1])
    tree = ElementTree()
    full_txt_file_name = txt_file_dir + txt_file_name
    full_xml_file_name = xml_file_dir + xml_file_name
    with open(full_txt_file_name) as f:
        txt_file_lines = f.readlines()

    tree.parse(template_file)
    root = tree.getroot()
    root.find('filename').text = png_file_name
    #size
    sz = root.find('size')
    sz.find('height').text = '352'
    sz.find('width').text = '1216'
    sz.find('depth').text = '3'
    #object
    #default object number is 1
    one_obj = 1
    for line in txt_file_lines:
        line_data = line.split()
        if(line_data[0][0] == '#'):
            continue
        if(line_data[0] == 'DontCare'):
            continue

        label = line_data[0]
        orig_xmin = line_data[4]
        orig_ymin = line_data[5]
        orig_xmax = line_data[6]
        orig_ymax = line_data[7]
        xmin,ymin,xmax,ymax = resize_annot(orig_xmin,orig_ymin,orig_xmax,orig_ymax,image_width,image_high)

        if(one_obj == 1):
            one_obj = 0
            obj = root.find('object')
            obj.find('name').text = label
            bb = obj.find('bndbox')
            bb.find('xmin').text = str(xmin)
            bb.find('ymin').text = str(ymin)
            bb.find('xmax').text = str(xmax)
            bb.find('ymax').text = str(ymax)
        else:
            obj_more = copy.deepcopy(obj)
            obj_more.find('name').text = label
            bb = obj_more.find('bndbox')
            bb.find('xmin').text = str(xmin)
            bb.find('ymin').text = str(ymin)
            bb.find('xmax').text = str(xmax)
            bb.find('ymax').text = str(ymax)
            root.append(obj_more)
    tree.write(full_xml_file_name,encoding='utf-8')

#Draw frame in image and save it
def draw_rectangle(x1,y1,x2,y2,pic):
    image = cv2.imread(pic)
    x = int(x1)
    y = int(y1)
    w = int(x2) - x
    h = int(y2) - y
    #rectangle = cv2.rectangle(image,(int(x1),int(y1),int(x2),int(y2)),(255,255,0),1)
    rectangle = cv2.rectangle(image,(x,y,w,h),(255,255,0),1)
    cv2.imwrite(pic,rectangle)

#Draw frame in image according to xml annotation file
def draw_frame_by_xml(src_pic,annot_file):
    if not os.path.exists(raw_image_dir + src_pic):
        print('No source image file')
        return
    if not os.path.exists(xml_file_dir + annot_file):
        print('No annotation file')
        return
    if not os.path.exists(tgt_image_dir):
        print('No target image folder')
        return

    tree = ElementTree() 
    shutil.copyfile(raw_image_dir + src_pic,tgt_image_dir + src_pic)
    tree.parse(xml_file_dir + annot_file)
    root = tree.getroot()
    while(1):
        obj = root.find('object')
        if obj is None:
            break
        bb = obj.find('bndbox')
        xmin = bb.find('xmin').text
        ymin = bb.find('ymin').text
        xmax = bb.find('xmax').text
        ymax = bb.find('ymax').text
        #print(xmin + ' ' + ymin + ' ' + xmax + ' ' + ymax)
        draw_rectangle(xmin,ymin,xmax,ymax,tgt_image_dir + src_pic)
        root.remove(obj)

#resize_pic('000006.png')
#txt2xml('000006.txt','000006.xml')
#draw_frame_by_xml('000006.png','000006.xml')

for i in range(101):
    image_file_name = str(i).zfill(6) + '.png'
    txt_file_name = str(i).zfill(6) + '.txt'
    xml_file_name = str(i).zfill(6) + '.xml'

    resize_pic(image_file_name)
    txt2xml(txt_file_name,xml_file_name)
    draw_frame_by_xml(image_file_name,xml_file_name)    
    if(i%100 == 0):
        print(i)
