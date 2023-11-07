import os
import random
import xml.etree.ElementTree as ET

import numpy as np

from utils.utils import get_classes

#ラベルリストの指定
classes_path        = 'model_data/cls_classes.txt'
#学習データの何パーセントを評価データに用いるか
trainval_percent    = 1.0
train_percent       = 1.0
#保存先のフォルダ名
VOCdevkit_path  = 'train_data'
#保存するファイル名
VOCdevkit_sets  = ['train', 'val']
classes, _      = get_classes(classes_path)


photo_nums  = np.zeros(len(VOCdevkit_sets))
nums        = np.zeros(len(classes))
def convert_annotation(image_id, list_file):

    in_file = open(os.path.join(VOCdevkit_path, 'xml/%s.xml'%(image_id)), encoding='utf-8')
    tree=ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = 0 
        if obj.find('difficult')!=None:
            difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)), int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text)))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))
        
        nums[classes.index(cls)] = nums[classes.index(cls)] + 1
        
if __name__ == "__main__":
    random.seed(0)
    if " " in os.path.abspath(VOCdevkit_path):
        raise ValueError("Input path error.")

    
    print("Generate txt in ImageSets.")
    xmlfilepath     = os.path.join(VOCdevkit_path, 'xml')
    saveBasePath    = os.path.join(VOCdevkit_path, 'saved')
    
    temp_xml        = os.listdir(xmlfilepath)
    total_xml       = []
    for xml in temp_xml:
        if xml.endswith(".xml"):
            total_xml.append(xml)
            
    num     = len(total_xml)  
    list    = range(num)  
    tv      = int(num*trainval_percent)  
    tr      = int(tv*train_percent)  
    trainval= random.sample(list,tv)  
    train   = random.sample(trainval,tr)  
    print("train and val size",tv)
    print("train size",tr)
    ftrainval   = open(os.path.join(saveBasePath,'trainval.txt'), 'w')  
    ftest       = open(os.path.join(saveBasePath,'test.txt'), 'w')  
    ftrain      = open(os.path.join(saveBasePath,'train.txt'), 'w')  
    fval        = open(os.path.join(saveBasePath,'val.txt'), 'w')  
    
    for i in list:  
        name=total_xml[i][:-4]+'\n'  
        if i in trainval:  
            ftrainval.write(name)  
            if i in train:  
                ftrain.write(name)  
            else:  
                fval.write(name)  
        else:  
            ftest.write(name)  
    
    ftrainval.close()  
    ftrain.close()  
    fval.close()  
    ftest.close()
    print("Generate txt in ImageSets done.")

    
    print("Generate train.txt and val.txt for train.")
    type_index = 0
    for image_set in VOCdevkit_sets:
        #読み込みファイルのアドレス指定
        image_ids = open(os.path.join(VOCdevkit_path, 'saved/%s.txt'%( image_set)), encoding='utf-8').read().strip().split()
        #保存先ファイルアドレス指定
        list_file = open('train_data/%s.txt'%(image_set), 'w', encoding='utf-8')
        for image_id in image_ids:
            list_file.write('%s/img/%s.jpg'%(os.path.abspath(VOCdevkit_path), image_id))

            convert_annotation(image_id, list_file)
            list_file.write('\n')
        photo_nums[type_index] = len(image_ids)
        type_index += 1
        list_file.close()
    print("Generate train.txt and val.txt in train_data folder as train_data.")
    
    def printTable(List1, List2):
        for i in range(len(List1[0])):
            print("|", end=' ')
            for j in range(len(List1)):
                print(List1[j][i].rjust(int(List2[j])), end=' ')
                print("|", end=' ')
            print()

    str_nums = [str(int(x)) for x in nums]
    tableData = [
        classes, str_nums
    ]
    colWidths = [0]*len(tableData)
    len1 = 0
    for i in range(len(tableData)):
        for j in range(len(tableData[i])):
            if len(tableData[i][j]) > colWidths[i]:
                colWidths[i] = len(tableData[i][j])
    printTable(tableData, colWidths)

    if photo_nums[0] <= 500:
        print("finish")

    if np.sum(nums) == 0:
        print("No data.")