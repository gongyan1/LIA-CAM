import os
import os.path as osp
import json
import random
from mmengine import list_from_file

# root = r'C:\Users\SXQ\Desktop\车道线检测\MyDataWithAngle'
root = r'Lane_Withangle_test'
data_root_path = osp.join(root, 'data')
angle_path = osp.join(root, 'angle.json')
alldata_txt_path = osp.join(root, 'alldata.txt')
train_txt_path = osp.join(root, 'train.txt')
val_txt_path = osp.join(root, 'val.txt')

def generate_txts():
    txts = open(alldata_txt_path, 'w')
    angle = open(angle_path, 'r')
    angle_data = json.load(angle)
    for p1, p2 in zip(os.listdir(osp.join(data_root_path,'img')), os.listdir(osp.join(data_root_path,'label'))):
        txts.write((osp.splitext(p1)[0]) + ' '+ str(angle_data[p1]) + '\n')
    txts.close()
    
generate_txts()

def generate_split_txt(train_ratio, val_ratio):
    assert round(train_ratio+val_ratio) == 1.0
    alldata = list_from_file(alldata_txt_path)
    random.shuffle(alldata)
    train_len = int(len(alldata)*train_ratio)
    train_data = alldata[:train_len]
    val_data = alldata[train_len:]

    train_txt = open(train_txt_path, 'w')
    val_txt = open(val_txt_path, 'w')
    for x in train_data:
        train_txt.write(x+'\n')
    for x in val_data:
        val_txt.write(x+'\n')
    
    train_txt.close()
    val_txt.close()

generate_split_txt(0.2, 0.8)
