import os
import json
import shutil
import numpy as np
from tqdm import tqdm
import cv2

def rec_xyxy2rec_4xy(rectangles):
    new_list = [[] for _ in range(len(rectangles))]

    if len(rectangles[0])==2:
        for i in range(len(rectangles)):
            rect = rectangles[i]
            
            x1, y1 = rect[0]
            x2, y2 = rect[1]
            new_list[i] = [[x1, y1], [x1, y2], [x2, y2], [x2, y1]]
    return new_list

def getpoints(labelme):   #将 1，2，3，4，5，1，2，3，4，1，2，3 这样的列表 拆分 [1,2,3,4,5] [1,2,3,4] [1,2,3]
    sublist1 = [] 
    points = []
    temppoint = []
    for i in labelme['shapes']:
        if i['label'].isdigit():
        
            if not sublist1 or int(i['label']) == sublist1[-1] + 1:

                sublist1.append(int(i['label']))
                temppoint.append([int(i['points'][0][0]),int(i['points'][0][1])])

            else:
                points.append(temppoint)
                temppoint = []
                temppoint.append([int(i['points'][0][0]),int(i['points'][0][1])])
                sublist1 = [int(i['label'])]
    if sublist1:
        points.append(temppoint)
        
    for i,point in enumerate(points): 
        if len(point) != 5:                 #判断对齐需要的关节点数量
            a = len(point)
            for j in range(5-a):
                points[i].append([0,0])
    return points   #[2,5,2] 两组


# 框的类别          xuyao 修改
bbox_class = {
    'fish':0,
    'red-male':1,
    'common-female':2  
}

# 关键点的类别    xuyao 修改
keypoint_class = ['1', '2', '3','4','5']

try:
    os.mkdir('labels')
    os.mkdir('labels/train')
    os.mkdir('labels/val')
except:
    print('已经创建文件夹！！')

# 需要同时转换labelme 与 x-anylabel才行
def process_single_json(labelme_path, save_folder='../../labels/train'):  #问题只识别一种标签
    
    with open(labelme_path, 'r', encoding='utf-8') as f:
            labelme = json.load(f)

    img_width = labelme['imageWidth']   # 图像宽度
    img_height = labelme['imageHeight'] # 图像高度
    recs = [ann['points'] for ann in labelme['shapes'] if ann['shape_type']=='rectangle']

    points = getpoints(labelme)
    if len(recs[0])==2:
        temprec = rec_xyxy2rec_4xy(recs)    # 这段代码转换了 labelme与x-anylabel   xyxy24xy
    else:
        temprec = [[[point for point in rect] for rect in group] for group in recs]
    
    PointOrder = []   #获取 哪些点在哪一个框框里
    for i in range(len(temprec)):
        box = np.array(temprec[i]).astype(int)
        if points!=None:
            pointnum = []   
            for j in range(len(points)):
                num = 0 
                for k in range(len(points[j])):
                    if cv2.pointPolygonTest(box, points[j][k], False)==1:
                        num+=1
                pointnum.append(num)
           #  ---------------------如果都是相同的  旧继续  ---------------------  这里有缺陷

            PointOrder.append(pointnum.index(max(pointnum)))

    suffix = labelme_path.split('.')[-2]
    yolo_txt_path = suffix + '.txt'
    with open(yolo_txt_path, 'w', encoding='utf-8') as f:
        for i in range(len(recs)):
            rec = recs[i]
            yolo_str = ''

            bbox_class_id = bbox_class['fish']   ## -----------------记得修改！！
            yolo_str += '{} '.format(bbox_class_id)

            # 左上角和右下角的 XY 像素坐标
            if len(rec)==2:
                bbox_top_left_x = int(min(rec[0][0], rec[1][0]))
                bbox_bottom_right_x = int(max(rec[0][0], rec[1][0]))
                bbox_top_left_y = int(min(rec[0][1], rec[1][1]))
                bbox_bottom_right_y = int(max(rec[0][1], rec[1][1]))
            else:
                bbox_top_left_x = int(min(rec[0][0], rec[2][0]))
                bbox_bottom_right_x = int(max(rec[0][0], rec[2][0]))
                bbox_top_left_y = int(min(rec[0][1], rec[2][1]))
                bbox_bottom_right_y = int(max(rec[0][1], rec[2][1]))

            print(bbox_top_left_x,bbox_bottom_right_x,bbox_top_left_y,bbox_bottom_right_y)
            # 框中心点的 XY 像素坐标
            bbox_center_x = int((bbox_top_left_x + bbox_bottom_right_x) / 2)
            bbox_center_y = int((bbox_top_left_y + bbox_bottom_right_y) / 2)
            # 框宽度
            bbox_width = bbox_bottom_right_x - bbox_top_left_x
            # 框高度
            bbox_height = bbox_bottom_right_y - bbox_top_left_y
            # 框中心点归一化坐标
            bbox_center_x_norm = bbox_center_x / img_width
            bbox_center_y_norm = bbox_center_y / img_height
            # 框归一化宽度
            bbox_width_norm = bbox_width / img_width
            # 框归一化高度
            bbox_height_norm = bbox_height / img_height

            yolo_str += '{:.5f} {:.5f} {:.5f} {:.5f} '.format(bbox_center_x_norm, bbox_center_y_norm, bbox_width_norm, bbox_height_norm)

            for point in points[PointOrder[i]]:
                if point[0] == 0 and point[1] == 0:
                    yolo_str += '{:.5f} {:.5f} {} '.format(0, 0, 0) # 2-可见不遮挡 1-遮挡 0-没有点
                else:
                    keypoint_x_norm = point[0] / img_width
                    keypoint_y_norm = point[1] / img_height
                    yolo_str += '{:.5f} {:.5f} {} '.format(keypoint_x_norm, keypoint_y_norm, 2) # 2-可见不遮挡 1-遮挡 0-没有点
                
            # 写入 txt 文件中
            f.write(yolo_str + '\n')
    shutil.move(yolo_txt_path, save_folder)
    print('{} --> {} 转换完成'.format(labelme_path, yolo_txt_path)) 

os.chdir('labelme_jsons/train')
save_folder = '../../labels/train'
for labelme_path in os.listdir():
    try:
        process_single_json(labelme_path, save_folder=save_folder)
    except:
        print('******有误******', labelme_path)
print('YOLO格式的txt标注文件已保存至 ', save_folder)


os.chdir('../../')
os.chdir('labelme_jsons/val')
save_folder = '../../labels/val'
for labelme_path in os.listdir():
    try:
        process_single_json(labelme_path, save_folder=save_folder)
    except:
        print('******有误******', labelme_path)
print('YOLO格式的txt标注文件已保存至 ', save_folder)