import ultralytics
from ultralytics import YOLO
import torch
import numpy as np
import cv2,time,os

def get_speed(position):    
    tempxy=(0,0)
    listspeed=[]
    for i,xy in enumerate(position):
        if i==0:
            tempxy =xy
            print(tempxy)
            continue
        
        speed_x = 0
        speed_y = 0

        speed_x = xy[0]-tempxy[0]
        speed_y = xy[1]-tempxy[1]

        speed = np.sqrt(speed_x ** 2 + speed_y ** 2)
        listspeed.append(speed)

        tempxy =xy
    return listspeed


bbox_color = (0, 255, 0)             
bbox_color2 = (123, 12, 123)
original_bbox_color = (150, 0, 0)
wrong_color = (0, 0, 255) 
bbox_thickness = 2               


bbox_labelstr = {
    'font_size':0.5,         # 字体大小
    'font_thickness':1,   # 字体粗细
    'offset_x':0,          # X 方向，文字偏移距离，向右为正
    'offset_y':-5,        # Y 方向，文字偏移距离，向下为正
}


kpt_color_map = {
    0:{'name':'0', 'color':[255, 255, 255], 'radius':3},      
    1:{'name':'1', 'color':[0, 255, 0], 'radius':3},      
    2:{'name':'2', 'color':[100, 50, 255], 'radius':3},     
    3:{'name':'3', 'color':[0, 255, 255], 'radius':3},     
    4:{'name':'4', 'color':[255, 0, 255], 'radius':3},       
} 


tracking_color_map = {     
    0:{'name':'0', 'color':[0, 0, 255], 'radius':15},      #这里可以仅仅为1
    1:{'name':'1', 'color':[0, 255, 0], 'radius':3},      
    2:{'name':'2', 'color':[100, 50, 255], 'radius':3},     
    3:{'name':'3', 'color':[0, 255, 255], 'radius':3},     
    4:{'name':'4', 'color':[255, 0, 255], 'radius':3},      
} 

'''
0:{'name':'0', 'color':[255, 0, 0], 'radius':2},      # 30度角点
1:{'name':'1', 'color':[0, 255, 0], 'radius':2},      # 60度角点
2:{'name':'2', 'color':[0, 0, 255], 'radius':2},      # 90度角点
3:{'name':'3', 'color':[0, 255, 255], 'radius':2},      # 90度角点
4:{'name':'4', 'color':[255, 0, 255], 'radius':2},      # 90度角点
5:{'name':'5', 'color':[124, 0, 0], 'radius':2},      # 90度角点
6:{'name':'6', 'color':[0, 124, 0], 'radius':2},      # 90度角点
7:{'name':'7', 'color':[0, 0, 124], 'radius':2},      # 90度角点
8:{'name':'8', 'color':[124, 124, 0], 'radius':2},      # 90度角点
9:{'name':'9', 'color':[0, 124, 124], 'radius':2},      # 90度角点
10:{'name':'10', 'color':[124, 0, 124], 'radius':2},      # 90度角点
11:{'name':'11', 'color':[124, 0, 255], 'radius':2},
12:{'name':'12', 'color':[0, 124, 255], 'radius':2},
13:{'name':'13', 'color':[124, 124, 255], 'radius':2},
14:{'name':'14', 'color':[124, 124, 124], 'radius':2},
15:{'name':'15', 'color':[255, 255, 255], 'radius':2},
16:{'name':'16', 'color':[255, 255, 255], 'radius':2},
'''


# 骨架连接 BGR 配色
skeleton_map = [
    {'srt_kpt_id':0, 'dst_kpt_id':1, 'color':[173,255, 173], 'thickness':2}, 
    {'srt_kpt_id':1, 'dst_kpt_id':2, 'color':[196, 75, 255], 'thickness':2},        
    {'srt_kpt_id':2, 'dst_kpt_id':3, 'color':[180, 187, 28], 'thickness':2},        
    {'srt_kpt_id':3, 'dst_kpt_id':4, 'color':[47,255, 255], 'thickness':2}  ]

'''
{'srt_kpt_id':15, 'dst_kpt_id':13, 'color':[196, 75, 255], 'thickness':3},        
{'srt_kpt_id':13, 'dst_kpt_id':11, 'color':[180, 187, 28], 'thickness':3},        
{'srt_kpt_id':16, 'dst_kpt_id':14, 'color':[47,255, 173], 'thickness':3},         
{'srt_kpt_id':14, 'dst_kpt_id':12, 'color':[47,255, 173], 'thickness':3}, 
{'srt_kpt_id':11, 'dst_kpt_id':12, 'color':[47,255, 173], 'thickness':3}, 
{'srt_kpt_id':5, 'dst_kpt_id':11, 'color':[47,255, 173], 'thickness':3}, 
{'srt_kpt_id':6, 'dst_kpt_id':12, 'color':[47,255, 173], 'thickness':3}, 
{'srt_kpt_id':5, 'dst_kpt_id':6, 'color':[47,255, 173], 'thickness':3}, 
{'srt_kpt_id':5, 'dst_kpt_id':7, 'color':[47,255, 173], 'thickness':3}, 
{'srt_kpt_id':6, 'dst_kpt_id':8, 'color':[47,255, 173], 'thickness':3}, 
{'srt_kpt_id':7, 'dst_kpt_id':9, 'color':[47,255, 173], 'thickness':3}, 
{'srt_kpt_id':8, 'dst_kpt_id':10, 'color':[47,255, 173], 'thickness':3}, 
{'srt_kpt_id':1, 'dst_kpt_id':2, 'color':[47,255, 173], 'thickness':3},
{'srt_kpt_id':0, 'dst_kpt_id':1, 'color':[47,255, 173], 'thickness':3},  
{'srt_kpt_id':0, 'dst_kpt_id':2, 'color':[47,255, 173], 'thickness':3}, 
{'srt_kpt_id':1, 'dst_kpt_id':3, 'color':[47,255, 173], 'thickness':3}, 
{'srt_kpt_id':2, 'dst_kpt_id':4, 'color':[47,255, 173], 'thickness':3}, 

{'srt_kpt_id':3, 'dst_kpt_id':5, 'color':[47,255, 173], 'thickness':3}, 
{'srt_kpt_id':4, 'dst_kpt_id':6, 'color':[47,255, 173], 'thickness':3},  
'''
def get_key(dict, value):
    return [k for k, v in dict.items() if v == value]

def visiualize(img_bgr,results):
    num_bbox = len(results.boxes.cls)
    bboxes_xyxy = results.boxes.xyxy.cpu().numpy().astype('uint32')
    bboxes_keypoints = results.keypoints.cpu().numpy().astype('uint32')

    for idx in range(num_bbox): # 遍历每个框
        
        # 获取该框坐标
        bbox_xyxy = bboxes_xyxy[idx] 
        
        # 获取框的预测类别（对于关键点检测，只有一个类别）
        bbox_label = results[0].names[0]
        
        # 画框
        img_bgr = cv2.rectangle(img_bgr, (bbox_xyxy[0], bbox_xyxy[1]), (bbox_xyxy[2], bbox_xyxy[3]), original_bbox_color, bbox_thickness)
        
        # 写框类别文字：图片，文字字符串，文字左上角坐标，字体，字体大小，颜色，字体粗细
        img_bgr = cv2.putText(img_bgr, bbox_label, (bbox_xyxy[0]+bbox_labelstr['offset_x'], bbox_xyxy[1]+bbox_labelstr['offset_y']), cv2.FONT_HERSHEY_SIMPLEX, bbox_labelstr['font_size'], bbox_color, bbox_labelstr['font_thickness'])
        
        bbox_keypoints = bboxes_keypoints[idx] # 该框所有关键点坐标和置信度

        # temp =results.keypoints.conf[idx]
        
        # 画该框的骨架连接
        for skeleton in skeleton_map:
            # if temp[skeleton['srt_kpt_id']] < conf_point:
            #     continue
            
            # 获取起始点坐标
            srt_kpt_id = skeleton['srt_kpt_id']
            srt_kpt_x = bbox_keypoints[srt_kpt_id][0]
            srt_kpt_y = bbox_keypoints[srt_kpt_id][1]
            
            # 获取终止点坐标
            dst_kpt_id = skeleton['dst_kpt_id']
            dst_kpt_x = bbox_keypoints[dst_kpt_id][0]
            dst_kpt_y = bbox_keypoints[dst_kpt_id][1]
            
            # 获取骨架连接颜色
            skeleton_color = skeleton['color']
            
            # 获取骨架连接线宽
            skeleton_thickness = skeleton['thickness']
            
            # 画骨架连接
            img_bgr = cv2.line(img_bgr, (srt_kpt_x, srt_kpt_y),(dst_kpt_x, dst_kpt_y),color=skeleton_color,thickness=skeleton_thickness)

        
        # 画该框的关键点
        for i,kpt_id in enumerate(kpt_color_map):
            # if temp[i]<conf_point:
            #     continue
            # 获取该关键点的颜色、半径、XY坐标
            
            kpt_color = kpt_color_map[kpt_id]['color']
            kpt_radius = kpt_color_map[kpt_id]['radius']
            kpt_x = bbox_keypoints[kpt_id][0]
            kpt_y = bbox_keypoints[kpt_id][1]
            
            # 画圆：图片、XY坐标、半径、颜色、线宽（-1为填充）
            img_bgr = cv2.circle(img_bgr, (kpt_x, kpt_y), kpt_radius, kpt_color, -1)
            
            # 写关键点类别文字：图片，文字字符串，文字左上角坐标，字体，字体大小，颜色，字体粗细
            # kpt_label = str(kpt_id) # 写关键点类别 ID
            kpt_label = str(kpt_color_map[kpt_id]['name']) # 写关键点类别名称
            img_bgr = cv2.putText(img_bgr, kpt_label, (kpt_x+bbox_labelstr['offset_x'], kpt_y+bbox_labelstr['offset_y']), cv2.FONT_HERSHEY_SIMPLEX, bbox_labelstr['font_size'], kpt_color, bbox_labelstr['font_thickness'])


def visiualize_tracker(img_bgr,results,tracker):
    
    bboxes_xyxy = [x.astype('uint32').tolist()[:-1] for x in tracker]
    ids = [int(x[-1]) for x in tracker]

    origin_bboxes_xyxy = results.boxes.xyxy.cpu().numpy().astype('uint32')
    num_bbox = len(origin_bboxes_xyxy)
    bboxes_keypoints = results.keypoints.data.cpu().numpy().astype('uint32')

    for idx in range(num_bbox):

        origin_bbox_xyxy =origin_bboxes_xyxy[idx]
    
        if idx+1 <= len(ids):
            bbox_xyxy = bboxes_xyxy[idx] 
            if sum(bbox_xyxy)>10000: 
                continue
            bbox_label = results[0].names[0]+':'+str(ids[idx])
            # img_bgr = cv2.rectangle(img_bgr, (origin_bbox_xyxy[0], origin_bbox_xyxy[1]), (origin_bbox_xyxy[2], origin_bbox_xyxy[3]), original_bbox_color, bbox_thickness)   #1
            # img_bgr = cv2.rectangle(img_bgr, (bbox_xyxy[0], bbox_xyxy[1]), (bbox_xyxy[2], bbox_xyxy[3]), bbox_color, bbox_thickness) #2
            img_bgr = cv2.putText(img_bgr, bbox_label, (bbox_xyxy[0]+bbox_labelstr['offset_x'], bbox_xyxy[1]+bbox_labelstr['offset_y']), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bbox_color2, 2)

        else:
            bbox_label = results[0].names[0]+':'+'No find!'
            # img_bgr = cv2.rectangle(img_bgr, (origin_bbox_xyxy[0], origin_bbox_xyxy[1]), (origin_bbox_xyxy[2], origin_bbox_xyxy[3]), wrong_color, bbox_thickness)   #1
            img_bgr = cv2.putText(img_bgr, bbox_label, (origin_bbox_xyxy[0]+bbox_labelstr['offset_x'], origin_bbox_xyxy[1]+bbox_labelstr['offset_y']), cv2.FONT_HERSHEY_SIMPLEX, 0.5, wrong_color, 2)
            
        bbox_keypoints = bboxes_keypoints[idx] 

       
        # for skeleton in skeleton_map:
   
        #     srt_kpt_id = skeleton['srt_kpt_id']
        #     srt_kpt_x = bbox_keypoints[srt_kpt_id][0]
        #     srt_kpt_y = bbox_keypoints[srt_kpt_id][1]
        #     dst_kpt_id = skeleton['dst_kpt_id']
        #     dst_kpt_x = bbox_keypoints[dst_kpt_id][0]
        #     dst_kpt_y = bbox_keypoints[dst_kpt_id][1]
        #     skeleton_color = skeleton['color']
        #     skeleton_thickness = skeleton['thickness']
        #     # img_bgr = cv2.line(img_bgr, (srt_kpt_x, srt_kpt_y),(dst_kpt_x, dst_kpt_y),color=skeleton_color,thickness=skeleton_thickness)


        for i,kpt_id in enumerate(kpt_color_map):

            kpt_color = kpt_color_map[kpt_id]['color']
            kpt_radius = kpt_color_map[kpt_id]['radius']
            kpt_x = bbox_keypoints[kpt_id][0]
            kpt_y = bbox_keypoints[kpt_id][1]
            
            img_bgr = cv2.circle(img_bgr, (kpt_x, kpt_y), kpt_radius, kpt_color, -1)
            
            kpt_label = str(kpt_color_map[kpt_id]['name']) 
            # img_bgr = cv2.putText(img_bgr, kpt_label, (kpt_x+bbox_labelstr['offset_x'], kpt_y+bbox_labelstr['offset_y']), cv2.FONT_HERSHEY_SIMPLEX, bbox_labelstr['font_size'], kpt_color, bbox_labelstr['font_thickness'])


def tracker_point_visiualize(img_bgr,results,tracker,ID_connect):

    points = [x.astype('uint32').tolist()[:-1] for x in tracker]
    ids = [int(x[-1]) for x in tracker]

    num_bbox = len(results.boxes.cls)
    bboxes_xyxy = results.boxes.xyxy.cpu().numpy().astype('uint32')
    bboxes_keypoints = results.keypoints.data.cpu().numpy().astype('uint32')

    for idx in range(num_bbox): 
        
       
        bbox_xyxy = bboxes_xyxy[idx] 
        
        # img_bgr = cv2.rectangle(img_bgr, (bbox_xyxy[0], bbox_xyxy[1]), (bbox_xyxy[2], bbox_xyxy[3]), original_bbox_color, bbox_thickness)

        bbox_keypoints = bboxes_keypoints[idx] 


        for i,kpt_id in enumerate(tracking_color_map):
            # if temp[i]<conf_point:
            #     continue
           
            kpt_color = kpt_color_map[kpt_id]['color']
            kpt_radius = kpt_color_map[kpt_id]['radius']
            kpt_x = bbox_keypoints[kpt_id][0]
            kpt_y = bbox_keypoints[kpt_id][1]
            
        
            img_bgr = cv2.circle(img_bgr, (kpt_x, kpt_y), kpt_radius, kpt_color, -1)

        if idx+1 <= len(ids):
            track_points = points[idx] 

            bbox_label = str(ids[idx])
            # print(ID_connect)
            if ID_connect != None :    
                bbox_label = bbox_label + ':' + str( get_key(ID_connect, ids[idx]-1) )
            if ids[idx]==1:
                img_bgr = cv2.putText(img_bgr, bbox_label, (track_points[0]+bbox_labelstr['offset_x'], track_points[1]+bbox_labelstr['offset_y']), cv2.FONT_HERSHEY_SIMPLEX,0.5, bbox_color, 2)
            elif ids[idx]==2:
                img_bgr = cv2.putText(img_bgr, bbox_label, (track_points[0]+bbox_labelstr['offset_x'], track_points[1]+bbox_labelstr['offset_y']), cv2.FONT_HERSHEY_SIMPLEX,0.5, bbox_color2, 2)
            for i,kpt_id in enumerate(tracking_color_map):
                # if temp[i]<conf_point:
                #     continue
            
                kpt_color = kpt_color_map[kpt_id]['color']
                kpt_radius = kpt_color_map[kpt_id]['radius']
                kpt_x = track_points[0]
                kpt_y = track_points[1]

