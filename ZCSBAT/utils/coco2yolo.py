import os
import json
 
# 将目标检测框的信息转为yolo格式
def cc2yolo_bbox(img_width, img_height, bbox):
    dw = 1. / img_width
    dh = 1. / img_height
    x = bbox[0] + bbox[2] / 2.0
    y = bbox[1] + bbox[3] / 2.0
    w = bbox[2]
    h = bbox[3]
    x = format(x * dw, '.5f')
    w = format(w * dw, '.5f')
    y = format(y * dh, '.5f')
    h = format(h * dh, '.5f')
    return (x, y, w, h)
 
# 将关键点keypoints的信息转为yolo格式
def cc2yolo_keypoints(img_width, img_height, keypoints):
    list = []
    dw = 1. / img_width
    dh = 1. / img_height
    keypoint_num = len(keypoints)
    for i in range(keypoint_num):
        if i % 3 == 0:
            list.append(format(keypoints[i] * dw, '.5f'))
        if i % 3 == 1:
            list.append(format(keypoints[i] * dh, '.5f'))
        if i % 3 == 2:
            list.append(keypoints[i])
    result = tuple(list)
    return result
 
json_file_path = r'E:\AI\pose\person_keypoints_val2017.json'
data = json.load(open(json_file_path, 'r'))
yolo_anno_path = r'E:\AI\pose\txt\/'
if not os.path.exists(yolo_anno_path):
    os.makedirs(yolo_anno_path)
 
cate_id_map = {}
num = 0
for cate in data['categories']:
    cate_id_map[cate['id']] = num
    num += 1
 
success_count = 0  # 初始化成功转换的计数器
 
for img in data['images']:
    filename = img['file_name']
    img_width = img['width']
    img_height = img['height']
    img_id = img['id']
    yolo_txt_name = filename.split('.')[0] + '.txt'
 
    with open(yolo_anno_path + yolo_txt_name, 'w') as f:
        for anno in data['annotations']:
            if anno['image_id'] == img_id:
                f.write(str(cate_id_map[anno['category_id']]) + ' ')
                bbox_info = cc2yolo_bbox(img_width, img_height, anno['bbox'])
                keypoints_info = cc2yolo_keypoints(img_width, img_height, anno['keypoints'])
                for item in bbox_info:
                    f.write(item + ' ')
                for item in keypoints_info:
                    f.write(str(item) + ' ')
                f.write('\n')
 
    # 打印转换成功的提示，并增加计数器
    print(f'成功转换: {filename}')
    success_count += 1  # 增加成功转换的计数
 
# 打印总共成功转换的文件数量
print(f'总共成功转换了 {success_count} 个文件')