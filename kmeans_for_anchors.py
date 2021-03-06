import numpy as np
import xml.etree.ElementTree as ET
import glob
import random


def cas_iou(box, cluster):
    """
    计算一个anchor到9个中心簇的距离
    :param box: size （2，） eg. [0.218, 0.32]
    :param cluster:  size （9，2） eg. [[0.088, 0.122], [0.99, 0.734], ...]
    :return:
    """
    x = np.minimum(cluster[:, 0], box[0])
    y = np.minimum(cluster[:, 1], box[1])

    intersection = x * y
    area1 = box[0] * box[1]

    area2 = cluster[:, 0] * cluster[:, 1]
    iou = intersection / (area1 + area2 -intersection)

    return iou


def avg_iou(box, cluster):
    """
    计算某一个anchor到cluster最大的那个，然后将其平均
    """
    return np.mean([np.max(cas_iou(box[i], cluster)) for i in range(box.shape[0])])


def kmeans(box, k):
    # 取出一共有多少框，即多少个(w,h)
    row = box.shape[0]  # 30638
    
    # 每个框(w,h)到k个中心点的距离
    distance = np.empty((row, k))   # size (30638, 9)
    
    # 最后的聚类位置
    last_clu = np.zeros((row,))     # size (30638)

    np.random.seed()

    # 随机选9个框（w，h）当聚类中心
    cluster = box[np.random.choice(row, k, replace=False)]  # size (9,2) eg. [[0.088, 0.12],...]
    # cluster = random.sample(row, k)
    while True:
        # 计算所有anchor到cluster的距离
        for i in range(row):
            distance[i] = 1 - cas_iou(box[i], cluster)
        
        # 取出所有anchor到k个中最短的下标
        near = np.argmin(distance, axis=1)  # size 30638, 30638个框到9个簇心的最短距离的位置

        if (last_clu == near).all():
            break
        
        # 求每一个类的中位点
        for j in range(k):  # 第j个类
            cluster[j] = np.median(
                box[near == j], axis=0)     # box[near==j]表示 30638个框到第0类最近的所有框   的中位数

        last_clu = near

    return cluster


def load_data(path):
    """
    VOC 数据集每个图片的标注*.xml
    :param path:
    :return: 返回所有图像中每个object的w和h
    """
    data = []
    # 对于每一个xml都寻找box
    for xml_file in glob.glob('{}/*xml'.format(path)):
        tree = ET.parse(xml_file)
        height = int(tree.findtext('./size/height'))
        width = int(tree.findtext('./size/width'))
        # 对于每一个目标都获得它的宽高
        for obj in tree.iter('object'):
            xmin = int(float(obj.findtext('bndbox/xmin'))) / width
            ymin = int(float(obj.findtext('bndbox/ymin'))) / height
            xmax = int(float(obj.findtext('bndbox/xmax'))) / width
            ymax = int(float(obj.findtext('bndbox/ymax'))) / height

            xmin = np.float64(xmin)
            ymin = np.float64(ymin)
            xmax = np.float64(xmax)
            ymax = np.float64(ymax)
            # 得到宽高
            data.append([xmax-xmin, ymax-ymin])
    return np.array(data)


if __name__ == '__main__':
    # 运行该程序会计算'./VOCdevkit/VOC2007/Annotations'的xml
    # 会生成yolo_anchors.txt
    SIZE = 416
    anchors_num = 9
    # 载入数据集，可以使用VOC的xml
    path = r'./VOCdevkit/VOC2007/Annotations'
    
    # 载入所有的xml
    # 存储格式为转化为比例后的width,height
    data = load_data(path)  # size 30638 eg. [[0.21, 0.32], [0.10, 0.32], ...]
    
    # 使用k聚类算法
    out = kmeans(data, anchors_num)  # size(9,2)
    out = out[np.argsort(out[:, 0])]
    print('acc:{:.2f}%'.format(avg_iou(data, out) * 100))
    print(out*SIZE)
    data = out*SIZE
    f = open("model_data/yolo_anchors.txt", 'w')
    row = np.shape(data)[0]
    for i in range(row):
        if i == 0:
            x_y = "%d,%d" % (data[i][0], data[i][1])
        else:
            x_y = ", %d,%d" % (data[i][0], data[i][1])
        f.write(x_y)
    f.close()