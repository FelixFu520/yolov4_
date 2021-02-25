import cv2  
from random import shuffle
import numpy as np
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from PIL import Image
from utils.utils import bbox_iou, merge_bboxes


def jaccard(_box_a, _box_b):
    b1_x1, b1_x2 = _box_a[:, 0] - _box_a[:, 2] / 2, _box_a[:, 0] + _box_a[:, 2] / 2  # 0,0 w,h --> x1y1 x2y2    b1_x1 size=(N)
    b1_y1, b1_y2 = _box_a[:, 1] - _box_a[:, 3] / 2, _box_a[:, 1] + _box_a[:, 3] / 2
    b2_x1, b2_x2 = _box_b[:, 0] - _box_b[:, 2] / 2, _box_b[:, 0] + _box_b[:, 2] / 2
    b2_y1, b2_y2 = _box_b[:, 1] - _box_b[:, 3] / 2, _box_b[:, 1] + _box_b[:, 3] / 2
    box_a = torch.zeros_like(_box_a)
    box_b = torch.zeros_like(_box_b)
    box_a[:, 0], box_a[:, 1], box_a[:, 2], box_a[:, 3] = b1_x1, b1_y1, b1_x2, b1_y2
    box_b[:, 0], box_b[:, 1], box_b[:, 2], box_b[:, 3] = b2_x1, b2_y1, b2_x2, b2_y2
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),  # 同一个box和9个anchors对比，取最大的xy。size(7,9,2)
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)

    inter = inter[:, :, 0] * inter[:, :, 1]     # size (7,9)
    # 计算先验框和真实框各自的面积
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    # 求IOU
    union = area_a + area_b - inter
    return inter / union  # [A,B]


# ---------------------------------------------------#
#   平滑标签
# ---------------------------------------------------#
def smooth_labels(y_true, label_smoothing,num_classes):
    return y_true * (1.0 - label_smoothing) + label_smoothing / num_classes


def box_ciou(b1, b2):
    """
    输入为：
    ----------
    b1: tensor, shape=(N, 4), xywh  预测
    b2: tensor, shape=(N, 4), xywh  真实

    返回为：
    -------
    ciou: tensor, shape=(N, 1)
    """
    # 求出预测框左上角右下角
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh/2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half
    # 求出真实框左上角右下角
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh/2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    # 求真实框和预测框所有的iou
    intersect_mins = torch.max(b1_mins, b2_mins)
    intersect_maxes = torch.min(b1_maxes, b2_maxes)
    intersect_wh = torch.max(intersect_maxes - intersect_mins, torch.zeros_like(intersect_maxes))
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    union_area = b1_area + b2_area - intersect_area
    iou = intersect_area / torch.clamp(union_area,min = 1e-6)

    # 计算中心的差距
    center_distance = torch.sum(torch.pow((b1_xy - b2_xy), 2), axis=-1)
    
    # 找到包裹两个框的最小框的左上角和右下角
    enclose_mins = torch.min(b1_mins, b2_mins)
    enclose_maxes = torch.max(b1_maxes, b2_maxes)
    enclose_wh = torch.max(enclose_maxes - enclose_mins, torch.zeros_like(intersect_maxes))
    # 计算对角线距离
    enclose_diagonal = torch.sum(torch.pow(enclose_wh,2), axis=-1)
    ciou = iou - 1.0 * (center_distance) / torch.clamp(enclose_diagonal,min = 1e-6)
    
    v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(b1_wh[..., 0]/torch.clamp(b1_wh[..., 1],min = 1e-6)) - torch.atan(b2_wh[..., 0]/torch.clamp(b2_wh[..., 1],min = 1e-6))), 2)
    alpha = v / torch.clamp((1.0 - iou + v),min=1e-6)
    ciou = ciou - alpha * v
    return ciou


def clip_by_tensor(t,t_min,t_max):
    t=t.float()
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min  # 使 t_min < t
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max  # 使 t < t_max
    return result


def MSELoss(pred,target):
    return (pred-target)**2


def BCELoss(pred,target):
    epsilon = 1e-7
    pred = clip_by_tensor(pred, epsilon, 1.0 - epsilon)
    output = -target * torch.log(pred) - (1.0 - target) * torch.log(1.0 - pred)
    return output


class YOLOLoss(nn.Module):
    def __init__(self, anchors, num_classes, img_size, label_smooth=0, cuda=True):
        super(YOLOLoss, self).__init__()
        self.anchors = anchors  # anchors size (9,2)
        self.num_anchors = len(anchors)     # anchors数量 9
        self.num_classes = num_classes      # 类别数量 20
        self.bbox_attrs = 5 + num_classes   # bbox 25
        self.img_size = img_size    # （416，416）
        self.feature_length = [img_size[0]//32, img_size[0]//16, img_size[0]//8]  # feature map大小【13，26，52】
        self.label_smooth = label_smooth    # 0

        self.ignore_threshold = 0.5
        self.lambda_conf = 1.0
        self.lambda_cls = 1.0
        self.lambda_loc = 1.0
        self.cuda = cuda

    def forward(self, input, targets=None):
        """
        计算损失值
        :param input: 网络的输出，是三个yolo layer层之一， （4，75，13，13）
        :param targets: 标签，是长度为4的列表，里面有不同的object数量
        :return:
        """
        # input为bs,3*(5+num_classes),13/26/52,13/26/52
        
        # 一共多少张图片
        bs = input.size(0)      # 4
        # 特征层的高
        in_h = input.size(2)    # 13
        # 特征层的宽
        in_w = input.size(3)    # 13

        # 计算步长
        # 每一个特征点对应原来的图片上多少个像素点
        # 如果特征层为13x13的话，一个特征点就对应原来的图片上的32个像素点
        stride_h = self.img_size[1] / in_h      # 32
        stride_w = self.img_size[0] / in_w      # 32

        # 把先验框的尺寸调整成特征层大小的形式
        # 计算出先验框在特征层上对应的宽高
        scaled_anchors = [(a_w / stride_w, a_h / stride_h) for a_w, a_h in self.anchors]    # size(9,2)
        # bs,3*(5+num_classes),13,13 -> bs,3,13,13,(5+num_classes)
        prediction = input.view(bs, int(self.num_anchors/3),
                                self.bbox_attrs, in_h, in_w).permute(0, 1, 3, 4, 2).contiguous()
        
        # 对prediction预测进行调整
        conf = torch.sigmoid(prediction[..., 4])  # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.

        # 修改targets格式，获得对应的anchors，使得其匹配prediction格式，作出mask。总结下：target到特征图上，然后作出target——>anchors的mask
        # mask：(bs,3, featuremap_w,featuremap_h)           这个特征图含有object的mask, 是否含有object是通过target来确定的
        # noobj_mask:(bs,3, featuremap_w,featuremap_h)这个特征图不含object的mask，是否含有object是通过target来确定的，后面还会通过get_ignore增加
        # t_box:(bs,3, featuremap_w,featuremap_h,4)         这个特征图含有object的 box(tx,ty,tw,th) 是否含有object是通过target来确定的
        # tconf:(bs,3, featuremap_w,featuremap_h)           这个特征图含有object的 置信度
        # tcls:(bs,3, featuremap_w,featuremap_h，20)         这个特征图含有object的 类别位置
        # box_loss_scale_x:(bs,3, featuremap_w,featuremap_h)  这个特征图含有object的 相对于整张图的w target[b][i, 2]
        # box_loss_scale_y:(bs,3, featuremap_w,featuremap_h)
        mask, noobj_mask, t_box, tconf, tcls, box_loss_scale_x, box_loss_scale_y = self.get_target(targets, scaled_anchors, in_w, in_h, self.ignore_threshold)

        # 1、将prediction转换成特征图模式，即参考图。2、扩充noobj_mask，将pred和gt的iou大于阈值的点，noobj标记为0
        # noobj_mask:(bs,3, featuremap_w,featuremap_h)这个特征图不含object的mask, 是在get_target上增加了一些，增加内容是
        #           prediction预测的和target预测的IOU大于阈值，noobj赋值为0
        # pred_boxes_for_ciou:(bs,3, featuremap_w,featuremap_h，4) 将predict的格式调整为特征图格式
        noobj_mask, pred_boxes_for_ciou = self.get_ignore(prediction, targets, scaled_anchors, in_w, in_h, noobj_mask)

        if self.cuda:
            mask, noobj_mask = mask.cuda(), noobj_mask.cuda()
            box_loss_scale_x, box_loss_scale_y= box_loss_scale_x.cuda(), box_loss_scale_y.cuda()
            tconf, tcls = tconf.cuda(), tcls.cuda()
            pred_boxes_for_ciou = pred_boxes_for_ciou.cuda()
            t_box = t_box.cuda()

        box_loss_scale = 2-box_loss_scale_x*box_loss_scale_y  # （4，3，13，13）
        #  losses.
        # box_ciou(含有object的预测框（N，4），含有object的真实框（N，4））--> (N，1)
        # 为什么 * box_loss_scale[mask.bool()] ？
        ciou = (1 - box_ciou(pred_boxes_for_ciou[mask.bool()], t_box[mask.bool()])) * box_loss_scale[mask.bool()]  # size N

        # bbox误差
        loss_loc = torch.sum(ciou / bs)

        # 置信度误差 = 有目标的损失 和 无目标的损失
        # 二分类交叉墒BCELoss(conf,mask):(4,3,13,13)  conf:(4,3,13,13)  mask:(4,3,13,13)
        loss_conf = torch.sum(BCELoss(conf, mask) * mask / bs) + \
                    torch.sum(BCELoss(conf, mask) * noobj_mask / bs)

        # 类别误差
        # smooth_labels(y_true, label_smoothing, num_classes)
        # y_true:[N,20] N是本层head所检测出来的object个数
        # label_smoothing: 0
        # num_classes: 20
        loss_cls = torch.sum(BCELoss(pred_cls[mask == 1], smooth_labels(tcls[mask == 1],self.label_smooth,self.num_classes))/bs)

        loss = loss_conf * self.lambda_conf + loss_cls * self.lambda_cls + loss_loc * self.lambda_loc
        return loss, loss_conf.item(), loss_cls.item(), loss_loc.item()

    def get_target(self, target, anchors, in_w, in_h, ignore_threshold):
        """
        功能：通过target(N,5)、anchors(9,2)、featuremapw/h获得   所有anchor中与target最匹配的anchor
        target: 真实标签，size bs=4，每个元素是某张图所包含目标object的个数（N，5）
        anchors: 特征图featuremap的anchors （9，2）
        in_w:   featuremap的w
        in_h:   featuremap的h
        ignore_threshold: 忽略阈值

        return:
        mask: (bs,3, featuremap_w,featuremap_h)           这个特征图含有object的mask
        noobj_mask:(bs,3, featuremap_w,featuremap_h)      这个特征图不含object的mask
        t_box:(bs,3, featuremap_w,featuremap_h,4)         这个特征图含有object的 box(tx,ty,tw,th)
        tconf:(bs,3, featuremap_w,featuremap_h)           这个特征图含有object的 置信度
        tcls:(bs,3, featuremap_w,featuremap_h，20)         这个特征图含有object的 类别位置
        box_loss_scale_x:(bs,3, featuremap_w,featuremap_h)  这个特征图含有object的 相对于整张图的w
        box_loss_scale_y:(bs,3, featuremap_w,featuremap_h)
        """
        flag = 0    # 测试使用，无意义
        # 计算一共有多少张图片
        bs = len(target)
        # 获得先验框
        anchor_index = [[0,1,2],[3,4,5],[6,7,8]][self.feature_length.index(in_w)]   # eg. [0,1,2], 通过in_w特征图大小来判断是那个yolo layer， 获取anchors的index
        subtract_index = [0,3,6][self.feature_length.index(in_w)]   # eg. 0 后面用于减
        # 创建全是0或者全是1的阵列
        mask = torch.zeros(bs, int(self.num_anchors/3), in_h, in_w, requires_grad=False)        # （4，3，13，13） 有object的mask
        noobj_mask = torch.ones(bs, int(self.num_anchors/3), in_h, in_w, requires_grad=False)   # （4，3，13，13） 无object的mask

        tx = torch.zeros(bs, int(self.num_anchors/3), in_h, in_w, requires_grad=False)          # （4，3，13，13） 存放标签在featuremap的x
        ty = torch.zeros(bs, int(self.num_anchors/3), in_h, in_w, requires_grad=False)          # （4，3，13，13）
        tw = torch.zeros(bs, int(self.num_anchors/3), in_h, in_w, requires_grad=False)          # （4，3，13，13）
        th = torch.zeros(bs, int(self.num_anchors/3), in_h, in_w, requires_grad=False)          # （4，3，13，13）
        t_box = torch.zeros(bs, int(self.num_anchors/3), in_h, in_w, 4, requires_grad=False)    # （4，3，13，13，4）
        tconf = torch.zeros(bs, int(self.num_anchors/3), in_h, in_w, requires_grad=False)       # （4，3，13，13）
        tcls = torch.zeros(bs, int(self.num_anchors/3), in_h, in_w, self.num_classes, requires_grad=False)  # （4，3，13，13，20）

        box_loss_scale_x = torch.zeros(bs, int(self.num_anchors/3), in_h, in_w, requires_grad=False)    # （4，3，13，13）
        box_loss_scale_y = torch.zeros(bs, int(self.num_anchors/3), in_h, in_w, requires_grad=False)    # （4，3，13，13）
        for b in range(bs):     # 一个batch size中每张图片
            if len(target[b]) == 0:
                continue

            # 计算出在特征层上的点位
            gxs = target[b][:, 0:1] * in_w     # gxs表示真实标签 左上角点 在特征图上的位置 size(N,1)
            gys = target[b][:, 1:2] * in_h     # (N,1) N表示一张途中object数量
            
            gws = target[b][:, 2:3] * in_w  # (N,1)
            ghs = target[b][:, 3:4] * in_h  # (N,1)

            # 计算出 这个object 属于哪个网格
            gis = torch.floor(gxs)  # (N,1)
            gjs = torch.floor(gys)  # (N,1)
            
            # 计算真实框/标签的位置(或者叫w和h）
            gt_box = torch.FloatTensor(torch.cat([torch.zeros_like(gws), torch.zeros_like(ghs), gws, ghs], 1))  # (N,4) eg. [0,0,x,y]
            
            # 计算出所有先验框的位置（或者叫w和h）
            anchor_shapes = torch.FloatTensor(torch.cat((torch.zeros((self.num_anchors, 2)), torch.FloatTensor(anchors)), 1))   # (9,4) eg. [0,0,x,y]
            # 计算真实和anchors的重合程度，计算gt_box（N，4）与anchors（9，4）的重合度。
            anch_ious = jaccard(gt_box, anchor_shapes)  # size (N,9)

            # Find the best matching anchor box
            best_ns = torch.argmax(anch_ious, dim=-1)    # size=N, tensor([0, 1, 3, 5, 3, 5, 1, 5]) 意思是一个object对应一个iou最大的anchor 的下标
            for i, best_n in enumerate(best_ns):    # 一张图片中，多个object
                if best_n not in anchor_index:  # 本yolo layer层
                    continue
                # Masks
                gi = gis[i].long()  # size 1 ， type tensor 第一个object 在featuremap的i
                gj = gjs[i].long()  # 一个object 在featuremap 的 j
                gx = gxs[i]  # 5.3750
                gy = gys[i]  # 3.4219
                gw = gws[i]  # 5.0625
                gh = ghs[i]  # 5.1562
                if (gj < in_h) and (gi < in_w):  # gi，gj小于特征图的宽和高
                    best_n = best_n - subtract_index    # best_n - subtract_index 表示 第几个anchor
                    # 判定哪些先验框内部真实的存在物体
                    noobj_mask[b, best_n, gj, gi] = 0
                    mask[b, best_n, gj, gi] = 1
                    flag += 1
                    # print(bs, "batch :", i, "object:")
                    # 计算先验框中心调整参数
                    tx[b, best_n, gj, gi] = gx
                    ty[b, best_n, gj, gi] = gy
                    # 计算先验框宽高调整参数
                    tw[b, best_n, gj, gi] = gw
                    th[b, best_n, gj, gi] = gh
                    # 用于获得xywh的比例
                    box_loss_scale_x[b, best_n, gj, gi] = target[b][i, 2]   # w
                    box_loss_scale_y[b, best_n, gj, gi] = target[b][i, 3]   # h,相对于整张图
                    # 物体置信度
                    tconf[b, best_n, gj, gi] = 1
                    # 种类
                    tcls[b, best_n, gj, gi, target[b][i, 4].long()] = 1
                else:
                    print('Step {0} out of bound'.format(b))
                    print('gj: {0}, height: {1} | gi: {2}, width: {3}'.format(gj, in_h, gi, in_w))
                    continue
        t_box[...,0] = tx
        t_box[...,1] = ty
        t_box[...,2] = tw
        t_box[...,3] = th
        print("flags:", flag)
        return mask, noobj_mask, t_box, tconf, tcls, box_loss_scale_x, box_loss_scale_y

    def get_ignore(self, prediction, target, scaled_anchors, in_w, in_h, noobj_mask):
        """
        功能：1、将prediction转换成特征图模式，即参考图。2、扩充noobj_mask，将pred和gt的iou大于阈值的点，noobj标记为0
        input：
            prediction：(bs, 3, feature_w, feature_h, 25)
            target：list size 4  每个是（N，5）
            scaled_anchors：（9，2）
            in_w/h: 13/26/52
            noobj_mask: no object mask (bs,3,13,13)
        return:
            noobj_mask:
            pred_boxes:

        """
        bs = len(target)
        anchor_index = [[0,1,2],[3,4,5],[6,7,8]][self.feature_length.index(in_w)]
        scaled_anchors = np.array(scaled_anchors)[anchor_index]
        # 预测框的xy位置的调整参数
        x = torch.sigmoid(prediction[..., 0])  
        y = torch.sigmoid(prediction[..., 1])
        # 预测框的宽高调整参数
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height

        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor

        # 生成网格，先验框中心，网格左上角
        grid_x = torch.linspace(0, in_w - 1, in_w).repeat(in_w, 1).repeat(int(bs*self.num_anchors/3), 1, 1).view(x.shape).type(FloatTensor)
        grid_y = torch.linspace(0, in_h - 1, in_h).repeat(in_h, 1).t().repeat(int(bs*self.num_anchors/3), 1, 1).view(y.shape).type(FloatTensor)

        # 生成先验框的宽高
        anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
        anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))

        anchor_w = anchor_w.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(w.shape)   # （4，3，13，13）
        anchor_h = anchor_h.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(h.shape)
        
        # 计算调整后的先验框xy与wh, 理论见公式
        pred_boxes = FloatTensor(prediction[..., :4].shape)     # （4，3，13，13，4）
        pred_boxes[..., 0] = x + grid_x
        pred_boxes[..., 1] = y + grid_y
        pred_boxes[..., 2] = torch.exp(w) * anchor_w
        pred_boxes[..., 3] = torch.exp(h) * anchor_h
        for i in range(bs):  # 一个batchsize中每张图片
            pred_boxes_for_ignore = pred_boxes[i]   # （3，13，13，4） 预测的所有bbox
            pred_boxes_for_ignore = pred_boxes_for_ignore.view(-1, 4)
            if len(target[i]) > 0:
                gx = target[i][:, 0:1] * in_w   # 真实标签（相对于整张图） --> 真实标签（相对于特征图）
                gy = target[i][:, 1:2] * in_h
                gw = target[i][:, 2:3] * in_w
                gh = target[i][:, 3:4] * in_h
                gt_box = torch.FloatTensor(torch.cat([gx, gy, gw, gh],-1)).type(FloatTensor)

                anch_ious = jaccard(gt_box, pred_boxes_for_ignore)  # （N，507=3*13*13）N表示一张图中有的object个数
                anch_ious_max, _ = torch.max(anch_ious,dim=0)   # size（507）
                anch_ious_max = anch_ious_max.view(pred_boxes[i].size()[:3])    # （3，13，13）
                noobj_mask[i][anch_ious_max>self.ignore_threshold] = 0
        return noobj_mask, pred_boxes


def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a


class Generator(object):
    def __init__(self,batch_size,
                 train_lines, image_size,
                 ):
        
        self.batch_size = batch_size
        self.train_lines = train_lines
        self.train_batches = len(train_lines)
        self.image_size = image_size
        
    def get_random_data(self, annotation_line, input_shape, jitter=.3, hue=.1, sat=1.5, val=1.5):
        '''r实时数据增强的随机预处理'''
        line = annotation_line.split()
        image = Image.open(line[0])
        iw, ih = image.size
        h, w = input_shape
        box = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])

        # resize image
        new_ar = w/h * rand(1-jitter,1+jitter)/rand(1-jitter,1+jitter)
        scale = rand(.5, 1.5)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)

        # place image
        dx = int(rand(0, w-nw))
        dy = int(rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_image.paste(image, (dx, dy))
        image = new_image

        # flip image or not
        flip = rand()<.5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # distort image
        hue = rand(-hue, hue)
        sat = rand(1, sat) if rand()<.5 else 1/rand(1, sat)
        val = rand(1, val) if rand()<.5 else 1/rand(1, val)
        x = cv2.cvtColor(np.array(image,np.float32)/255, cv2.COLOR_RGB2HSV)
        x[..., 0] += hue*360
        x[..., 0][x[..., 0]>1] -= 1
        x[..., 0][x[..., 0]<0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:,:, 0]>360, 0] = 360
        x[:, :, 1:][x[:, :, 1:]>1] = 1
        x[x<0] = 0
        image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB)*255

        # correct boxes
        box_data = np.zeros((len(box),5))
        if len(box)>0:
            np.random.shuffle(box)
            box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
            box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
            if flip: box[:, [0,2]] = w - box[:, [2,0]]
            box[:, 0:2][box[:, 0:2]<0] = 0
            box[:, 2][box[:, 2]>w] = w
            box[:, 3][box[:, 3]>h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box
            box_data = np.zeros((len(box),5))
            box_data[:len(box)] = box
        if len(box) == 0:
            return image_data, []

        if (box_data[:,:4]>0).any():
            return image_data, box_data
        else:
            return image_data, []

    def get_random_data_with_Mosaic(self, annotation_line, input_shape, hue=.1, sat=1.5, val=1.5):
        '''random preprocessing for real-time data augmentation'''
        h, w = input_shape
        min_offset_x = 0.4
        min_offset_y = 0.4
        scale_low = 1-min(min_offset_x,min_offset_y)
        scale_high = scale_low+0.2

        image_datas = [] 
        box_datas = []
        index = 0

        place_x = [0,0,int(w*min_offset_x),int(w*min_offset_x)]
        place_y = [0,int(h*min_offset_y),int(h*min_offset_y),0]
        for line in annotation_line:
            # 每一行进行分割
            line_content = line.split()
            # 打开图片
            image = Image.open(line_content[0])
            image = image.convert("RGB") 
            # 图片的大小
            iw, ih = image.size
            # 保存框的位置
            box = np.array([np.array(list(map(int,box.split(',')))) for box in line_content[1:]])
            
            # 是否翻转图片
            flip = rand()<.5
            if flip and len(box)>0:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                box[:, [0,2]] = iw - box[:, [2,0]]

            # 对输入进来的图片进行缩放
            new_ar = w/h
            scale = rand(scale_low, scale_high)
            if new_ar < 1:
                nh = int(scale*h)
                nw = int(nh*new_ar)
            else:
                nw = int(scale*w)
                nh = int(nw/new_ar)
            image = image.resize((nw,nh), Image.BICUBIC)

            # 进行色域变换
            hue = rand(-hue, hue)
            sat = rand(1, sat) if rand()<.5 else 1/rand(1, sat)
            val = rand(1, val) if rand()<.5 else 1/rand(1, val)
            x = cv2.cvtColor(np.array(image,np.float32)/255, cv2.COLOR_RGB2HSV)
            x[..., 0] += hue*360
            x[..., 0][x[..., 0]>1] -= 1
            x[..., 0][x[..., 0]<0] += 1
            x[..., 1] *= sat
            x[..., 2] *= val
            x[x[:,:, 0]>360, 0] = 360
            x[:, :, 1:][x[:, :, 1:]>1] = 1
            x[x<0] = 0
            image = cv2.cvtColor(x, cv2.COLOR_HSV2RGB) # numpy array, 0 to 1
            
            image = Image.fromarray((image*255).astype(np.uint8))
            # 将图片进行放置，分别对应四张分割图片的位置
            dx = place_x[index]
            dy = place_y[index]
            new_image = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image)

            
            index = index + 1
            box_data = []
            # 对box进行重新处理
            if len(box)>0:
                np.random.shuffle(box)
                box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
                box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
                box[:, 0:2][box[:, 0:2]<0] = 0
                box[:, 2][box[:, 2]>w] = w
                box[:, 3][box[:, 3]>h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w>1, box_h>1)]
                box_data = np.zeros((len(box),5))
                box_data[:len(box)] = box
            
            image_datas.append(image_data)
            box_datas.append(box_data)

        # 将图片分割，放在一起
        cutx = np.random.randint(int(w*min_offset_x), int(w*(1 - min_offset_x)))
        cuty = np.random.randint(int(h*min_offset_y), int(h*(1 - min_offset_y)))

        new_image = np.zeros([h,w,3])
        new_image[:cuty, :cutx, :] = image_datas[0][:cuty, :cutx, :]
        new_image[cuty:, :cutx, :] = image_datas[1][cuty:, :cutx, :]
        new_image[cuty:, cutx:, :] = image_datas[2][cuty:, cutx:, :]
        new_image[:cuty, cutx:, :] = image_datas[3][:cuty, cutx:, :]

        # 对框进行进一步的处理
        new_boxes = np.array(merge_bboxes(box_datas, cutx, cuty))

        if len(new_boxes) == 0:
            return new_image, []
        if (new_boxes[:,:4]>0).any():
            return new_image, new_boxes
        else:
            return new_image, []

    def generate(self, train = True, mosaic = True):
        while True:
            shuffle(self.train_lines)
            lines = self.train_lines
            inputs = []
            targets = []
            flag = True
            n = len(lines)
            for i in range(len(lines)):
                if mosaic == True:
                    if flag and (i+4) < n:
                        img,y = self.get_random_data_with_Mosaic(lines[i:i+4], self.image_size[0:2])
                        i = (i+4) % n
                    else:
                        img,y = self.get_random_data(lines[i], self.image_size[0:2])
                        i = (i+1) % n
                    flag = bool(1-flag)
                else:
                    img,y = self.get_random_data(lines[i], self.image_size[0:2])
                    i = (i+1) % n
                if len(y)!=0:
                    boxes = np.array(y[:,:4],dtype=np.float32)
                    boxes[:,0] = boxes[:,0]/self.image_size[1]
                    boxes[:,1] = boxes[:,1]/self.image_size[0]
                    boxes[:,2] = boxes[:,2]/self.image_size[1]
                    boxes[:,3] = boxes[:,3]/self.image_size[0]

                    boxes = np.maximum(np.minimum(boxes,1),0)
                    boxes[:,2] = boxes[:,2] - boxes[:,0]
                    boxes[:,3] = boxes[:,3] - boxes[:,1]
    
                    boxes[:,0] = boxes[:,0] + boxes[:,2]/2
                    boxes[:,1] = boxes[:,1] + boxes[:,3]/2
                    y = np.concatenate([boxes,y[:,-1:]],axis=-1)
                    
                img = np.array(img,dtype = np.float32)

                inputs.append(np.transpose(img/255.0,(2,0,1)))              
                targets.append(np.array(y,dtype = np.float32))
                if len(targets) == self.batch_size:
                    tmp_inp = np.array(inputs)
                    tmp_targets = targets
                    inputs = []
                    targets = []
                    yield tmp_inp, tmp_targets
