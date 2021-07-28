import cv2
from PIL import Image, ImageDraw, ImageFont, ImageOps
import numpy as np
import argparse
import sys
import os
import functools
import time

path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join("../../"))


currentPath = os.path.join(path, "..")
OUTPUT_DIR = os.path.join(currentPath, 'outputs/')
MODEL_PATH = os.path.join(currentPath, "model/nanodet_m.onnx")
IMAGE_SIZE = (416, 416)
CONF_TH = 0.3
NMS_TH = 0.45
CLASSES = 80
clsnames = [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
             'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
             'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
             'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
             'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
             'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
             'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
             'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
             'hair drier', 'toothbrush']


def display_process_time(func):
    @functools.wraps(func)
    def decorated(*args, **kwargs):
        s1 = time.time()
        res = func(*args, **kwargs)
        s2 = time.time()
        print('%s process time %f ms' % (func.__name__, 1000*(s2-s1)))
        return res

    return decorated


def plot_one_box(x, img, color=None, label=None):
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, 2)
    # cv2.rectangle(img, (int(x[0]), int(x[1]) - 15), (int(x[0]) + 100, int(x[1]) + 2), (255, 128, 128), -1)
    cv2.putText(img, label, (int(x[0]), int(x[1]) - 8), cv2.FONT_ITALIC, 0.8, (0, 255, 0), thickness=2,
                lineType=cv2.LINE_AA)


def draw_dets(img, dets, dst):
    for x1, y1, x2, y2, conf, cls in dets:
        label = clsnames[int(cls)]
        plot_one_box(x=[x1, y1, x2, y2], img=img, label=label, color=[0, 0, 255])
    cv2.imencode('.jpg', img)[1].tofile(dst)


class my_nanodet():
    def __init__(self, model_path=MODEL_PATH, img_size=IMAGE_SIZE,
                 conf_thres=CONF_TH, nms_thres=NMS_TH):
        self.model_path = model_path
        self.strides = (8, 16, 32)
        self.input_shape = img_size
        self.reg_max = 7
        self.prob_threshold = conf_thres
        self.iou_threshold = nms_thres
        self.project = np.arange(self.reg_max + 1)
        self.mean = np.array([103.53, 116.28, 123.675], dtype=np.float32).reshape([1, 1, -1])
        self.std = np.array([57.375, 57.12, 58.395], dtype=np.float32).reshape([1, 1, -1])

        self.mlvl_anchors = []
        self.model = None

        self._init()

    def _init(self):
        self.model = cv2.dnn.readNet(self.model_path)
        for i in range(len(self.strides)):
            anchors = self._make_grid((self.input_shape[0] // self.strides[i],
                                       self.input_shape[1] // self.strides[i]),
                                      self.strides[i])
            self.mlvl_anchors.append(anchors)

    def _make_grid(self, featmap_size, stride):
        feat_h, feat_w = featmap_size
        shift_x = np.arange(0, feat_w) * stride
        shift_y = np.arange(0, feat_h) * stride
        xv, yv = np.meshgrid(shift_x, shift_y)
        xv = xv.flatten()
        yv = yv.flatten()
        cx = xv + 0.5 * (stride-1)
        cy = yv + 0.5 * (stride - 1)
        return np.stack((cx, cy), axis=-1)

    def _softmax(self, x, axis=1):
        x_exp = np.exp(x)
        # 如果是列向量，则axis=0
        x_sum = np.sum(x_exp, axis=axis, keepdims=True)
        s = x_exp / x_sum
        return s

    def _normalize(self, img):
        img = img.astype(np.float32)
        img = (img - self.mean) / self.std
        return img

    def padding_resize_image(self, srcimg, keep_ratio=True):
        if keep_ratio and srcimg.size[1] != srcimg.size[0]:
            border = (srcimg.size[1] - srcimg.size[0]) // 2
            if border > 0:
                img = ImageOps.expand(srcimg, border=(border, 0, border, 0), fill=0)
            else:
                img = ImageOps.expand(srcimg, border=(0, -border, 0, -border), fill=0)
        else:
            img = srcimg
        img_np = np.array(img.resize(self.input_shape))
        return img_np

    @display_process_time
    def pre_process(self, image):
        image_resized = self.padding_resize_image(image)
        image_norm = self._normalize(image_resized.astype(np.float32))
        image_inp = np.expand_dims(np.transpose(image_norm, (2, 0, 1)), axis=0)
        return image_inp, image_resized[:, :, ::-1].astype(np.uint8)

    @display_process_time
    def inference(self, inp):
        self.model.setInput(inp)
        prediction = self.model.forward(self.model.getUnconnectedOutLayersNames())
        return prediction

    @display_process_time
    def post_process(self, prediction):
        cls_scores, bbox_preds = prediction[::2], prediction[1::2]
        det_bboxes, det_conf, det_classid = self.get_bboxes_single(cls_scores, bbox_preds, 1, rescale=False)
        return det_bboxes, det_conf, det_classid

    def detect(self, image):
        img_inp, img_resized = self.pre_process(image)
        prediction = self.inference(img_inp)
        det_bboxes, det_conf, det_cls = self.post_process(prediction)
        if not len(det_bboxes.shape) == len(det_conf.shape):
            det_conf, det_cls = np.expand_dims(det_conf, axis=-1), np.expand_dims(det_cls, axis=-1)
        dets = np.concatenate([det_bboxes, det_conf, det_cls], axis=-1)
        return dets, img_resized

    def get_bboxes_single(self, cls_scores, bbox_preds, scale_factor, rescale=False):
        mlvl_bboxes = []
        mlvl_scores = []
        for stride, cls_score, bbox_pred, anchors in zip(self.strides, cls_scores, bbox_preds, self.mlvl_anchors):
            if cls_score.ndim==3:
                cls_score = cls_score.squeeze(axis=0)
            if bbox_pred.ndim==3:
                bbox_pred = bbox_pred.squeeze(axis=0)
            bbox_pred = self._softmax(bbox_pred.reshape(-1, self.reg_max + 1), axis=1)
            # bbox_pred = np.sum(bbox_pred * np.expand_dims(self.project, axis=0), axis=1).reshape((-1, 4))
            bbox_pred = np.dot(bbox_pred, self.project).reshape(-1,4)
            bbox_pred *= stride

            # nms_pre = cfg.get('nms_pre', -1)
            nms_pre = 1000
            if nms_pre > 0 and cls_score.shape[0] > nms_pre:
                max_scores = cls_score.max(axis=1)
                topk_inds = max_scores.argsort()[::-1][0:nms_pre]
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                cls_score = cls_score[topk_inds, :]

            bboxes = self.distance2bbox(anchors, bbox_pred, max_shape=self.input_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(cls_score)

        mlvl_bboxes = np.concatenate(mlvl_bboxes, axis=0)
        if rescale:
            mlvl_bboxes /= scale_factor
        mlvl_scores = np.concatenate(mlvl_scores, axis=0)

        bboxes_wh = mlvl_bboxes.copy()
        bboxes_wh[:, 2:4] = bboxes_wh[:, 2:4] - bboxes_wh[:, 0:2]  ####xywh
        classIds = np.argmax(mlvl_scores, axis=1)
        confidences = np.max(mlvl_scores, axis=1)  ####max_class_confidence

        indices = cv2.dnn.NMSBoxes(bboxes_wh.tolist(), confidences.tolist(), self.prob_threshold, self.iou_threshold)
        if len(indices)>0:
            mlvl_bboxes = mlvl_bboxes[indices[:, 0]].astype(np.int32)
            confidences = confidences[indices[:, 0]]
            classIds = classIds[indices[:, 0]]
            return mlvl_bboxes, confidences, classIds
        else:
            print('nothing detect')
            return np.array([]), np.array([]), np.array([])

    def distance2bbox(self, points, distance, max_shape=None):
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]
        if max_shape is not None:
            x1 = np.clip(x1, 0, max_shape[1])
            y1 = np.clip(y1, 0, max_shape[0])
            x2 = np.clip(x2, 0, max_shape[1])
            y2 = np.clip(y2, 0, max_shape[0])
        return np.stack([x1, y1, x2, y2], axis=-1)


if __name__ == '__main__':
    detector = my_nanodet(model_path=MODEL_PATH)
    image_dir = os.path.join(currentPath, "data")
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
    images_list = [os.path.join(image_dir, img)
                   for img in os.listdir(image_dir)
                   if os.path.splitext(img)[1] in ['.jpg', '.png', '.bmp']]

    for image_file in images_list:
        # Read image
        print('=== ' + os.path.basename(image_file) + '===')
        output_path = os.path.join(OUTPUT_DIR, os.path.basename(image_file))

        img = Image.open(image_file)
        dets, img_resized = detector.detect(img)
        draw_dets(img_resized, dets, output_path)




