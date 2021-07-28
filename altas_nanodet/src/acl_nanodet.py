import sys
import os
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageOps

import acl

path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join("../../"))

from atlas_utils.acl_resource import AclResource
from atlas_utils.acl_model import Model
from atlas_utils.acl_image import AclImage
from atlas_utils.acl_dvpp import Dvpp
import atlas_utils.constants as const
import atlas_utils.utils as utils

currentPath = os.path.join(path, "..")
OUTPUT_DIR = os.path.join(currentPath, 'outputs/')
MODEL_PATH = os.path.join(currentPath, "model/nanodet_m.om")
MODEL_WIDTH = 416
MODEL_HEIGHT = 416


class Nanodet(object):
    """
    Class for portrait segmentation
    """
    def __init__(self, model_path, model_width, model_height,
                 prob_threshold=0.4, iou_threshold=0.3, strides=(8, 16, 32)):
        self._model_path = model_path
        self._model_width = model_width
        self._model_height = model_height
        self._img_width = 0
        self._img_height = 0
        self._model = None
        self._dvpp = None

        with open('../coco.names', 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')
        self.num_classes = len(self.classes)
        self.prob_threshold = prob_threshold
        self.iou_threshold = iou_threshold
        self.input_shape = (MODEL_HEIGHT, MODEL_WIDTH)
        self.strides = strides
        self.reg_max = 7
        self.project = np.arange(self.reg_max + 1)
        self.mean = np.array([103.53, 116.28, 123.675], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([57.375, 57.12, 58.395], dtype=np.float32).reshape(1, 1, 3)
        self.mlvl_anchors = []
        for i in range(len(self.strides)):
            anchors = self._make_grid((int(self.input_shape[0] / self.strides[i]), int(self.input_shape[1] / self.strides[i])), self.strides[i])
            self.mlvl_anchors.append(anchors)

    def init(self):
        """
        Initialize
        """
        self._dvpp = Dvpp()
        # Load model
        self._model = Model(self._model_path)

        return const.SUCCESS

    def padding_resize_image(self, srcimg, keep_ratio=True):
        if keep_ratio and srcimg.size[1] != srcimg.size[0]:
            border = (srcimg.size[1] - srcimg.size[0]) // 2
            if border > 0:
                img = ImageOps.expand(srcimg, border=(border, 0, border, 0), fill=0)
            else:
                img = ImageOps.expand(srcimg, border=(0, -border, 0, -border), fill=0)
        else:
            img = srcimg
        img_np = np.array(img.resize((MODEL_WIDTH, MODEL_HEIGHT)))
        return img_np

    @utils.display_time
    def pre_process(self, image):
        """
        preprocess
        """
        image_resized = self.padding_resize_image(image)
        image_norm = self._normalize(image_resized.astype(np.float32))
        image_inp = np.transpose(image_norm, (2, 0, 1))
#        image_dvpp = image.copy_to_dvpp()
#        yuv_image = self._dvpp.jpegd(image_dvpp)
#        resized_image = self._dvpp.resize(yuv_image,
#                                          self._model_width, self._model_height)
        return image_inp.copy()

    @utils.display_time
    def inference(self, input_data):
        """
        model inference
        """
        return self._model.execute(input_data)

    @utils.display_time
    def post_process(self, infer_output, image_file):
        """
        Post-processing, analysis of inference results
        """
        output_path = os.path.join(OUTPUT_DIR, os.path.basename(image_file))
        origin_img = Image.open(image_file)
        pad_img = self.padding_resize_image(origin_img)
        pad_img = Image.fromarray(pad_img.astype('uint8')).convert('RGB')
        draw = ImageDraw.Draw(pad_img)
        font = ImageFont.load_default()

        cls_scores = []
        bbox_preds = []
        for out in infer_output:
            if out.shape[-1] == self.num_classes:
                cls_scores.append(out.squeeze())
            else:

                bbox_preds.append(out.squeeze())
        cls_scores.sort(key=lambda x: x.shape[-2], reverse=True)
        bbox_preds.sort(key=lambda x: x.shape[-2], reverse=True)
        det_bboxes, det_conf, det_classid = self._get_bboxes_single(cls_scores, bbox_preds, 1, rescale=False)

        for i in range(det_bboxes.shape[0]):
            draw.rectangle(((det_bboxes[i][0], det_bboxes[i][1]), (det_bboxes[i][2], det_bboxes[i][3])),
                           fill=None, outline='green', width=3)
            draw.text((det_bboxes[i][0], det_bboxes[i][1]-10), self.classes[det_classid[i]], font=font, fill='green')

        pad_img.save(output_path)

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
        img = (img - self.mean) / self.std
        return img

    def _distance2bbox(self, points, distance, max_shape=None):
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

    def _get_bboxes_single(self, cls_scores, bbox_preds, scale_factor, rescale=False):
        mlvl_bboxes = []
        mlvl_scores = []
        for stride, cls_score, bbox_pred, anchors in zip(self.strides, cls_scores, bbox_preds, self.mlvl_anchors):
            if cls_score.ndim == 3:
                cls_score = cls_score.squeeze(axis=0)
            if bbox_pred.ndim == 3:
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

            bboxes = self._distance2bbox(anchors, bbox_pred, max_shape=self.input_shape)
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


def main():
    """
    main
    """
    image_dir = os.path.join(currentPath, "data")

    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    acl_resource = AclResource()
    acl_resource.init()

    detector = Nanodet(MODEL_PATH, MODEL_WIDTH, MODEL_HEIGHT, strides=(8, 16, 32))
    ret = detector.init()
    utils.check_ret("Classify init ", ret)

    images_list = [os.path.join(image_dir, img)
                   for img in os.listdir(image_dir)
                   if os.path.splitext(img)[1] in const.IMG_EXT]

    for image_file in images_list:
        # Read image
        print('=== ' + os.path.basename(image_file) + '===')
        image = Image.open(image_file)
#        image_acl = AclImage(image_padding)
        # preprocess the picture
        image_inp = detector.pre_process(image)
        # Inference
        result = detector.inference([image_inp, ])
        # Post-processing
        detector.post_process(result, image_file)


if __name__ == '__main__':
    main()

