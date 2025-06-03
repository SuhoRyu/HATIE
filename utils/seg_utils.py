import torch
import numpy as np
import detectron2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

COCO_CLS = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
            'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
            'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
            'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

def get_bbox_from_mask(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    return [xmin, ymin, xmax, ymax]

def leave_max_score(scores, names):
    nameset = list(set(names))
    idx = [names.index(name) for name in nameset]
    for i, (score, name) in enumerate(zip(scores, names)):
        if score > scores[idx[nameset.index(name)]]:
            idx[nameset.index(name)] = i
    return nameset, idx

def load_detector(detector, device='cpu'):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(detector))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(detector)
    cfg.MODEL.DEVICE = device
    predictor = DefaultPredictor(cfg)
    return predictor

def get_main_objs_segments(detector, img, query_info):
    outputs = detector(img)
    score, mask, class_id = outputs["instances"].scores.cpu(), outputs["instances"].pred_masks.cpu(), outputs["instances"].pred_classes.cpu()
    bbox = [get_bbox_from_mask(mask[i].numpy()) for i in range(len(mask)) if torch.sum(mask[i]).item()>0]
    class_name = [COCO_CLS[class_id[i]] for i in range(len(class_id)) if torch.sum(mask[i]).item()>0]
    score = [score[i] for i in range(len(score)) if torch.sum(mask[i]).item()>0]
    mask = [mask[i] for i in range(len(mask)) if torch.sum(mask[i]).item()>0]
    nameset, max_idx = leave_max_score(score, class_name)

    if query_info['type'] == 'attr_chg' or query_info['type'] == 'obj_resize' or query_info['type'] == 'obj_remove': main_objs = [query_info['target'][1]]
    elif query_info['type'] == 'obj_rep': main_objs = [query_info['target'], query_info['original'][1]]
    elif query_info['type'] == 'obj_add': main_objs = [query_info['target'], query_info['anchor'][1]]
    else: main_objs = query_info['editable_objs']

    score = [score[max_idx[i]] for i in range(len(max_idx)) if nameset[i] in main_objs]
    mask = [mask[max_idx[i]] for i in range(len(max_idx)) if nameset[i] in main_objs]
    bbox = [bbox[max_idx[i]] for i in range(len(max_idx)) if nameset[i] in main_objs]
    class_name = [name for name in nameset if name in main_objs]

    return class_name, score, bbox, mask

