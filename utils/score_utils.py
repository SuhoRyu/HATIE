import clip
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from transformers import AutoImageProcessor, AutoModel
import lpips
import numpy as np
import cv2
from torchvision.transforms import ToTensor
totensor = ToTensor()
from . import t2v_metrics

clip_models = clip.available_models()
IMAGE_SIZE = 512

#build models
def build_clip(model, device):
    if model in clip_models:
        clip_model, clip_preprocess = clip.load(model, device=device)
    else:
        raise ValueError(f"Model {model} not found in available models.")
    clip_model.eval()
    clip_preprocess = transforms.Compose([
        clip_preprocess.transforms[0],   # Resize
        clip_preprocess.transforms[-1]   # Normalize
    ])  
    clip_csim = nn.CosineSimilarity(dim=1, eps=1e-08).to(device)

    return clip_model, clip_preprocess, clip_csim

def build_dino(model, device):
    _dino_preprocess = AutoImageProcessor.from_pretrained(model)
    _dino_preprocess = transforms.Compose([
        transforms.Resize((_dino_preprocess.crop_size['height'], _dino_preprocess.crop_size['width'])),
        transforms.Normalize(mean=_dino_preprocess.image_mean, std=_dino_preprocess.image_std)
    ])
    dino_preprocess = lambda x: {'pixel_values': _dino_preprocess(x).unsqueeze(0)}
    
    dino_model = AutoModel.from_pretrained(model).to(device)
    dino_csim = nn.CosineSimilarity(dim=0).to(device)

    return dino_model, dino_preprocess, dino_csim

def build_lpips(model, device):
    return lpips.LPIPS(net=model).to(device)

def build_vqa(model, device):
    return t2v_metrics.VQAScore(model=model, device=device)






#metric functions
def clip_imgtext_score(image, text, _clip):
    model, preprocess, csim = _clip
    img = preprocess(image).unsqueeze(0)
    text = clip.tokenize([text]).to(img.device)

    image_features =model.encode_image(img)
    text_features = model.encode_text(text)

    score = (1+csim(image_features, text_features))/2

    return score.item()

def clip_imgimg_score(image1, image2, _clip):
    model, preprocess, csim = _clip
    img1 = preprocess(image1).unsqueeze(0)
    img2 = preprocess(image2).unsqueeze(0)

    image1_features = model.encode_image(img1)
    image2_features = model.encode_image(img2)

    score = (1+csim(image1_features, image2_features))/2

    return score.item()

def vqa_score(image, text, _vqa):
    return _vqa(images=[image], texts=[text])[0][0].item()

def lpips_score(image1, image2, _lpips):
    img1 = nn.functional.interpolate(image1.unsqueeze(0), size = (64,64), mode = 'bilinear').squeeze(0)
    img2 = nn.functional.interpolate(image2.unsqueeze(0), size = (64,64), mode = 'bilinear').squeeze(0)
    d = _lpips(img1.unsqueeze(0), img2.unsqueeze(0), normalize=True)

    return 1 - d.item()

def DINO_score(image1, image2, _dino):
    model, preprocess, csim = _dino
    inputs1 = preprocess(image1)
    outputs1 = model(**inputs1)
    image_features1 = outputs1.last_hidden_state
    image_features1 = image_features1.mean(dim=1)

    inputs2 = preprocess(image2)
    outputs2 = model(**inputs2)
    image_features2 = outputs2.last_hidden_state
    image_features2 = image_features2.mean(dim=1)

    sim = csim(image_features1[0],image_features2[0]).item()
    sim = (sim+1)/2

    return sim

def L2_score(image1, image2):
    img2 = nn.functional.interpolate(image2.unsqueeze(0), size = (image1.shape[1],image1.shape[2]), mode = 'bilinear').squeeze(0)
    l2max = nn.MSELoss()(torch.zeros_like(image1), torch.ones_like(image1)).item()
    return 1 - nn.MSELoss()(image1, img2).item() / l2max

def degradation_test_score_A(image1, image2, threshold, _clip, _lpips, _dino):
    score = [0,0,0,0]
    img1 = image1.mean(dim=0, keepdim=True)
    img1 = img1.repeat(3,1,1)
    img2 = image2.mean(dim=0, keepdim=True)
    img2 = img2.repeat(3,1,1)

    for power in range(0, 10):
        size = int(IMAGE_SIZE / (2**power))
        if size < 2: break
        img1 = nn.functional.interpolate(img1.unsqueeze(0), size = (size,size), mode = 'bilinear').squeeze(0)
        img2 = nn.functional.interpolate(img2.unsqueeze(0), size = (size,size), mode = 'bilinear').squeeze(0)
        if clip_imgimg_score(img1, img2, _clip) > threshold and score[0] == 0: score[0] = (10-power)/10
        if lpips_score(img1, img2, _lpips) > threshold and score[1] == 0: score[1] = (10-power)/10
        if DINO_score(img1, img2, _dino) > threshold and score[2] == 0: score[2] = (10-power)/10
        if L2_score(img1, img2) > threshold and score[3] == 0: score[3] = (10-power)/10
        if all(score): break
    
    return score

def degradation_test_score_B(image1, image2, level, _clip, _lpips, _dino):
    img1 = image1.mean(dim=0, keepdim=True)
    img1 = img1.repeat(3,1,1)

    img2 = image2.mean(dim=0, keepdim=True)
    img2 = img2.repeat(3,1,1)

    img1 = nn.functional.interpolate(img1.unsqueeze(0), size = (max(4, int(img1.shape[1]/level)),max(4, int(img1.shape[2]/level))), mode = 'bilinear').squeeze(0)
    img2 = nn.functional.interpolate(img2.unsqueeze(0), size = (max(4, int(img2.shape[1]/level)),max(4, int(img2.shape[2]/level))), mode = 'bilinear').squeeze(0)
    
    return [clip_imgimg_score(img1,img2,_clip), lpips_score(img1,img2,_lpips), DINO_score(img1,img2,_dino), L2_score(img1, img2)]

def size_score(mask_bf, mask_af, query_type, threshold=None):
    size_change_rate = (torch.sum(mask_af)/torch.sum(mask_bf))**0.5

    score = 0.0
    if query_type == "enlarge":
        if size_change_rate > threshold[0]: score = 1.0
        elif size_change_rate < 1.0: score = 0.0
        else: score = (size_change_rate - 1.0)/(threshold[0] - 1.0)
            
    elif query_type == "shrink":
        if size_change_rate < threshold[1]: score = 1.0
        elif size_change_rate > 1.0: score = 0.0
        else: score = (1.0 - size_change_rate)/(1.0 - threshold[1])

    elif query_type == "keep":
        if size_change_rate == 1.0: score = 1.0
        elif size_change_rate > 1.0: 
            max_ratio = 1.0/0.003 
            score = 1.0 - (size_change_rate - 1.0)/(max_ratio - 1.0)
        else:
            score = size_change_rate

    return score if type(score) == float else score.item()

def pos_score(mask_bf, mask_af):

    com_x_bf, com_y_bf = com(mask_bf)
    com_x_af, com_y_af = com(mask_af)

    relative_distance = ((com_x_bf - com_x_af)**2 + (com_y_bf - com_y_af)**2)**0.5 / torch.sum(mask_bf)**0.5
    max_distance = np.sqrt(2)*IMAGE_SIZE/(IMAGE_SIZE*np.sqrt(0.003)) #dataset filtering threshold
    score = ((max_distance - relative_distance) / max_distance)

    return score if type(score) == float else score.item()

def edge_score(image1, image2, _clip, _lpips, _dino):
    img1 = image1.mul(255).permute(1, 2, 0).byte().cpu().numpy()
    img2 = image2.mul(255).permute(1, 2, 0).byte().cpu().numpy()

    img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    edge1 = cv2.Canny(img1, 100, 400)
    edge2 = cv2.Canny(img2, 100, 400)

    edge1 = totensor(edge1).to(image1.device)
    edge2 = totensor(edge2).to(image2.device)

    edge1 = edge1.repeat(3,1,1)
    edge2 = edge2.repeat(3,1,1)

    return [clip_imgimg_score(edge1, edge2, _clip), lpips_score(edge1, edge2, _lpips), DINO_score(edge1, edge2, _dino), L2_score(edge1, edge2)]

def com(mask):
    area = torch.sum(mask).item()
    x,y = np.meshgrid(np.arange(IMAGE_SIZE, dtype=float),np.arange(IMAGE_SIZE, dtype=float))
    x *= mask.cpu().numpy()
    y *= mask.cpu().numpy()
    com_x = np.sum(x)/area
    com_y = np.sum(y)/area

    return com_x, com_y

def combine_bbox(box1, box2):
    x1 = min(box1[0], box2[0])
    y1 = min(box1[1], box2[1])
    x2 = max(box1[2], box2[2])
    y2 = max(box1[3], box2[3])

    return [x1, y1, x2, y2]

def square_pad(image):
    h, w = image.shape[1], image.shape[2]
    if h > w:
        pad = (h - w) // 2
        rem = (h - w) % 2
        image = nn.functional.pad(image, (pad, pad+rem))
    elif w > h:
        pad = (w - h) // 2
        rem = (w - h) % 2
        image = nn.functional.pad(image, (0, 0, pad, pad+rem))

    return image









#evaluation functions
def eval_objAdd(query, result, original_img, edited_img, _clip, _lpips, _dino, _vqa):
    device = original_img.device
    anchor_name = query['anchor'][1]
    target_name = query['target']
    location = query['location']
    if location == 'left' or location == 'right': location = f'on the {location} side of'
    elif location == 'front': location = f'in {location} of'

    if anchor_name in result['class_names']:
        anchor_idx = result['class_names'].index(anchor_name)
        edit_anchor_bbox = result['bboxs'][anchor_idx]
    else: edit_anchor_bbox = [0,0,IMAGE_SIZE,IMAGE_SIZE]

    if target_name in result['class_names']:
        target_idx = result['class_names'].index(target_name)
        target_bbox = result['bboxs'][target_idx]
        target_mask = result['masks'][target_idx].to(device).float()

        if target_bbox[2] - target_bbox[0] < 5 or target_bbox[3] - target_bbox[1] < 5:
            cilp_bg_cst = clip_imgimg_score(edited_img, original_img, _clip)
            edit_background = edited_img * (1 - target_mask).unsqueeze(0)
            original_background = original_img * (1 - target_mask).unsqueeze(0)
            lpips_bg_cst = lpips_score(edit_background, original_background, _lpips)
            dino_bg_cst = DINO_score(edit_background, original_background, _dino)
            l2_bg_cst = L2_score(edit_background, original_background)

            clip_obj_fid = 0
            seg_obj_fid = 0
            vqa_obj_fid = 0

            clip_pos_fid = 0
            vqa_pos_fid = 0
        
        else:
            at_merged_bbox = combine_bbox(edit_anchor_bbox, target_bbox)
            target_img = (edited_img * target_mask.unsqueeze(0))[:, target_bbox[1]:target_bbox[3], target_bbox[0]:target_bbox[2]]
            target_img = square_pad(target_img)
            edit_background = edited_img * (1 - target_mask).unsqueeze(0)
            original_background = original_img * (1 - target_mask).unsqueeze(0)
            anchorNtarget_img = edited_img[:, at_merged_bbox[1]:at_merged_bbox[3], at_merged_bbox[0]:at_merged_bbox[2]]
            anchorNtarget_img = square_pad(anchorNtarget_img) 

            cilp_bg_cst = clip_imgimg_score(edit_background, original_background, _clip)
            lpips_bg_cst = lpips_score(edit_background, original_background, _lpips)
            dino_bg_cst = DINO_score(edit_background, original_background, _dino)
            l2_bg_cst = L2_score(edit_background, original_background)
            clip_obj_fid = clip_imgtext_score(target_img, target_name, _clip)
            seg_obj_fid = result['scores'][target_idx].item()
            vqa_obj_fid = vqa_score(target_img, target_name, _vqa)
            clip_pos_fid = clip_imgtext_score(anchorNtarget_img, f'{target_name} {location} {target_name}', _clip)
            vqa_pos_fid = vqa_score(anchorNtarget_img, f'{target_name} {location} {target_name}', _vqa)

    else:
        cilp_bg_cst = clip_imgimg_score(edited_img, original_img, _clip)
        lpips_bg_cst = lpips_score(edited_img, original_img, _lpips)
        dino_bg_cst = DINO_score(edited_img, original_img, _dino)
        l2_bg_cst = L2_score(edited_img, original_img)

        clip_obj_fid = 0
        seg_obj_fid = 0
        vqa_obj_fid = 0

        clip_pos_fid = 0
        vqa_pos_fid = 0

    return [cilp_bg_cst, lpips_bg_cst, dino_bg_cst, l2_bg_cst], [clip_obj_fid, seg_obj_fid, vqa_obj_fid, clip_pos_fid, vqa_pos_fid]

def eval_objRep(query, ori_seg, result, original_img, edited_img, _clip, _lpips, _dino, _vqa):
    device = original_img.device
    original_id, _ = query['original']
    target_name = query['target']

    original_mask = torch.tensor(ori_seg[original_id]['mask']).to(device).float()
    
    if target_name in result['class_names']:
        target_idx = result['class_names'].index(target_name)
        target_bbox = result['bboxs'][target_idx]
        target_mask = result['masks'][target_idx].to(device).float()

        if target_bbox[2] - target_bbox[0] < 5 or target_bbox[3] - target_bbox[1] < 5:
            edit_background = edited_img * (1 - target_mask).unsqueeze(0) * (1 - original_mask).unsqueeze(0)
            original_background = original_img * (1 - target_mask).unsqueeze(0) * (1 - original_mask).unsqueeze(0)

            cilp_bg_cst = clip_imgimg_score(edit_background, original_background, _clip)
            lpips_bg_cst = lpips_score(edit_background, original_background, _lpips)
            dino_bg_cst = DINO_score(edit_background, original_background, _dino)
            l2_bg_cst = L2_score(edit_background, original_background)

            pos_obj_cst = 0

            clip_obj_fid = 0
            seg_obj_fid = 0
            vqa_obj_fid = 0

        else:
            target_img = (edited_img * target_mask.unsqueeze(0))[:, target_bbox[1]:target_bbox[3], target_bbox[0]:target_bbox[2]]
            target_img = square_pad(target_img)

            edit_background = edited_img * (1 - target_mask).unsqueeze(0) * (1 - original_mask).unsqueeze(0)
            original_background = original_img * (1 - target_mask).unsqueeze(0) * (1 - original_mask).unsqueeze(0)

            cilp_bg_cst = clip_imgimg_score(edit_background, original_background, _clip)
            lpips_bg_cst = lpips_score(edit_background, original_background, _lpips)
            dino_bg_cst = DINO_score(edit_background, original_background, _dino)
            l2_bg_cst = L2_score(edit_background, original_background)

            pos_obj_cst = pos_score(target_mask, original_mask)

            clip_obj_fid = clip_imgtext_score(target_img, target_name, _clip)
            seg_obj_fid = result['scores'][target_idx].item()
            vqa_obj_fid = vqa_score(target_img, target_name, _vqa)

    else:
        edit_background = edited_img * (1 - original_mask).unsqueeze(0)
        original_background = original_img * (1 - original_mask).unsqueeze(0)

        cilp_bg_cst = clip_imgimg_score(edit_background, original_background, _clip)
        lpips_bg_cst = lpips_score(edit_background, original_background, _lpips)
        dino_bg_cst = DINO_score(edit_background, original_background, _dino)
        l2_bg_cst = L2_score(edit_background, original_background)

        pos_obj_cst = 0

        clip_obj_fid = 0
        seg_obj_fid = 0
        vqa_obj_fid = 0
    
    return [cilp_bg_cst, lpips_bg_cst, dino_bg_cst, l2_bg_cst],[None, None, None, None, pos_obj_cst, None], [clip_obj_fid, seg_obj_fid, vqa_obj_fid, None, None]

def eval_objResize(query, ori_seg, result, original_img, edited_img, threshold, _clip, _lpips, _dino):
    device = original_img.device
    target_id, target_name = query['target']
    subtype = query['subtype']

    original_target_mask = torch.tensor(ori_seg[target_id]['mask']).to(device).float()
    original_target_bbox = torch.tensor(ori_seg[target_id]['bbox'])
    original_target_img = (original_img * original_target_mask.unsqueeze(0))[:,original_target_bbox[1]:original_target_bbox[3], original_target_bbox[0]:original_target_bbox[2]]
    original_target_img = square_pad(original_target_img)

    if target_name in result['class_names']:
        target_idx = result['class_names'].index(target_name)
        edit_target_bbox = result['bboxs'][target_idx]
        edit_target_mask = result['masks'][target_idx].to(device).float()

        if edit_target_bbox[2] - edit_target_bbox[0] < 5 or edit_target_bbox[3] - edit_target_bbox[1] < 5:
            edit_background = edited_img * (1 - edit_target_mask).unsqueeze(0) * (1 - original_target_mask).unsqueeze(0)
            original_background = original_img * (1 - edit_target_mask).unsqueeze(0) * (1 - original_target_mask).unsqueeze(0)

            cilp_bg_cst = clip_imgimg_score(edit_background, original_background, _clip)
            lpips_bg_cst = lpips_score(edit_background, original_background, _lpips)
            dino_bg_cst = DINO_score(edit_background, original_background, _dino)
            l2_bg_cst = L2_score(edit_background, original_background)

            clip_obj_cst = 0
            lpips_obj_cst = 0
            dino_obj_cst = 0
            l2_obj_cst = 0
            pos_obj_cst = 0

            obj_fid = 0
        
        else:
            edit_target_img = (edited_img * edit_target_mask.unsqueeze(0))[:,edit_target_bbox[1]:edit_target_bbox[3], edit_target_bbox[0]:edit_target_bbox[2]]
            edit_target_img = square_pad(edit_target_img)

            edit_background = edited_img * (1 - edit_target_mask).unsqueeze(0) * (1 - original_target_mask).unsqueeze(0)
            original_background = original_img * (1 - edit_target_mask).unsqueeze(0) * (1 - original_target_mask).unsqueeze(0)

            cilp_bg_cst = clip_imgimg_score(edit_background, original_background, _clip)
            lpips_bg_cst = lpips_score(edit_background, original_background, _lpips)
            dino_bg_cst = DINO_score(edit_background, original_background, _dino)
            l2_bg_cst = L2_score(edit_background, original_background)

            clip_obj_cst = clip_imgimg_score(edit_target_img, original_target_img, _clip)
            lpips_obj_cst = lpips_score(edit_target_img, original_target_img, _lpips)
            dino_obj_cst = DINO_score(edit_target_img, original_target_img, _dino)
            l2_obj_cst = L2_score(edit_target_img, original_target_img)
            pos_obj_cst = pos_score(edit_target_mask, original_target_mask)

            obj_fid = size_score(original_target_mask, edit_target_mask, subtype, threshold)

    else:
        edit_background = edited_img * (1 - original_target_mask).unsqueeze(0)
        original_background = original_img * (1 - original_target_mask).unsqueeze(0)

        cilp_bg_cst = clip_imgimg_score(edit_background, original_background, _clip)
        lpips_bg_cst = lpips_score(edit_background, original_background, _lpips)
        dino_bg_cst = DINO_score(edit_background, original_background, _dino)
        l2_bg_cst = L2_score(edit_background, original_background)

        clip_obj_cst = 0
        lpips_obj_cst = 0
        dino_obj_cst = 0
        l2_obj_cst = 0
        pos_obj_cst = 0

        obj_fid = 0

    return [cilp_bg_cst, lpips_bg_cst, dino_bg_cst, l2_bg_cst], [clip_obj_cst, lpips_obj_cst, dino_obj_cst, l2_obj_cst, pos_obj_cst, None], [obj_fid, None, None, None, None]

def eval_attrChg(query, ori_seg, result, original_img, edited_img, threshold, level, _clip, _lpips, _dino, _vqa):
    device = original_img.device
    target_id, target_name = query['target']
    after = query['after']

    original_target_mask = torch.tensor(ori_seg[target_id]['mask']).to(device).float()
    original_target_bbox = torch.tensor(ori_seg[target_id]['bbox'])
    original_target_img = (original_img * original_target_mask.unsqueeze(0))[:,original_target_bbox[1]:original_target_bbox[3], original_target_bbox[0]:original_target_bbox[2]]
    original_target_img = square_pad(original_target_img)

    if target_name in result['class_names']:
        target_idx = result['class_names'].index(target_name)
        edit_target_bbox = result['bboxs'][target_idx]
        edit_target_mask = result['masks'][target_idx].to(device).float()

        if edit_target_bbox[2] - edit_target_bbox[0] < 5 or edit_target_bbox[3] - edit_target_bbox[1] < 5:
            edit_background = edited_img * (1 - edit_target_mask).unsqueeze(0) * (1 - original_target_mask).unsqueeze(0)
            original_background = original_img * (1 - edit_target_mask).unsqueeze(0) * (1 - original_target_mask).unsqueeze(0)

            cilp_bg_cst = clip_imgimg_score(edit_background, original_background, _clip)
            lpips_bg_cst = lpips_score(edit_background, original_background, _lpips)
            dino_bg_cst = DINO_score(edit_background, original_background, _dino)
            l2_bg_cst = L2_score(edit_background, original_background)

            obj_cst = [[0,0,0],[0,0,0],[0,0,0],[0,0,0],0,0]

            clip_obj_fid = 0
            seg_obj_fid = 0
            vqa_obj_fid = 0
            
        else:
            edit_target_img = (edited_img * edit_target_mask.unsqueeze(0))[:,edit_target_bbox[1]:edit_target_bbox[3], edit_target_bbox[0]:edit_target_bbox[2]]
            edit_target_img = square_pad(edit_target_img)

            edit_background = edited_img * (1 - edit_target_mask).unsqueeze(0) * (1 - original_target_mask).unsqueeze(0)
            original_background = original_img * (1 - edit_target_mask).unsqueeze(0) * (1 - original_target_mask).unsqueeze(0)

            cilp_bg_cst = clip_imgimg_score(edit_background, original_background, _clip)
            lpips_bg_cst = lpips_score(edit_background, original_background, _lpips)
            dino_bg_cst = DINO_score(edit_background, original_background, _dino)
            l2_bg_cst = L2_score(edit_background, original_background)

            obj_cst = list(zip(edge_score(edit_target_img, original_target_img, _clip, _lpips, _dino),
                        degradation_test_score_A(edit_target_img, original_target_img, threshold, _clip, _lpips, _dino),
                        degradation_test_score_B(edit_target_img, original_target_img, level, _clip, _lpips, _dino)))
            pos_obj_cst = pos_score(edit_target_mask, original_target_mask)
            size_obj_cst = size_score(original_target_mask, edit_target_mask, 'keep')
            obj_cst.extend([pos_obj_cst, size_obj_cst])

            clip_obj_fid = clip_imgtext_score(edit_target_img, f'{after} {target_name}', _clip)
            seg_obj_fid = result['scores'][target_idx].item() 
            vqa_obj_fid = vqa_score(edit_target_img, f'{after} {target_name}', _vqa)

    else:
        edit_background = edited_img * (1 - original_target_mask).unsqueeze(0)
        original_background = original_img * (1 - original_target_mask).unsqueeze(0)

        cilp_bg_cst = clip_imgimg_score(edit_background, original_background, _clip)
        lpips_bg_cst = lpips_score(edit_background, original_background, _lpips)
        dino_bg_cst = DINO_score(edit_background, original_background, _dino)
        l2_bg_cst = L2_score(edit_background, original_background)

        obj_cst = [[0,0,0],[0,0,0],[0,0,0],[0,0,0],0,0]

        clip_obj_fid = 0
        seg_obj_fid = 0
        vqa_obj_fid = 0
    
    return [cilp_bg_cst, lpips_bg_cst, dino_bg_cst, l2_bg_cst], obj_cst, [clip_obj_fid, seg_obj_fid, vqa_obj_fid, None, None]

def eval_bgChg(query, ori_seg, result, original_img, edited_img, _clip, _lpips, _dino, _vqa):
    device = original_img.device
    obj_names = query['editable_objs']
    target = query['target']

    original_obj_masks = []
    original_obj_imgs = []
    for key in ori_seg.keys():
        original_obj_mask = torch.tensor(ori_seg[key]['mask']).to(device).float()
        original_obj_bbox = torch.tensor(ori_seg[key]['bbox'])
        original_obj_img = (original_img * original_obj_mask.unsqueeze(0))[:,original_obj_bbox[1]:original_obj_bbox[3], original_obj_bbox[0]:original_obj_bbox[2]]
        original_obj_img = square_pad(original_obj_img)
        original_obj_masks.append(original_obj_mask)
        original_obj_imgs.append(original_obj_img)
        
    edit_obj_masks = []
    edit_obj_imgs = []
    
    for name in obj_names:
        if name in result['class_names']:
            obj_idx = result['class_names'].index(name)
            edit_obj_mask = result['masks'][obj_idx].to(device).float()
            edit_obj_bbox = result['bboxs'][obj_idx]

            if edit_obj_bbox[2] - edit_obj_bbox[0] < 5 or edit_obj_bbox[3] - edit_obj_bbox[1] < 5:
                edit_obj_masks.append(torch.zeros_like(original_obj_masks[0]).to(device).float())
                edit_obj_imgs.append(torch.zeros_like(original_obj_imgs[0]))
            else:
                edit_obj_img = (edited_img * edit_obj_mask.unsqueeze(0))[:,edit_obj_bbox[1]:edit_obj_bbox[3], edit_obj_bbox[0]:edit_obj_bbox[2]]
                edit_obj_img = square_pad(edit_obj_img)
                edit_obj_masks.append(edit_obj_mask)
                edit_obj_imgs.append(edit_obj_img)

        else:
            edit_obj_masks.append(torch.zeros_like(original_obj_masks[0]))
            edit_obj_imgs.append(torch.zeros_like(original_obj_imgs[0]))

    edit_background = edited_img
    for mask in edit_obj_masks: 
        edit_background = edit_background * (1 - mask).unsqueeze(0)

    clip_obj_csts = [clip_imgimg_score(edit_img, ori_img, _clip) if torch.sum(edit_img)>0 else 0 for edit_img, ori_img in zip(edit_obj_imgs, original_obj_imgs)]
    lpips_obj_csts = [lpips_score(edit_img, ori_img, _lpips) if torch.sum(edit_img)>0 else 0 for edit_img, ori_img in zip(edit_obj_imgs, original_obj_imgs)]
    DINO_obj_csts = [DINO_score(edit_img, ori_img, _dino) if torch.sum(edit_img)>0 else 0 for edit_img, ori_img in zip(edit_obj_imgs, original_obj_imgs)]
    L2_obj_csts = [L2_score(edit_img, ori_img) if torch.sum(edit_img)>0 else 0 for edit_img, ori_img in zip(edit_obj_imgs, original_obj_imgs)]
    pos_obj_csts = [pos_score(edit_mask, ori_mask) if torch.sum(edit_mask)>0 else 0 for edit_mask, ori_mask in zip(edit_obj_masks, original_obj_masks)]
    size_obj_csts = [size_score(ori_mask, edit_mask, 'keep') if torch.sum(edit_mask)>0 else 0 for ori_mask, edit_mask in zip(original_obj_masks, edit_obj_masks)]

    clip_bg_fid = clip_imgtext_score(edit_background, target, _clip)
    vqa_bg_fid = vqa_score(edit_background, target, _vqa)

    return [clip_obj_csts, lpips_obj_csts, DINO_obj_csts, L2_obj_csts, pos_obj_csts, size_obj_csts], [clip_bg_fid, vqa_bg_fid]
    
def eval_styleChg(query, original_img, edited_img, threshold, level, _clip, _lpips, _dino, _vqa):
    target = query['target']
    
    csts = list(zip(edge_score(edited_img, original_img, _clip, _lpips, _dino),
                    degradation_test_score_A(edited_img, original_img, threshold, _clip, _lpips, _dino),
                    degradation_test_score_B(edited_img, original_img, level, _clip, _lpips, _dino)))
    
    clip_fid = clip_imgtext_score(edited_img, target, _clip)
    vqa_fid = vqa_score(edited_img, target, _vqa)

    return csts, [clip_fid, vqa_fid]

def eval_objRM(query, ori_seg, result, original_img, edited_img, _clip, _lpips, _dino, _vqa):
    device = original_img.device
    target_id = query['target'][0]
    target_name = query['target'][1]
    original_mask = torch.tensor(ori_seg[target_id]['mask']).to(device).float()

    if target_name in result['class_names']:
        target_idx = result['class_names'].index(target_name)
        target_bbox = result['bboxs'][target_idx]
        target_mask = result['masks'][target_idx].to(device).float()

        if target_bbox[2] - target_bbox[0] < 5 or target_bbox[3] - target_bbox[1] < 5:
            edit_background = edited_img * (1 - original_mask).unsqueeze(0)
            original_background = original_img * (1 - original_mask).unsqueeze(0)

            cilp_bg_cst = clip_imgimg_score(edit_background, original_background, _clip)
            lpips_bg_cst = lpips_score(edit_background, original_background, _lpips)
            dino_bg_cst = DINO_score(edit_background, original_background, _dino)
            l2_bg_cst = L2_score(edit_background, original_background)

            clip_obj_fid = 1.
            seg_obj_fid = 1.
            vqa_obj_fid = 1.

        else:
            target_img = (edited_img * target_mask.unsqueeze(0))[:, target_bbox[1]:target_bbox[3], target_bbox[0]:target_bbox[2]]
            target_img = square_pad(target_img)

            edit_background = edited_img * (1 - target_mask).unsqueeze(0) * (1 - original_mask).unsqueeze(0)
            original_background = original_img * (1 - target_mask).unsqueeze(0) * (1 - original_mask).unsqueeze(0)

            cilp_bg_cst = clip_imgimg_score(edit_background, original_background, _clip)
            lpips_bg_cst = lpips_score(edit_background, original_background, _lpips)
            dino_bg_cst = DINO_score(edit_background, original_background, _dino)
            l2_bg_cst = L2_score(edit_background, original_background)

            clip_obj_fid = 1. - clip_imgtext_score(target_img, target_name, _clip)
            seg_obj_fid = 1. - result['scores'][target_idx].item()
            vqa_obj_fid = 1. - vqa_score(target_img, target_name, _vqa)

    else:
        edit_background = edited_img * (1 - original_mask).unsqueeze(0)
        original_background = original_img * (1 - original_mask).unsqueeze(0)
        
        cilp_bg_cst = clip_imgimg_score(edit_background, original_background, _clip)
        lpips_bg_cst = lpips_score(edit_background, original_background, _lpips)
        dino_bg_cst = DINO_score(edit_background, original_background, _dino)
        l2_bg_cst = L2_score(edit_background, original_background)

        clip_obj_fid = 1.
        seg_obj_fid = 1.
        vqa_obj_fid = 1.
    
    return [cilp_bg_cst, lpips_bg_cst, dino_bg_cst, l2_bg_cst], [clip_obj_fid, seg_obj_fid, vqa_obj_fid, None, None]

def eval_result(query, ori_seg, result, original_img, edited_img, _clip, _lpips, _dino, _vqa):
    query_type = query['type']
    
    bg_cst, obj_cst, bg_fid, obj_fid = [None,None,None,None], [None,None,None,None,None,None], [None,None], [None,None,None,None,None]
    if query_type == 'obj_add': bg_cst, obj_fid = eval_objAdd(query, result, original_img, edited_img, _clip, _lpips, _dino, _vqa)
    elif query_type == 'obj_rep': bg_cst, obj_cst, obj_fid = eval_objRep(query, ori_seg, result, original_img, edited_img, _clip, _lpips, _dino, _vqa)
    elif query_type == 'obj_resize': bg_cst, obj_cst, obj_fid = eval_objResize(query, ori_seg, result, original_img, edited_img, [1.6,0.6], _clip, _lpips, _dino)
    elif query_type == 'attr_chg': bg_cst, obj_cst, obj_fid = eval_attrChg(query, ori_seg, result, original_img, edited_img, 0.9, 20, _clip, _lpips, _dino, _vqa)
    elif query_type == 'bg_chg': obj_cst, bg_fid = eval_bgChg(query, ori_seg, result, original_img, edited_img, _clip, _lpips, _dino, _vqa)
    elif query_type == 'style_chg': bg_cst, bg_fid = eval_styleChg(query, original_img, edited_img, 0.9, 20, _clip, _lpips, _dino, _vqa)
    elif query_type == 'obj_remove': bg_cst, obj_fid = eval_objRM(query, ori_seg, result, original_img, edited_img, _clip, _lpips, _dino, _vqa)

    return bg_cst, obj_cst, bg_fid, obj_fid