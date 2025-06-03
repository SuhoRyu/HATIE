import os
import sys
import torch
import numpy as np
import pickle
import json
from PIL import Image
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.score_utils import *
from utils import load_queries
from tqdm import tqdm
from utils.fid import compute_fid

IMAGE_SIZE = 512

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_file", default="queries/queries_wo_remove.pkl")
    parser.add_argument("--original_seg_file", default="HATIE/editable_objs_mask.pkl")
    parser.add_argument("--prefix", type=str, default='')
    parser.add_argument("--original_images", default='original_images/')
    parser.add_argument("--edited_images", default=None)
    parser.add_argument("--outdir", default="outputs/")
    parser.add_argument("--use_gpu", type=int, default=1)
    parser.add_argument("--compute_err", type=int, default=1)
    """
    # You 'can' use different pretrained models for below, but we highly recommend using the default ones.
    # To use different models, You may need to modify the codes, especially the image preprocessors of the metric models.
    # Plus, all the tunnings are based on the default models. 
    """
    parser.add_argument("--clip", type=str, default="ViT-B/32")
    parser.add_argument("--dino", type=str, default="facebook/dinov2-base")
    parser.add_argument("--lpips", type=str, default="alex")
    parser.add_argument("--vqa", type=str, default="clip-flant5-xxl")
    args = parser.parse_args()
    return args


def main():
    #parse and print arguments
    args = parse_args()
    if args.original_images is None:
        raise ValueError("Please provide path to the original images.")
    if args.edited_images is None:
        raise ValueError("Please provide path to the edited images.")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if args.use_gpu==1 else torch.device('cpu')
    for arg in vars(args): print(f"[{arg}]:\n{getattr(args, arg)}")
    print()

    #build models
    print("Loading models ...")
    _clip = build_clip(args.clip, device)
    _dino = build_dino(args.dino, device)
    _lpips = build_lpips(args.lpips, device)
    _vqa = build_vqa(args.vqa, device)

    #load queries
    print("Loading data ...")
    queries, queryIds, _ = load_queries(args.query_file)

    #load original image segmentation data
    with open(args.original_seg_file, "rb") as f:
        original_seg_data = pickle.load(f)
    imageIds = [data['image'] for data in original_seg_data]

    numQueries = len(queryIds)
    queries_per_file = 2000
    numSegFiles = int(np.ceil(numQueries / queries_per_file))

    #load or make progress status for resuming
    os.makedirs(os.path.join(args.outdir, f"scores"), exist_ok=True)
    if os.path.exists(os.path.join(args.outdir, f"scores/score_stat.json")):
        with open(os.path.join(args.outdir, f"scores/score_stat.json"), "r") as f:
            score_stat = json.load(f)
        if args.prefix not in score_stat:
           score_stat[args.prefix] = [0 for _ in range(numSegFiles)]
    else:
        score_stat = {args.prefix:[0 for _ in range(numSegFiles)]}

    #Run Scoring
    print("Scoring ...\n")
    with torch.no_grad():
        #load or create score results
        if os.path.exists(os.path.join(args.outdir, f'scores/scores_{args.prefix}.pkl')):
            with open(os.path.join(args.outdir, f'scores/scores_{args.prefix}.pkl'), 'rb') as f:
                score_res = pickle.load(f)
        else:
            score_res = [{'query_info': None, 'instruction': None, 'bg_cst':None, 'obj_cst':None, 'bg_fid':None, 'obj_fid':None} for _ in range(numQueries)]

        #for each segmentation file
        for file_idx in range(numSegFiles):
            count = 0
            #skip finished or missing files
            if score_stat[args.prefix][file_idx] == 1: 
                print(f'Skipping segmentation file #{file_idx} ... \n')
                continue
            if not os.path.exists(os.path.join(args.outdir, f'segmentations/{args.prefix}/segmentation_{args.prefix}_{file_idx}.pkl')):
                print(f'Segmentation file #{file_idx} Missing. Skipping ... ')
                continue
            print(f'Processing segmentation file #{file_idx} ... ')
            
            with open(os.path.join(args.outdir, f'segmentations/{args.prefix}/segmentation_{args.prefix}_{file_idx}.pkl'), 'rb') as f:
                seg_res = pickle.load(f)
            
            #evaluate each query in the file
            original_img = None
            for i, result in tqdm(list(enumerate(seg_res)), 
                                  total=len(seg_res), 
                                  ncols = 50):
                if result['class_names'] == None: continue

                imgID = result['query_info']['imgID']
                query_idx = result['query_info']['query_idx']
                img_fname = result['query_info']['img_name']
                query = queries[imgID][query_idx]

                #skip if already evaluated or image is missing
                if score_res[i+queries_per_file*file_idx]['query_info'] == None and os.path.exists(os.path.join(args.edited_images, img_fname)):   
                    #load original image and segmentation data
                    if query_idx == 0 or original_img == None:
                        original_img = Image.open(os.path.join(args.original_images, imgID+'.jpg')).convert('RGB')
                        original_img = totensor(original_img).to(device)
                        if original_img.shape[0] == 1: original_img = original_img.repeat(3,1,1)
                        ori_seg = original_seg_data[imageIds.index(imgID)]['editable_objs']

                    #load edited image
                    edited_img = Image.open(os.path.join(args.edited_images, img_fname)).convert('RGB')
                    if edited_img.size[0] != IMAGE_SIZE or edited_img.size[1] != IMAGE_SIZE: edited_img = edited_img.resize((IMAGE_SIZE, IMAGE_SIZE))
                    edited_img = totensor(edited_img).to(device)
                    if edited_img.shape[0] == 1: edited_img = edited_img.repeat(3,1,1)

                    #score
                    bg_cst, obj_cst, bg_fid, obj_fid = eval_result(query, ori_seg, result, original_img, edited_img, _clip, _lpips, _dino, _vqa)

                    #log results
                    score_res[i+queries_per_file*file_idx]['query_info'] = result['query_info']
                    score_res[i+queries_per_file*file_idx]['instruction'] = queries[imgID][query_idx]['instruction']
                    score_res[i+queries_per_file*file_idx]['bg_cst'] = bg_cst
                    score_res[i+queries_per_file*file_idx]['obj_cst'] = obj_cst
                    score_res[i+queries_per_file*file_idx]['bg_fid'] = bg_fid
                    score_res[i+queries_per_file*file_idx]['obj_fid'] = obj_fid
                    
                    count+=1

            #save results      
            if count != 0:
                print('Saving...')
                with open(os.path.join(args.outdir, f'scores/scores_{args.prefix}.pkl'), 'wb') as f:
                    pickle.dump(score_res, f)

            #check if all queries in the segmentation file are finished
            if all([score_res[idx]['query_info']!=None for idx in range(queries_per_file*file_idx, min(queries_per_file*(file_idx+1), len(score_res)))]): 
                print(f'Finished file #{file_idx} ... \n')
                score_stat[args.prefix][file_idx] = 1
                with open(os.path.join(args.outdir, "scores/score_stat.json"), "w") as f:
                    json.dump(score_stat, f, indent=4)
                    
            #if not inform
            else:
                print(f'Segmentation File #{file_idx} incomplete ... \n')
                with open(os.path.join(args.outdir, "scores/score_stat.json"), "w") as f:
                    json.dump(score_stat, f, indent=4)

        #compute and saver FID score
        if not os.path.exists(os.path.join(args.outdir, f'scores/fid_{args.prefix}.json')):
            print("Computing FID score ...")
            fid_score, fid_err, imqual, imqual_err = compute_fid(args.prefix, 
                                                                 args.original_images, 
                                                                 args.edited_images, 
                                                                 args.outdir, 
                                                                 device=device,
                                                                 get_err=args.compute_err,
                                                                 numResample=30)
            with open(os.path.join(args.outdir, f'scores/fid_{args.prefix}.json'), 'w') as f:
                json.dump({'fid': fid_score, 'fid_err': fid_err, 'imqual': imqual, 'imqual_err': imqual_err,
                           "prefix": args.prefix, "original": args.original_images, "edited": args.edited_images}, f, indent=4)
                print("FID score saved.")
        else:
            print("FID score already computed.")
        
        

    return




if __name__ == "__main__":
    main()