import os
import sys
import cv2
import pickle
import torch
import json
import argparse
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.seg_utils import load_detector, get_main_objs_segments
from utils import load_queries
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_file", default="queries/queries_wo_remove.pkl")
    parser.add_argument("--prefix", type=str, default='')
    parser.add_argument("--edited_images", default=None)
    parser.add_argument("--outdir", default="outputs/")
    parser.add_argument("--image_format", default="jpg")
    parser.add_argument("--use_gpu", type=int, default=1)
    """
    # You can use different pretrained model for below, but we recommend using the default one.
    # Keep in mind that, all the tunnings are based on the default model. 
    """
    parser.add_argument("--detector", default="COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
    args = parser.parse_args()
    return args

def main():
    #parse and print arguments
    args = parse_args()
    if args.edited_images is None:
        raise ValueError("Please provide path to the edited images.")
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.use_gpu==1 else 'cpu'
    for arg in vars(args): print(f"[{arg}]:\n{getattr(args, arg)}")
    print()

    #load queries
    print("Preparing process ...")
    queries, queryIds, queryIdxs = load_queries(args.query_file)

    #load detector
    detector = load_detector(args.detector, device=device)
    
    numQuery = len(queryIds)
    queries_per_file = 2000
    numFiles = int(np.ceil(numQuery / queries_per_file))

    #load or make progress status for resuming
    os.makedirs(os.path.join(args.outdir, f"segmentations/{args.prefix}"), exist_ok=True)
    if os.path.exists(os.path.join(args.outdir, f"segmentations/seg_stat.json")):
        with open(os.path.join(args.outdir, f"segmentations/seg_stat.json"), "r") as f:
            seg_stat = json.load(f)
        if args.prefix not in seg_stat:
            seg_stat[args.prefix] = [0 for _ in range(numFiles)]
    else:
        seg_stat = {args.prefix:[0 for _ in range(numFiles)]}

    #Run segmentation
    print("Running segmentation ...\n")
    with torch.no_grad():
        for file_idx in range(numFiles):
            #Skip finished files
            if seg_stat[args.prefix][file_idx] == 1: 
                print(f'File #{file_idx} Already Done ... \n')
                continue
            print(f'Processing file #{file_idx}/{numFiles} ... ')
            
            #Load or create segmentation results
            if os.path.exists(os.path.join(args.outdir, f'segmentations/{args.prefix}/segmentation_{args.prefix}_{file_idx}.pkl')):
                with open(os.path.join(args.outdir, f'segmentations/{args.prefix}/segmentation_{args.prefix}_{file_idx}.pkl'), 'rb') as f:
                    segResults = pickle.load(f)
            else:
                segResults = []
            
            #Run for each query in the file
            missing_files = []
            for fqidx, (qid, (iid, qidx)) in tqdm(enumerate(zip(queryIds[file_idx*queries_per_file:(file_idx+1)*queries_per_file], 
                                                                queryIdxs[file_idx*queries_per_file:(file_idx+1)*queries_per_file])),
                                                  total=len(queryIds[file_idx*queries_per_file:(file_idx+1)*queries_per_file]),
                                                  ncols = 50):
                img_fname = f'{args.prefix}_{qid}.{args.image_format}'
                if len(segResults) <= fqidx: 
                    segResults.append({'query_info': {'id': qid, 'imgID': iid, 'query_idx': qidx, 'img_name': img_fname}, 
                                        'class_names': None, 'scores': None, 'bboxs': None, 'masks': None})
                
                #Run only for unfinished queries
                if segResults[fqidx]['class_names'] is None:
                    #Skip and log if image is missing
                    if not os.path.exists(os.path.join(args.edited_images, img_fname)): 
                        missing_files.append(img_fname)
                        continue
                    
                    #Load image and run segmentation
                    img = cv2.imread(os.path.join(args.edited_images, img_fname))
                    if img.shape[0] != 512 or img.shape[1] != 512: img = cv2.resize(img, (512, 512))
                    query_info = queries[iid][qidx]

                    (
                     segResults[fqidx]['class_names'],
                     segResults[fqidx]['scores'],
                     segResults[fqidx]['bboxs'],
                     segResults[fqidx]['masks']
                    ) = get_main_objs_segments(detector, img, query_info)

            #Save results
            print(f'Saving ...')
            with open(os.path.join(args.outdir, f'segmentations/{args.prefix}/segmentation_{args.prefix}_{file_idx}.pkl'), 'wb') as f:
                pickle.dump(segResults, f)
            
            #Check if all queries in the file are finished
            if all([segResults[i]['class_names']!=None for i in range(len(segResults))]): 
                #if so, mark the file as finished and print
                print(f'Finished file #{file_idx} ... \n')
                seg_stat[args.prefix][file_idx] = 1
                with open(os.path.join(args.outdir, "segmentations/seg_stat.json"), "w") as f:
                    json.dump(seg_stat, f, indent=4)

            else:
                #if not, mark the file as unfinished and print missing files
                print(f'File #{file_idx} not finished ... ')
                with open(os.path.join(args.outdir, "segmentations/seg_stat.json"), "w") as f:
                    json.dump(seg_stat, f, indent=4)
                for missing_file in missing_files:
                    print(f"Missing file: {missing_file}")
                print()
            
    return





if __name__ == "__main__":
    main()