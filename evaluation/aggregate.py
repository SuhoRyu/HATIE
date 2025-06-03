import os
import sys
import torch
import numpy as np
import pickle
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_queries
from utils.agg_utils import *
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_file", default="queries/queries_wo_remove.pkl")
    parser.add_argument("--prefix", type=str, default='')
    parser.add_argument("--outdir", default="outputs/")
    parser.add_argument("--compute_errors", type=int, default=1)
    args = parser.parse_args()
    return args


def main():
    #parse and print arguments
    args = parse_args()
    for arg in vars(args): print(f"[{arg}]:\n{getattr(args, arg)}")
    print()

    #load queries & scores
    print("Loading data ...")
    queries, _, _ = load_queries(args.query_file)
    with open(os.path.join(args.outdir, f'scores/scores_{args.prefix}.pkl'), "rb") as f:
        scores = pickle.load(f)
    with open(os.path.join(args.outdir, f'scores/fid_{args.prefix}.json'), 'r') as f:
        fid_score = json.load(f)
    img_qual = [fid_score['imqual'], fid_score['imqual_err']]

    #Aggregate scores
    print("Calculating scores ...")
    with torch.no_grad():
        if os.path.exists(os.path.join(args.outdir, f'scores/scores_{args.prefix}_agg.pkl')):
            print("Loading previous calculations ...")
            with open(os.path.join(args.outdir, f'scores/scores_{args.prefix}_agg.pkl'), "rb") as f:
                aggScores = pickle.load(f)

        else:
            print("Calculating per query scores ...")
            aggScores = []
            for result in scores:
                query_info = result['query_info']
                query_type = queries[query_info['imgID']][query_info['query_idx']]['type']
                query_info['type'] = query_type
                bg_cst, obj_cst, bg_fid, obj_fid = agg_score(query_type, result)

                aggScores.append({'query_info': query_info, 'bg_cst': bg_cst, 'obj_cst': obj_cst, 'bg_fid': bg_fid, 'obj_fid': obj_fid})

            with open(os.path.join(args.outdir, f'scores/scores_{args.prefix}_agg.pkl'), 'wb') as f:
                pickle.dump(aggScores, f)
        
        #Calculate total scores
        print("Calculating total scores with error ...")
        total_scores = merge_scores(aggScores, img_qual, get_err=args.compute_errors)
        with open(os.path.join(args.outdir, f'scores/scores_{args.prefix}_total.json'), 'w') as f:
            json.dump(total_scores, f, indent=4)

        #Calculate scores for each query type 
        print("Calculating scores for each query type ...")
        qtype_scores = query_type_scores(aggScores, get_err=args.compute_errors, do_remove=args.query_file.endswith('w_remove.pkl'))
        with open(os.path.join(args.outdir, f'scores/scores_{args.prefix}_qtypes.json'), 'w') as f:
            json.dump(qtype_scores, f, indent=4)

        print(f"Calcualtion complete.")

    return

if __name__ == "__main__":
    main() 