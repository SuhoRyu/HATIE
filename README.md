# HATIE: Towards Scalable Human-Aligned Benchmark for Text-guided Image Editing

[Suho Ryu](https://scholar.google.com/citations?user=fQCeEH0AAAAJ&hl)\,
Kihyun Kim\,
Eugene Baek\,
Dongsoo Shin\,
[Joonseok Lee](https://viplab.snu.ac.kr/)\
CVPR '25 Highlight |
[GitHub](https://github.com/SuhoRyu/HATIE) | [arXiv](https://arxiv.org/abs/2505.00502)

![t2i](images/main.png)

HATIE is a comprehensive evaluation framework developed to objectively assess text-guided image editing models. HATIE introduces a large-scale, diverse benchmark set, covering various editing tasks. It employs an automated, multifaceted evaluation pipeline that aligns closely with human perception. HATIE enables scalable, reproducible, and precise benchmarking of image editing models.

 
## Setup
### Environment setup
The following instructions should work on most machines. However, depending on your specific environment and system configuration, you may need to install PyTorch and CUDA separately to ensure compatibility.

```bash
conda create -n hatie python=3.10
conda activate hatie

conda install pytorch=2.7.0 torchvision=0.22.0 torchaudio=2.7.0 cudatoolkit=11.8 -c pytorch -c conda-forge
pip install -r requirements.txt
```

### Load Codes and Data 
```bash
# clone repository
git clone https://github.com/SuhoRyu/HATIE.git
cd HATIE
```
```bash
# download image set
git lfs install
git clone https://huggingface.co/SHRyu97/HATIE
unzip HATIE/original_images.zip
```
Alternatively, you can download "HATIE_original_images.zip" and "editable_objs_mask.pkl" from [HuggingFace](https://huggingface.co/SHRyu97/HATIE). After downloading, unzip the archive (if needed) and place the contents in any desired location on your system.

## Run
### 1. Run Your Model
The files "queries/queries_w_remove.pkl" and "queries/queries_wo_remove.pkl" contain all necessary information of the benchmark queries. Specifically, "queries_w_remove.pkl" includes object removal, whereas "queries_wo_remove.pkl" contains queries excluding object removal cases. You can load the desired query file using the pickle module.

```python
import pickle
with open('queries_w_remove.pkl', 'rb') as f:
    queries = pickle.load(f)
```

The queries are formatted as a dictionary containing lists of queries, structured as shown below.

```bash
{
 '2373554':
    [
     {'type': 'obj_rep',
      'original': ['3143517', 'person'],
      'target': 'cup',
      'id': 23847,
      'original_caption': 'a young person stands on snowshoes.',
      'target_caption': 'A cup stands on its snowshoes.',
      'instruction': 'Replace the person with a cup.'},
     {'type': ...
    ],
 '2370790': ...
}
```

The keys of the outermost dictionary ('2373554', '2370790', etc.) represent the image IDs of the original images, corresponding directly to the filenames ('2373554.jpg', '2370790.jpg', etc.). Each key maps to a list of query dictionaries that should be applied to the respective original image. Within each query dictionary, you'll find generated text prompts tailored for your model. Specifically, 'original_caption' and 'target_caption' are intended for description-based models, while 'instruction' is suitable for instruction-based models. Each query dictionary also includes a unique query ID ('id') that identifies each query across the entire dataset. 

Run your model across all original images and query prompts, saving the generated outputs into a single folder. You can freely choose the naming prefix, but ensure consistency throughout. Each edited image filename should end with "_{query ID}.jpg". For example, if your chosen prefix is "output_modelA", an appropriate filename would be "output_modelA_23847.jpg" (query ID = 23847 in the example above). While JPG is set as the default format in the provided code, you're free to select a different format if preferred.

### 2. Target Segmentation
The first step in the HATIE pipeline involves locating and segmenting all necessary objects for evaluation from the edited images. Open the file `segment.sh`, which should appear as shown below.

```bash
#!/bin/bash

python evaluation/segment_targets.py \
    --query_file "queries/queries_wo_remove.pkl" \    # The query file you used for editing
    --prefix "output_modelA" \                        # The file name prefix of edited images
    --edited_images "path/to/edited/images" \         # path to the edited images
    --outdir "outputs" \                              # path to where you desire to save the benchmark results
    --image_format "jpg" \                            # format of the edited images
    --use_gpu 1                                       # use GPU = 1, don't use GPU = 0
```

Set each option appropriately based on your specific run configurations, then execute the script as follows:

```bash
./segment.sh
```

Then, the code will save the resulting segmentation mask files into the output directory you specified.

### 3. Scoring
The next step is the actual scoring phase. Using the segmentation mask files obtained from the previous step, the following script computes scores for each edited output image using all metrics included in HATIE. Open the "score.sh" file, which should appear as shown below.

```bash
#!/bin/bash

python evaluation/score.py \
    --query_file "queries/queries_wo_remove.pkl" \           # The query file you used for editing
    --prefix "output_modelA" \                               # The file name prefix of edited images
    --original_seg_file "HATIE/editable_objs_mask.pkl" \     # path to the original images' object segmentation mask file 
    --original_images "original_images" \                    # path to original images
    --edited_images "path/to/edited/images" \                # path to edited images
    --outdir "outputs" \                                     # path to where you desire to save the benchmark results
    --use_gpu 1 \                                            # use GPU = 1, don't use GPU = 0
    --compute_err 1                                          # compute only the benchmark scores = 0, compute scores with error = 1
```

Set each option according to your specific run configuration, then execute the script as follows:

```bash
./score.sh
```

The code will log the output scores into the output directory you specified.

### 4. Aggregate Scores
The final phase aggregates all the scores obtained during the scoring phase into final model scores. Open the "aggregate.sh" file, which should look like the example below.

```bash
#!/bin/bash

python evaluation/aggregate.py \
    --query_file "queries/queries_wo_remove.pkl" \    # The query file you used for editing
    --prefix "output_modelA" \                        # The file name prefix of edited images
    --outdir "outputs" \                              # path to where you desire to save the benchmark results
    --compute_errors 1                                # compute only the benchmark scores = 0, compute scores with error = 1
```

Set each option appropriately for your run, then execute the script with the following command:

```bash
./aggregate.sh
```

The code will aggregate the scores of each output into final model scores and save the results in the output directory you specified.

You will receive two JSON files containing the final model scores:

1. "{output path}/scores/scores_{prefix}_total.json"
   - This file contains the aggregated total score along with scores for the five evaluation criteria:
   object fidelity, object consistency, background fidelity, background consistency, and image quality.

2. "{output path}/scores/scores_{prefix}_qtype.json"
   – This file provides scores aggregated separately for each query type:
   object add, replace, remove, attribute change, resize, background change and style change.

Each score is represented as a list of two values: "[score, error]", where the first value is the score, and the second is the corresponding error. If the error option was disabled, the error values will be "null".

### 5. Resume
Except for "aggregate.sh", both "segment.sh" and "score.sh" are safe to interrupt and resume. To resume a process, simply re-run the corresponding script. The code will automatically continue from where it left off.

The editing and segmentation processes do not need to be fully completed before running the subsequent scripts. Each script will proceed using the available results and will exit with a message indicating any unfinished items. This means you can run parts of the HATIE pipeline multiple times during the (potentially time-consuming) editing process to save time on evaluation.

## Acknowledments
This implementation of HATIE integrates several pre-existing tools and libraries. Specifically, it imports:

* [Detectron2](https://github.com/facebookresearch/detectron2)
* [LPIPS](https://github.com/richzhang/PerceptualSimilarity)
* [CLIP](https://github.com/openai/CLIP)
* [DINOv2](https://github.com/facebookresearch/dinov2)

In addition, modified versions of 

* [pytorch-fid](https://github.com/mseitzer/pytorch-fid)
* [t2v\_metrics](https://github.com/linzhiqiu/t2v_metrics)

are embedded directly into the codebase. We acknowledge and appreciate the contributions of these original repositories.

 
## Citing HATIE
```latex
@inproceedings{ryu2025towards,
  title={Towards Scalable Human-aligned Benchmark for Text-guided Image Editing},
  author={Ryu, Suho and Kim, Kihyun and Baek, Eugene and Shin, Dongsoo and Lee, Joonseok},
  journal={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  year={2025}
}
```

 
