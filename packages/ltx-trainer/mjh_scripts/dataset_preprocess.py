import os
import json
import cv2
from tqdm import tqdm
import pdb
import argparse
from datasets import load_dataset
from collections import Counter
from pathlib import Path

"""
📌 Step1: Split Scenes: -/ltx-trainer/scripts/split_scenes.py
📌 Step2: Video Caption: -/ltx-trainer/scripts/video_caption.py
👀 Step3: Get Aspect Ratio Buckets: -/ltx-trainer/mjh_scripts/dataset_preprocess.py
📌 Step4: Precompute Latents: -/ltx-trainer/scripts/process_dataset.py
👀 Step5: Transfer jsonl file for Training: -/ltx-trainer/scripts/dataset_preprocess.py
Output Structure:
        ~/data/
            └── .precomputed/
                └── latents/            # Cached video latents
                    └── dataset1
                        └── *.pt
                    └── dataset2
                        └── *.pt
                ├── conditions/         # Cached text embeddings
                ├── audio_latents/      # (only if --with-audio) Cached audio latents
                └── reference_latents/  # (only for IC-LoRA) Cached reference video latents
            └── dataset1/
                └── *.mp4
            ├── dataset1_video_clip.json
            ├── dataset1_video_clip_for_getting_bucket.jsonl
            ├── dataset1_video_clip_for_training.jsonl
            └── dataset2/
                └── *.mp4
            └── ...

This script extracts video information (resolution, frame count, fps) from a JSON file containing video paths and captions. It saves the extracted information to a new JSONL file and provides statistics on the most common resolution buckets.
Usage:
    uv run python dataset_statistics.py 
    --json_file input.json 
    --output_file output.jsonl 
    --topk 10
Example:
    Mode-get_bucket:
        uv run python /root/autodl-tmp/mjh_proj/LTX-2/packages/ltx-trainer/mjh_scripts/dataset_preprocess.py \
            --json_file /root/autodl-tmp/data/scenes_clip_official_v2_dataset.json
    Mode-transfer2jsonl:
        python /root/autodl-tmp/mjh_proj/LTX-2/packages/ltx-trainer/mjh_scripts/dataset_preprocess.py \
            --mode transfer2jsonl \
            --json_file /root/autodl-tmp/data/spython /root/autodl-tmp/mjh_proj/LTX-2/packages/ltx-trainer/mjh_scripts/dataset_preprocess.pycenes_clip_official_v2_dataset_video_info.json \
            --jsonl_output_path /root/autodl-tmp/data/scenes_clip_official_v2_dataset_for_training.json \
            --cache_dir /root/autodl-tmp/data/.ltx2_precomputed
"""

BUCKET_EXAMPLE = {
    "1.78-97":[512,288,97],"0.55-65":[352,640,65]
}   # (WHF)


def get_closet_bucket(height, width, frame, aspect_ratios):
    aspect_ratio = width / height
    closest_bucket = min(aspect_ratios.keys(), key=lambda x: abs(aspect_ratio - float(x.split('-')[0])))
    return closest_bucket


def extract_video_info(json_file, output_file):
    """
    Arg:
        json_file: Path to the input JSON file containing video clip paths and captions.
        output_file: Path to the output JSONL file where extracted video information will be saved.
    """
    with open(json_file, 'r', encoding="utf-8") as f:
        data = json.load(f)
    results = []
    for item in tqdm(data):
        video_path = item['media_path']
        if not os.path.isabs(video_path):
            video_path_full = os.path.join(os.path.dirname(json_file), video_path)
        else:
            video_path_full = video_path
        cap = cv2.VideoCapture(video_path_full)
        if not cap.isOpened():
            print(f"Failed to open {video_path_full}")
            continue
        w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        t = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        cap.release()
        bucket = get_closet_bucket(w, h, t, BUCKET_EXAMPLE)
        results.append({
            "video_path": video_path_full,
            "caption": item.get("caption", ""),
            "ori_reso": f"{w}x{h}x{t}",             # WxHxF
            # "reso": f"×".join(map(str, BUCKET_EXAMPLE[bucket])),                      # WxH
            "bucket": bucket,                         # aspect ratio bucket
            "fps": fps
        })
    # with open(output_file, 'w') as f:
    #     for r in results:
    #         f.write(json.dumps(r, ensure_ascii=False) + '\n')
    with open(output_file, 'w', encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)
    print(f"\033[35m Saved video info to {output_file}\033[0m")


def get_aspect_ratio_buckets(jsonl_file, topk=10, file_type="json"):
    ds = load_dataset(file_type, data_files=jsonl_file, split='train')
    # bucket_counts = ds.to_pandas()['reso'].value_counts()
    # print(bucket_counts)
    # pdb.set_trace()
    # print(f"Top {topk} resolution buckets:")
    reso_counter = Counter()
    for r in ds:
        reso_counter[(r['bucket'])] += 1
    for index, (bucket, count) in enumerate(reso_counter.most_common(topk), 1):
        print(f"top {index}\t|\t bucket:{bucket} \t stand reso:{BUCKET_EXAMPLE[bucket]} \tcount:{count}")


def transfer2jsonl(jsonl_file, output_file, cache_dir, w_audio=True):
    """
    Transfer json file to jsonl file for training.
    Arg:
        jsonl_file: Path to the input JSON file containing video clip paths and captions.
        output_file: Path to the output JSONL file where extracted video information will be saved.
    """
    jsonl_file = Path(jsonl_file)
    root_dir = jsonl_file.parent
    if not output_file: output_file = jsonl_file.replace('.json', f'_for_training.jsonl')
    cache_dir = root_dir / ".ltx2_precomputed" if not cache_dir else Path(cache_dir)
    if not jsonl_file.exists() or not cache_dir.exists():
        raise FileNotFoundError(f"Input jsonl file {jsonl_file} or cache dir {cache_dir} does not exist.")
    data_sources = ['latents', 'conditions', 'audio_latents'] if w_audio else ['latents', 'conditions']
    with open(jsonl_file, 'r', encoding="utf-8") as f:
        data = json.load(f)
    for _data in tqdm(data):
        video_path = Path(_data['video_path'])
        # get relative path of video to root_dir
        video_rel_path = video_path.relative_to(root_dir)
        for source_key in data_sources:
            latent_path = cache_dir / source_key / video_rel_path.with_suffix('.pt')
            if latent_path.exists():
                _data[source_key] = str(latent_path)
            else:
                raise FileNotFoundError(f"Latent file {latent_path} does not exist. Please run the precompute script first.")
    with open(output_file, 'w', encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    print(f"Transfered '{jsonl_file}' to \033[35m {output_file}\033[0m")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_file', type=str, default="/root/autodl-tmp/data/scenes_clip_official_dataset.json", help='Path to the input JSON file')
    parser.add_argument('--jsonl_output_path', type=str, default=None, help='Path to the output JSONL file')
    parser.add_argument('--cache_dir', type=str, default=None, help='Directory to cache precomputed latents')
    parser.add_argument('--w_audio', action='store_false', help='Whether include audio information')
    parser.add_argument('--mode', type=str, default="get_bucket", choices=["get_bucket", "transfer2jsonl"], help='Mode of operation')
    parser.add_argument('--topk', type=int, default=20, help='Number of top resolution buckets to display')
    args = parser.parse_args()
    if args.mode=="get_bucket": # Get Aspect Ratio Buckets
        assert args.json_file.endswith('.json'), "Input file must be a JSON file."
        if not args.jsonl_output_path: args.jsonl_output_path = args.json_file.replace('.json', f'_for_getting_bucket.jsonl')
        extract_video_info(args.json_file, args.jsonl_output_path)
        get_aspect_ratio_buckets(args.jsonl_output_path, topk=args.topk)
    elif args.mode=="transfer2jsonl": # Transfer json file to jsonl file for training
        transfer2jsonl(args.json_file, args.jsonl_output_path, args.cache_dir, args.w_audio)
    else:
        raise ValueError(f"Invalid mode: {args.mode}. Choose from ['get_bucket', 'transfer2jsonl']")

