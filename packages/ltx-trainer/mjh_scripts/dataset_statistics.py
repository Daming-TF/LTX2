import os
import json
import cv2
from tqdm import tqdm
import pdb
import argparse
from datasets import load_dataset
from collections import Counter
"""
This script extracts video information (resolution, frame count, fps) from a JSON file containing video paths and captions. It saves the extracted information to a new JSONL file and provides statistics on the most common resolution buckets.
Usage:
    uv run python dataset_statistics.py 
    --json_file input.json 
    --output_file output.jsonl 
    --topk 10
Example:
    uv run python /root/autodl-tmp/mjh_proj/LTX-2/packages/ltx-trainer/mjh_scripts/dataset_statistics.py --json_file /root/autodl-tmp/data/scenes_clip_official_v2_dataset.json
"""


def extract_video_info(jsonl_file, output_file):
    with open(jsonl_file, 'r', encoding="utf-8") as f:
        data = json.load(f)
    results = []
    for item in tqdm(data):
        video_path = item['media_path']
        if not os.path.isabs(video_path):
            video_path_full = os.path.join(os.path.dirname(jsonl_file), video_path)
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
        results.append({
            "video_path": video_path_full,
            "caption": item.get("caption", ""),
            "ori_reso": f"{w}x{h}x{t}",             # WxHxF
            "reso": f"{w//32*32}x{h//32*32}x{(t-1)//8*8+1}",                      # WxH
            "fps": fps
        })
    # with open(output_file, 'w') as f:
    #     for r in results:
    #         f.write(json.dumps(r, ensure_ascii=False) + '\n')
    with open(output_file, 'w', encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)
    print(f"\033[35m Saved video info to {output_file}\033[0m")


def stat_reso_buckets(jsonl_file, topk=10, file_type="json"):
    ds = load_dataset(file_type, data_files=jsonl_file, split='train')
    # bucket_counts = ds.to_pandas()['reso'].value_counts()
    # print(bucket_counts)
    # pdb.set_trace()
    # print(f"Top {topk} resolution buckets:")

    reso_counter = Counter()
    for r in ds:
        reso_counter[(r['reso'],r['ori_reso'])] += 1
    for index, ((reso, ori_reso), count) in enumerate(reso_counter.most_common(topk), 1):
        print(f"top {index}\t|\tstandard(w*h*t):{reso}\toriginal:{ori_reso}\tcount:{count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_file', type=str, default="/root/autodl-tmp/data/scenes_clip_official_dataset.json", help='Path to the input JSON file')
    parser.add_argument('--output_file', type=str, default=None, help='Path to the output JSONL file')
    parser.add_argument('--file_endswith', type=str, default="jsonl")
    parser.add_argument('--topk', type=int, default=20, help='Number of top resolution buckets to display')
    args = parser.parse_args()
    if not args.output_file: args.output_file = args.json_file.replace('.json', f'_video_info.{args.file_endswith}') 
    extract_video_info(args.json_file, args.output_file)
    stat_reso_buckets(args.output_file, topk=args.topk)
