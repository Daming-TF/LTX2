# import cv2

# video_path = "/root/autodl-tmp/data/scenes_clip_official_v2/test-Scene-003.mp4"
# cap = cv2.VideoCapture(video_path)
# frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# cap.release()
# print("Total frames:", frame_count)




# import json
# import os

# jsonl_file = "/root/autodl-tmp/data/scenes_clip_official_v2_dataset_video_info.jsonl"
# with open(jsonl_file, 'r', encoding="utf-8") as f:
#     data = json.load(f)
# for _data in data:
#     video_path = _data['video_path']
#     if not os.path.basename(video_path).startswith("test2-Scene-001"): continue
#     print(_data['caption'])



# import torch
# # 1056x1920x97          640x352x65
# audio_file_path = "/root/autodl-tmp/data/.ltx2_precomputed/audio_latents/scenes_clip_official_v2/test-Scene-003.pt"
# # audio_file_path = "/root/autodl-tmp/data/.ltx2_precomputed/audio_latents/scenes_clip_official_v2/test2-Scene-001.pt"
# tmp = torch.load(audio_file_path, map_location="cpu", weights_only=True)
# print("audio latents infomation:", tmp["latents"].shape, tmp['num_time_steps'], tmp['frequency_bins'], tmp["duration"])


# video_file_path = "/root/autodl-tmp/data/.ltx2_precomputed/latents/scenes_clip_official_v2/test-Scene-003.pt"
# # video_file_path = "/root/autodl-tmp/data/.ltx2_precomputed/latents/scenes_clip_official_v2/test2-Scene-001.pt"
# tmp = torch.load(video_file_path, map_location="cpu", weights_only=True)
# print("video latents infomation:", tmp["latents"].shape, tmp['num_frames'], tmp['height'], tmp["width"], tmp["fps"])




######### Test dataloader
from accelerate import Accelerator
from datasets import load_dataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import pdb
from pathlib import Path
import torch

import sys
sys.path.append("/root/autodl-tmp/mjh_proj/LTX-2/packages/ltx-trainer/src")
from ltx_trainer.datasets import PrecomputedDataset
sys.path.append(Path(__file__).parent.parent)
from mjh_scripts.utils.batch_bucket import AspectRatioBatchSampler, prepare_train_dataset, collate_fn

BUCKET_EXAMPLE = {
    "1.78-97":[512,288,97],"0.55-65":[352,352,65]
}
num_workers = 1
data_sources = {
    "video_latents": "latents",
    "conditions": "conditions",
    "audio_latents": "audio_latents",
}

#### ori method to load dataset
# data_root = "/root/autodl-tmp/data/.ltx2_precomputed"
# dataset = PrecomputedDataset(data_root, data_sources)
# dataloader = DataLoader(
#     dataset,
#     batch_size=1,
#     shuffle=True,
#     drop_last=True,
#     num_workers=num_workers,
#     pin_memory=num_workers > 0,
#     persistent_workers=num_workers > 0,
# )
#####

#### bucket method to load dataset
accelerator = Accelerator(mixed_precision="bf16",gradient_accumulation_steps=1)
jsonl_file = "/root/autodl-tmp/data/scenes_clip_official_v2_dataset_for_training.jsonl"
dataset = load_dataset("json", data_files=jsonl_file, split='train')
if "sample_idx" not in dataset.column_names:
    dataset = dataset.add_column("sample_idx", list(range(len(dataset))))
ratio_nums = dataset.to_pandas().groupby('bucket').size().to_dict()
dataset = prepare_train_dataset(dataset, accelerator, data_sources, BUCKET_EXAMPLE)
generator=torch.Generator().manual_seed(42)
batch_sampler = AspectRatioBatchSampler(
    sampler=RandomSampler(dataset, generator=generator), 
    # sampler=SequentialSampler(dataset),
    dataset=dataset,
    batch_size=4, aspect_ratios=BUCKET_EXAMPLE, drop_last=True,
    ratio_nums=ratio_nums, 
    valid_num=0,
)
dataloader = torch.utils.data.DataLoader(
    dataset, 
    batch_sampler=batch_sampler, 
    collate_fn=collate_fn
)
######

for batch in dataloader:
    pdb.set_trace()
    print(batch['v_latents'].shape, batch['c_prompt_embeds'].shape, batch['a_latents'].shape)
    print(batch['sample_idx'])
#########


# import yaml
# import pdb
# config_path = "/root/autodl-tmp/mjh_proj/LTX-2/packages/ltx-trainer/configs/ltx2_av_lora_mjh.yaml"
# with open(config_path, "r") as file:
#     cfd = yaml.safe_load(file)
# # pdb.set_trace()
# tmp = cfd['validation']['video_dims']
# print(type(tmp), tmp)
