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




# ######### Test dataloader
# from torch.utils.data import DataLoader
# import pdb

# import sys
# sys.path.append("/root/autodl-tmp/mjh_proj/LTX-2/packages/ltx-trainer/src")
# from ltx_trainer.datasets import PrecomputedDataset
# data_root = "/root/autodl-tmp/data/.ltx2_precomputed"
# data_sources = {
#     "latents": "latents",
#     "conditions": "conditions",
#     "audio_latents": "audio_latents",
# }
# dataset = PrecomputedDataset(data_root, data_sources)
# ratio_nums = dataset.to_pandas().groupby('bucket').size().to_dict()
# num_workers = 1
# dataloader = DataLoader(
#     dataset,
#     batch_size=1,
#     shuffle=True,
#     drop_last=True,
#     num_workers=num_workers,
#     pin_memory=num_workers > 0,
#     persistent_workers=num_workers > 0,
# )
# for batch in dataloader:
#     pdb.set_trace()
#     print(batch['latents']['latents'].shape)
# #########


# from datasets import load_dataset
# import pdb
# json_file = "/root/autodl-tmp/data/scenes_clip_official_v2_dataset_video_info.jsonl"
# dataset = load_dataset(
#     "json",
#     data_files=json_file,
# )["train"]
# ratio_nums = dataset.to_pandas().groupby('bucket').size().to_dict()
# pdb.set_trace()
# print("successfully load dataset")


from pathlib import Path
tmp = Path(__file__).parent/"test.txt"
tmp = tmp.with_suffix('.jpg')
print(tmp)