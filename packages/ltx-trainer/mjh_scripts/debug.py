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



import torch
# 1056x1920x97          640x352x65
audio_file_path = "/root/autodl-tmp/data/.ltx2_precomputed/audio_latents/scenes_clip_official_v2/test-Scene-003.pt"
# audio_file_path = "/root/autodl-tmp/data/.ltx2_precomputed/audio_latents/scenes_clip_official_v2/test2-Scene-001.pt"
tmp = torch.load(audio_file_path, map_location="cpu", weights_only=True)
print("audio latents infomation:", tmp["latents"].shape, tmp['num_time_steps'], tmp['frequency_bins'], tmp["duration"])


video_file_path = "/root/autodl-tmp/data/.ltx2_precomputed/latents/scenes_clip_official_v2/test-Scene-003.pt"
# video_file_path = "/root/autodl-tmp/data/.ltx2_precomputed/latents/scenes_clip_official_v2/test2-Scene-001.pt"
tmp = torch.load(video_file_path, map_location="cpu", weights_only=True)
print("video latents infomation:", tmp["latents"].shape, tmp['num_frames'], tmp['height'], tmp["width"], tmp["fps"])