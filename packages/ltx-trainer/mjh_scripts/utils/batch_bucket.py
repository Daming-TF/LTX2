from einops import rearrange
import torch
from torch.utils.data import RandomSampler, BatchSampler, Dataset, Sampler
from typing import Iterator, Sequence
import pdb
from pathlib import Path
# import sys
# sys.path.append(str(Path(__file__).parent.parent.parent / "src"))
# from ltx_trainer.datasets import PrecomputedDataset


class AspectRatioBatchSampler(BatchSampler):
    def __init__(self,
            sampler: Sampler,
            dataset: Dataset,
            batch_size: int,
            aspect_ratios: dict,
            ratio_nums: dict = None,
            drop_last: bool = False,
            valid_num=0,   # take as valid aspect-ratio when sample number >= valid_num
            **kwargs) -> None:
        """
        Args:
            sampler (Sampler): Base sampler.
            dataset (Dataset): Dataset to sample from.
            batch_size (int): Size of mini-batch.
            aspect_ratios (dict): A dict of aspect ratios(Each element represents W*H*F), 
                e.g., {"1.78-97":[512,288,97],"0.55-65":[352,352,65], ...}   
        """
        if not isinstance(sampler, Sampler):
            raise TypeError('sampler should be an instance of ``Sampler``, '
                            f'but got {sampler}')
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError('batch_size should be a positive integer value, '
                             f'but got batch_size={batch_size}')
        self.sampler = sampler
        self.dataset = dataset
        self.batch_size = batch_size
        self.aspect_ratios = aspect_ratios
        self.drop_last = drop_last
        self.ratio_nums_gt = ratio_nums
        # buckets for each aspect ratio
        self._aspect_ratio_buckets = {ratio: [] for ratio in aspect_ratios.keys()}
        self.current_available_bucket_keys =  [str(k) for k, v in self.ratio_nums_gt.items() if v >= valid_num]

    def __iter__(self) -> Iterator[list[int]]:
        for idx in self.sampler:
            idx_bucket = self.dataset[idx]['bucket']
            if idx_bucket not in self.current_available_bucket_keys:
                raise ValueError(f"Bucket {idx_bucket} is not in the current available buckets: {self.current_available_bucket_keys}")
            bucket = self._aspect_ratio_buckets[idx_bucket]
            bucket.append(idx)
            # yield a batch of indices in the same aspect ratio group
            if len(bucket) == self.batch_size:
                yield bucket[:]
                del bucket[:]

        # yield the rest data and reset the buckets
        for bucket in self._aspect_ratio_buckets.values():
            while len(bucket) > 0:
                if len(bucket) <= self.batch_size:
                    if not self.drop_last:
                        yield bucket[:]
                    del bucket[:]


def _normalize_video_latents(data: dict) -> dict:
        """
        Normalize video latents to non-patchified format [C, F, H, W].
        Used for keeping backward compatibility with legacy datasets.
        """
        latents = data["latents"]

        # Check if latents are in legacy patchified format [seq_len, C]
        if latents.dim() == 2:
            # Legacy format: [seq_len, C] where seq_len = F * H * W
            num_frames = data["num_frames"]
            height = data["height"]
            width = data["width"]

            # Unpatchify: [seq_len, C] -> [C, F, H, W]
            latents = rearrange(
                latents,
                "(f h w) c -> c f h w",
                f=num_frames,
                h=height,
                w=width,
            )

            # Update the data dict with unpatchified latents
            data = data.copy()
            data["latents"] = latents

        return data


def prepare_train_dataset(dataset, accelerator, data_sources, aspect_ratios=None):
    def preprocess_train(examples):
        package = {}
        for i in range(len(examples["bucket"])):
            for dir_name, output_key in data_sources.items():
                try:
                    latent_tmp = torch.load(examples[output_key][i], map_location="cpu", weights_only=True)
                    # Normalize latent(video & audio) format if this is a latent source
                    if "latent" in dir_name.lower(): latent_tmp = _normalize_video_latents(latent_tmp)
                    assert isinstance(latent_tmp, dict), f"Expected loaded data to be a dict, but got {type(latent_tmp)}."
                    if dir_name=="latents": dir_name = "video_latents"
                    for k, v in latent_tmp.items():
                        package.setdefault(f"{dir_name[0]}_{k}", [])
                        package[f"{dir_name[0]}_{k}"].append(v)
                except Exception as e:
                    raise RuntimeError(f"Failed to load {output_key} from {examples[output_key][i]}: {e}") from e
        package['bucket'] = examples['bucket']
        # Keep original sample indices for easier debugging after batching.
        if 'sample_idx' in examples:
            package['sample_idx'] = examples['sample_idx']
        # pdb.set_trace()
        return package
    
    with accelerator.main_process_first():
        dataset = dataset.with_transform(preprocess_train)
    return dataset


def collate_fn(batch):
    package = {}
    for _batch in batch:
        for k, v in _batch.items():
            package.setdefault(k, [])
            package[k].append(v)
    for k, v in package.items():
        if isinstance(v[0], torch.Tensor):
            package[k] = torch.stack(v)
        # elif k == 'sample_idx':
        #     package[k] = torch.tensor(v, dtype=torch.long)
        elif not isinstance(v[0], str):
            package[k] = torch.tensor(v)
        else:
            package[k] = v
    return package