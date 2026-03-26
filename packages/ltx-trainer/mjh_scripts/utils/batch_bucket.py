from torch.utils.data import RandomSampler, BatchSampler, Dataset, Sampler
from typing import Iterator, Sequence

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