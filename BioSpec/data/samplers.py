import copy
import math
import random
import numpy as np
from torch import distributed as dist
from collections import defaultdict
from torch.utils.data.sampler import Sampler


class RandomIdentitySampler(Sampler):
    def __init__(self, data_source, num_instances=4):
        self.data_source = data_source
        self.num_instances = num_instances
        self.index_dic = defaultdict(list)
        for index, (_, pid, _, _) in enumerate(data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        self.num_identities = len(self.pids)

        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        list_container = []

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    list_container.append(batch_idxs)
                    batch_idxs = []

        random.shuffle(list_container)

        ret = []
        for batch_idxs in list_container:
            ret.extend(batch_idxs)

        return iter(ret)

    def __len__(self):
        return self.length


class DistributedRandomIdentitySampler(Sampler):
    def __init__(self, data_source, num_instances=4, 
                 num_replicas=None, rank=None, seed=0):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0

        self.data_source = data_source
        self.num_instances = num_instances
        self.index_dic = defaultdict(list)
        for index,data in enumerate(data_source):
            pid=data['pid']
                
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        self.num_identities = len(self.pids)

        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances
        assert self.length % self.num_instances == 0

        if self.length // self.num_instances % self.num_replicas != 0: 
            self.num_samples = math.ceil((self.length // self.num_instances - self.num_replicas) / self.num_replicas) * self.num_instances
        else:
            self.num_samples = math.ceil(self.length / self.num_replicas) 
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        random.seed(self.seed + self.epoch)
        np.random.seed(self.seed + self.epoch)

        list_container = []
        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    list_container.append(batch_idxs)
                    batch_idxs = []
        random.shuffle(list_container)

        list_container = list_container[:self.total_size//self.num_instances]
        assert len(list_container) == self.total_size//self.num_instances

        list_container = list_container[self.rank:self.total_size//self.num_instances:self.num_replicas]
        assert len(list_container) == self.num_samples//self.num_instances

        ret = []
        for batch_idxs in list_container:
            ret.extend(batch_idxs)

        return iter(ret)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class DistributedInferenceSampler(Sampler):
    """
    refer to: https://github.com/huggingface/transformers/blob/447808c85f0e6d6b0aeeb07214942bf1e578f9d2/src/transformers/trainer_pt_utils.py

    Distributed Sampler that subsamples indicies sequentially,
    making it easier to collate all results at the end.
    Even though we only use this sampler for eval and predict (no training),
    which means that the model params won't have to be synced (i.e. will not hang
    for synchronization even if varied number of forward passes), we still add extra
    samples to the sampler to make it evenly divisible (like in `DistributedSampler`)
    to make it easy to `gather` or `reduce` resulting tensors at the end of the loop.
    """
    def __init__(self, dataset, rank=None, num_replicas=None):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank

        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        indices += [indices[-1]] * (self.total_size - len(indices))
        indices = indices[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
        return iter(indices)

    def __len__(self):
        return self.num_samples