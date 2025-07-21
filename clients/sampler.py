import random
import numpy as np
from torch.utils.data import Sampler
from collections import Counter

class AdaptiveDrainSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

        self.buckets = {
            'Organ': dataset.organ_slices[:],
            'NonOrgan': dataset.non_organ_slices[:]
        }
        self.original_pools = {k: v[:] for k, v in self.buckets.items()}
        self.length = len(dataset) // batch_size
        self._shuffle_all()

    def _shuffle_all(self):
        for k in self.buckets:
            random.shuffle(self.buckets[k])

    def _draw(self, category, n):
        samples = []
        while n > 0:
            if not self.buckets[category]:
                self.buckets[category] = self.original_pools[category][:]
                random.shuffle(self.buckets[category])
            take = min(n, len(self.buckets[category]))
            samples.extend(self.buckets[category][:take])
            self.buckets[category] = self.buckets[category][take:]
            n -= take
        return samples

    def _label_counts(self, indices):
        label_counter = Counter()
        for idx in indices:
            if idx in self.dataset.organ_slices:
                label_counter['Organ'] += 1
            elif idx in self.dataset.non_organ_slices:
                label_counter['NonOrgan'] += 1
        return label_counter

    def __iter__(self):
        self._shuffle_all()

        total_organ = len(self.original_pools['Organ'])
        total_non = len(self.original_pools['NonOrgan'])
        weights = np.array([total_organ, total_non], dtype=np.float32)
        weights /= weights.sum()

        for _ in range(self.length):
            counts = np.random.multinomial(self.batch_size, weights)
            batch = []

            batch.extend(self._draw('Organ', counts[0]))
            batch.extend(self._draw('NonOrgan', counts[1]))

            # Ensure at least 1 Organ and 1 NonOrgan per batch
            label_map = self._label_counts(batch)
            if label_map['Organ'] < 1:
                batch.pop()
                batch += self._draw('Organ', 1)
            if label_map['NonOrgan'] < 1:
                batch.pop()
                batch += self._draw('NonOrgan', 1)

            random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.length