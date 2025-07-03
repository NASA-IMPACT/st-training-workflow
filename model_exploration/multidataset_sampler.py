import math
from collections import Counter
from itertools import accumulate, cycle
from typing import Iterator

import torch
from sentence_transformers.sampler import (
    MultiDatasetDefaultBatchSampler,
    ProportionalBatchSampler,
    RoundRobinBatchSampler,
)
from torch.utils.data import (
    BatchSampler,
    ConcatDataset,
    SequentialSampler,
    SubsetRandomSampler,
    TensorDataset,
)


class WeightedBatchSampler(MultiDatasetDefaultBatchSampler):
    """
    Batch sampler that samples from each dataset based on a specified weight.
    This allows for over-sampling from smaller or more important datasets and
    under-sampling from larger or less critical ones.
    """

    def __init__(
        self,
        dataset,
        batch_samplers,
        dataset_configs: dict[str, dict],
        generator: torch.Generator = None,
        seed: int = 0,
    ):
        __name__ = "WeightedBatchSampler"
        super().__init__(dataset, batch_samplers, generator, seed)
        weights = [config.get("weight", 1.0) for config in dataset_configs.values()]

        # Pre-calculate effective sizes using rounding for more intuitive behavior
        self.effective_sizes = [
            round(len(sampler) * weights[i])
            for i, sampler in enumerate(self.batch_samplers)
        ]
        # Pre-calculate total length for efficiency
        self.total_batches = sum(self.effective_sizes)

    def __iter__(self) -> Iterator[list[int]]:
        # print("--- WeightedBatchSampler Iteration ---")
        self.generator.manual_seed(self.seed + self.epoch)

        dataset_indices = []
        # Use the pre-calculated, rounded sizes
        for i, size in enumerate(self.effective_sizes):
            dataset_indices.extend([i] * size)

        # Shuffle the dataset indices to mix the order in which datasets are sampled
        dataset_idx_sampler = SubsetRandomSampler(
            dataset_indices,
            generator=self.generator,
        )
        sample_offsets = [0] + list(accumulate(len(ds) for ds in self.dataset.datasets))
        batch_samplers_iters = [iter(sampler) for sampler in self.batch_samplers]

        for dataset_idx in dataset_idx_sampler:
            sample_offset = sample_offsets[dataset_idx]
            try:
                yield [
                    idx + sample_offset
                    for idx in next(batch_samplers_iters[dataset_idx])
                ]
            except StopIteration:
                batch_samplers_iters[dataset_idx] = iter(
                    self.batch_samplers[dataset_idx],
                )
                yield [
                    idx + sample_offset
                    for idx in next(batch_samplers_iters[dataset_idx])
                ]

    def __len__(self) -> int:
        # Return the consistent, pre-calculated total number of batches
        return self.total_batches


# --- Demo Runner ---
def run_sampler_test(sampler, sampler_name):
    """A helper function to run and print results for any given sampler."""
    print(f"--- Testing: {sampler_name} ---")
    if hasattr(sampler, "weights"):
        print(f"Weights: {sampler.weights}")
    if hasattr(sampler, "effective_sizes"):
        print(f"Effective (calculated) batch counts: {sampler.effective_sizes}")
    print(f"Total batches per epoch: {len(sampler)}\n")

    batches_yielded = 0
    dataset_origin_counts = Counter()
    sample_offsets = [0] + list(accumulate(len(ds) for ds in sampler.dataset.datasets))

    print("First 5 batches yielded:")
    for i, batch in enumerate(sampler):
        batches_yielded += 1
        first_index = batch[0]

        origin_dataset_idx = 0
        # Determine which dataset the batch came from based on the index
        for j in range(1, len(sample_offsets)):
            if first_index < sample_offsets[j]:
                origin_dataset_idx = j - 1
                break
        else:  # Handle last dataset
            origin_dataset_idx = len(sample_offsets) - 2

        dataset_origin_counts[origin_dataset_idx] += 1
        if i < 5:
            print(
                f"  Batch {i+1:02d}: (from Dataset {chr(65+origin_dataset_idx)}) {batch}",
            )

    print("\n--- Verification ---")
    print(f"Sampler's reported length (__len__): {len(sampler)}")
    print(f"Actual batches yielded in the loop:   {batches_yielded}")

    print("\nActual batch counts from the loop:")
    for i in range(len(sampler.batch_samplers)):
        print(f"  - Dataset {chr(65+i)}: {dataset_origin_counts[i]}")

    print("-" * 35 + "\n")


# --- Main Demo Code ---
if __name__ == "__main__":
    print("--- Setting up Demo ---")
    BATCH_SIZE = 4

    # 1. Create mock datasets
    dataset_A = TensorDataset(torch.arange(0, 60))  # 15 batches
    dataset_B = TensorDataset(torch.arange(60, 90))  # 8 batches
    dataset_C = TensorDataset(torch.arange(90, 100))  # 3 batches
    concat_dataset = ConcatDataset([dataset_A, dataset_B, dataset_C])

    print(
        f"Dataset A: {len(dataset_A)} samples -> {math.ceil(len(dataset_A)/BATCH_SIZE)} batches",
    )
    print(
        f"Dataset B: {len(dataset_B)} samples -> {math.ceil(len(dataset_B)/BATCH_SIZE)} batches",
    )
    print(
        f"Dataset C: {len(dataset_C)} samples -> {math.ceil(len(dataset_C)/BATCH_SIZE)} batches",
    )
    print("-" * 35 + "\n")

    # 2. Create base samplers
    sampler_A = BatchSampler(
        SequentialSampler(dataset_A),
        batch_size=BATCH_SIZE,
        drop_last=False,
    )
    sampler_B = BatchSampler(
        SequentialSampler(dataset_B),
        batch_size=BATCH_SIZE,
        drop_last=False,
    )
    sampler_C = BatchSampler(
        SequentialSampler(dataset_C),
        batch_size=BATCH_SIZE,
        drop_last=False,
    )
    base_samplers = [sampler_A, sampler_B, sampler_C]

    # 3. Test each sampler
    generator = torch.Generator()

    # Test 1: Your WeightedBatchSampler
    dataset_configs = {
        "A": {"weight": 0.5},
        "B": {"weight": 1.0},
        "C": {"weight": 3.0},
    }
    weighted_sampler = WeightedBatchSampler(
        concat_dataset,
        base_samplers,
        dataset_configs,
        generator,
        42,
    )
    run_sampler_test(weighted_sampler, "Weighted Sampler (Your Implementation)")

    # Test 2: ProportionalBatchSampler from sentence-transformers
    proportional_sampler = ProportionalBatchSampler(
        concat_dataset,
        base_samplers,
        generator,
        42,
    )
    run_sampler_test(proportional_sampler, "Proportional Sampler (sbert)")

    # Test 3: RoundRobinBatchSampler from sentence-transformers
    round_robin_sampler = RoundRobinBatchSampler(
        concat_dataset,
        base_samplers,
        generator,
        42,
    )
    run_sampler_test(round_robin_sampler, "Round-Robin Sampler (sbert)")
