import random
from torch.utils.data import Sampler, RandomSampler, SequentialSampler


class TokenSampler(Sampler):
    """
    A Sampler which groups sequences into buckets based on length and constructs batches using 
    a (potentially) different number of sequences from each bucket to achieve a target number of 
    tokens in each batch. This approach has a number of advantages:
        - Faster training and eval since there are fewer pad tokens vs random batching
        - Potentially improved training stability since the number of tokens is approx the same
          each batch

    Note: There is a systematic error in the batch size (it will be slightly larger than the 
          target size on average) since we simply take the mean of the seq lengths in the bucket,
          this does not account for padding that will result from the largest seq in the batch.
    """

    def __init__(
        self,
        num_buckets,
        seq_lengths,
        batch_size,
        shuffle=True,
        drop_last=True
    ):
        """ Init method

        Args:
            num_buckets (int): Number of buckets to split sequences into
            seq_lengths (List[int]): The length of the sequences in the dataset (in the same order)
            batch_size (int): Target number of tokens in each batch
            shuffle (Optional[bool]): Shuffle the indices within each bucket
            drop_last (Optional[bool]): Forget about the indices remaining at the end of each bucket
        """

        if not drop_last:
            raise NotImplementedError("Keeping last elements is not yet supported")

        min_length = min(seq_lengths)
        max_length = max(seq_lengths) + 1
        bucket_width = (max_length - min_length) / num_buckets

        bucket_limits = []
        lower_limit = float(min_length)

        # Setup lower (inclusive) and upper (exclusive) seq length limits on buckets
        for _ in range(num_buckets):
            upper_limit = lower_limit + bucket_width
            bucket_limits.append((lower_limit, upper_limit))
            lower_limit = upper_limit

        buckets = [[] for _ in range(num_buckets)]
        lengths = [[] for _ in range(num_buckets)]

        # Add indices to correct bucket based on seq length
        for seq_idx, length in enumerate(seq_lengths):
            for b_idx, (lower, upper) in enumerate(bucket_limits):
                if lower <= length < upper:
                    buckets[b_idx].append(seq_idx)
                    lengths[b_idx].append(length)

        if shuffle:
            samplers = [RandomSampler(idxs) for idxs in buckets]
        else:
            samplers = [SequentialSampler(idxs) for idxs in buckets]

        # Work out approx number of sequences required for each bucket
        avg_lengths = [sum(ls) // len(ls) for ls in lengths]
        num_seqs = [batch_size // length for length in avg_lengths]
        num_seqs = [int(num_sq) for num_sq in num_seqs]

        num_batches = [len(bucket) // num_seqs[b_idx] for b_idx, bucket in enumerate(buckets)]
        num_batches = [int(num_bs) for num_bs in num_batches]

        self.num_seqs = num_seqs
        self.buckets = buckets
        self.num_batches = num_batches
        self.samplers = samplers

    def __iter__(self):
        iters = [iter(sampler) for sampler in self.samplers]
        rem_batches = self.num_batches[:]
        while sum(rem_batches) > 0:
            b_idx = random.choices(range(len(rem_batches)), weights=rem_batches, k=1)[0]
            batch_idxs = [next(iters[b_idx]) for _ in range(self.num_seqs[b_idx])]
            batch = [self.buckets[b_idx][idx] for idx in batch_idxs]
            rem_batches[b_idx] -= 1
            yield batch

    def __len__(self):
        return sum(self.num_batches)
