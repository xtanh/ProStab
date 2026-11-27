"""Shared featurization, adapted from SPURS; no dataset loading dependencies."""

from collections import defaultdict
import numpy as np
import torch
from .batch import CoordBatchConverter
from .data_utils import Alphabet


class Featurizer(object):
    def __init__(
        self,
        alphabet: Alphabet,
        to_pifold_format=False,
        coord_nan_to_zero=True,
        atoms=("N", "CA", "C", "O"),
        single_mut=False,
        mut_seq=False,
    ):
        self.alphabet = alphabet
        self.batcher = CoordBatchConverter(
            alphabet=alphabet,
            coord_pad_inf=alphabet.add_special_tokens,
            to_pifold_format=to_pifold_format,
            coord_nan_to_zero=coord_nan_to_zero,
        )
        self.single_mut = single_mut
        self.atoms = atoms
        self.cache = defaultdict(lambda: -1)
        self.mut_seq = mut_seq

    def __call__(self, raw_batch: dict):

        if not self.single_mut:

            raw_batch = raw_batch[0]
            wt_seq = [raw_batch["seq"]]
            wt_coords = [
                np.stack([raw_batch["coords"][atom] for atom in self.atoms], 1)
            ]

            coords, confidence, strs, tokens, lengths, coord_mask = (
                self.batcher.from_lists(
                    coords_list=wt_coords, confidence_list=None, seq_list=wt_seq
                )
            )
            mut_tokens_list = []
            for mut_seq in raw_batch["mut_seq"]:
                _, _, _, mut_tokens, _, _ = self.batcher.from_lists(
                    coords_list=wt_coords, confidence_list=None, seq_list=[mut_seq]
                )
                mut_tokens_list.append(mut_tokens)

            raw_batch["tokens"] = tokens
            raw_batch["mut_tokens"] = mut_tokens_list
            raw_batch["ddG"] = raw_batch["ddG"].reshape(-1)
            return raw_batch

        for protein in raw_batch:
            name = protein["name"]
            if isinstance(self.cache[name], int):
                seqs = [protein["seq"]]
                coords = [np.stack([protein["coords"][atom] for atom in self.atoms], 1)]

                coords, confidence, strs, tokens, lengths, coord_mask = (
                    self.batcher.from_lists(
                        coords_list=coords, confidence_list=None, seq_list=seqs
                    )
                )
                self.cache[name] = {"tokens": tokens, "mut_tokens": None}
            else:
                tokens = self.cache[name]["tokens"]
                mut_tokens = self.cache[name]["mut_tokens"]
            protein["tokens"] = tokens
            protein["mut_tokens"] = None

        ddg = torch.stack([protein["ddG"] for protein in raw_batch])
        return {
            "raw_batch": raw_batch,
            "mut_ids": [protein["mut_ids"] for protein in raw_batch],
            "append_tensors": torch.stack(
                [protein["append_tensors"] for protein in raw_batch]
            ),
            "ddG": ddg,
            "name": [protein["name"] + protein["chain_ids"] for protein in raw_batch],
            "dataset": [protein["dataset"] for protein in raw_batch],
        }
