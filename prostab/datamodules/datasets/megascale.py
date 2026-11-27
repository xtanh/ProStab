"""MegaScale datasets, adapted from SPURS.

https://github.com/luo-group/SPURS/blob/9cf686eb8304740775c4cfdd2437d36a96e97732/spurs/datamodules/datasets/megascale.py
"""
import json
import os
import pickle

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from prostab import utils
from .ddgbench import ddgBenchDataset
from .featurizer import Featurizer  # Legacy import path.
from .utils import get_pdb, parse_pdb

log = utils.get_logger(__name__)
ALPAHBET = "ACDEFGHIKLMNPQRSTVWYX"


class MegaScaleDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        reduce: str = "",
        split: str = "train",
        single_mut: bool = False,
        mut_seq: bool = False,
        std_ratio: float = 0.75,
        loss_ratio: float = 1.0,
        train_ratio: float = 0.05,
    ):
        self.mut_seq = mut_seq
        self.split = split
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_path = os.path.join(current_dir, "../../../")
        fname = os.path.join(
            root_path,
            "data/dataset/megascale/Tsuboyama2023_Dataset2_Dataset3_20230416.csv",
        )
        df = pd.read_csv(
            fname, usecols=["ddG_ML", "mut_type", "WT_name", "aa_seq", "dG_ML"]
        )
        # remove unreliable data and more complicated mutations
        df = df.loc[df.ddG_ML != "-", :].reset_index(drop=True)
        df = df.loc[
            ~df.mut_type.str.contains("ins")
            & ~df.mut_type.str.contains("del")
            & ~df.mut_type.str.contains(":"),
            :,
        ].reset_index(drop=True)

        self.df = df
        if self.split != "test":
            mmseq_wt_search = os.path.join(
                root_path, "data/dataset/megascale/mmseq_mut_search_0.25.m8"
            )
            ret = []
            with open(mmseq_wt_search, "r") as f:
                for line in f.readlines():
                    second_column_value = int(line.split("\t")[1])
                    ret.append(second_column_value)
            # we dont want the rows in the ret
            previous_len = len(df)
            df = df.loc[~df.index.isin(ret), :].reset_index(drop=True)
            cur_len = len(df)
            log.info(f"removed {previous_len - cur_len} rows from the dataset")

        # load splits produced by mmseqs clustering
        with open(
            os.path.join(root_path, "data/dataset/megascale/mega_splits.pkl"), "rb"
        ) as f:
            splits = pickle.load(
                f
            )  # this is a dict with keys train/val/test and items holding FULL PDB names for a given split

        self.split_wt_names = {
            "val": [],
            "test": [],
            "train": [],
            "train_s669": [],
            "all": [],
            "cv_train_0": [],
            "cv_train_1": [],
            "cv_train_2": [],
            "cv_train_3": [],
            "cv_train_4": [],
            "cv_val_0": [],
            "cv_val_1": [],
            "cv_val_2": [],
            "cv_val_3": [],
            "cv_val_4": [],
            "cv_test_0": [],
            "cv_test_1": [],
            "cv_test_2": [],
            "cv_test_3": [],
            "cv_test_4": [],
        }

        self.wt_seqs = {}
        self.mut_rows = {}
        self.std_ratio = std_ratio
        self.loss_ratio = loss_ratio
        self.train_ratio = train_ratio
        if self.split == "all":
            all_names = np.concatenate([splits["train"], splits["val"], splits["test"]])
            self.split_wt_names[self.split] = all_names
        else:
            if reduce == "prot" and split == "train":
                n_prots_reduced = 58
                self.split_wt_names[self.split] = np.random.choice(
                    splits["train"], n_prots_reduced
                )
            else:
                self.split_wt_names[self.split] = splits[self.split]
        self.wt_names = self.split_wt_names[self.split]

        removed_wt_names = []
        for wt_name in tqdm(self.wt_names):
            wt_rows = df.query('WT_name == @wt_name and mut_type == "wt"').reset_index(
                drop=True
            )
            self.mut_rows[wt_name] = df.query(
                'WT_name == @wt_name and mut_type != "wt"'
            ).reset_index(drop=True)
            if type(reduce) is float and self.split == "train":
                self.mut_rows[wt_name] = self.mut_rows[wt_name].sample(
                    frac=float(reduce), replace=False
                )
            if len(wt_rows) == 0:
                removed_wt_names.append(wt_name)
            else:
                self.wt_seqs[wt_name] = wt_rows.aa_seq[0]
        previous_len = len(self.wt_names)
        self.wt_names = list(set(self.wt_names) - set(removed_wt_names))
        cur_len = len(self.wt_names)
        log.info(f"removed {previous_len - cur_len} wt names from the dataset")

        structure_path = os.path.join(
            root_path, "data/dataset/megascale/AlphaFold_model_PDBs/"
        )
        structure_path_json = os.path.join(structure_path, "../parsed_structure.json")

        self.structure_path = structure_path

        parse_pdb(structure_path, structure_path_json)

        log.info("loading structure dataset")
        with open(structure_path_json, "r") as file:
            self.json_dataset = json.load(file)

    def cal_index2mt(self, index):
        return self.index2mt[index]

    def __len__(self):

        return len(self.wt_names)

    def _get_wt_item(self, index):

        wt_name = self.wt_names[index]
        wt_seq = self.wt_seqs[wt_name]
        mut_data = self.mut_rows[wt_name]

        wt_name = wt_name.split(".pdb")[0].replace("|", ":")

        pdb = self.json_dataset[wt_name]
        protein = get_pdb(pdb, wt_seq, wt_name)

        if self.mut_seq:
            protein["S"] = [protein["S"]]

        dataset_name = []
        for i in range(len(mut_data)):
            mut_seq = mut_data.iloc[i]

            if self.mut_seq:
                pdb["seq"] = mut_seq.aa_seq
                mut_protein = get_pdb(pdb, mut_seq.aa_seq, wt_name)
                protein["S"].append(mut_protein["S"])

            if (
                "ins" in mut_seq.mut_type
                or "del" in mut_seq.mut_type
                or ":" in mut_seq.mut_type
            ):
                return None

            assert len(mut_seq.aa_seq) == len(wt_seq)
            wt = mut_seq.mut_type[0]
            mut = mut_seq.mut_type[-1]
            mut_id = int(mut_seq.mut_type[1:-1]) - 1
            assert wt_seq[mut_id] == wt
            assert mut_seq.aa_seq[mut_id] == mut

            if mut_seq.ddG_ML == "-":
                return None
            ddG = -torch.tensor([float(mut_seq.ddG_ML)], dtype=torch.float32)

            wt_onehot = torch.zeros((21))
            wt_onehot[ALPAHBET.index(wt)] = 1
            mt_onehot = torch.zeros((21))
            mt_onehot[ALPAHBET.index(mut)] = 1
            append_tensor = torch.cat([wt_onehot, mt_onehot])
            append_tensor = append_tensor.float()

            protein["mut_ids"].append(mut_id)
            protein["ddG"].append(ddG)
            protein["append_tensors"].append(append_tensor)
            protein["mut_seq"].append(mut_seq.aa_seq)

        protein["ddG"] = torch.stack(protein["ddG"]).to(
            protein["X"].device, non_blocking=True
        )
        protein["append_tensors"] = torch.stack(protein["append_tensors"])
        protein["dataset"] = "megascale"
        protein["pdb_path"] = self.structure_path

        if self.mut_seq:
            protein["S"] = torch.cat(protein["S"], dim=0).clone()
            protein["X"] = protein["X"].expand(len(protein["S"]), -1, -1, -1).clone()
            protein["mask"] = protein["mask"].expand(len(protein["S"]), -1).clone()
            protein["chain_M"] = (
                protein["chain_M"].expand(len(protein["S"]), -1).clone()
            )
            protein["chain_M_chain_M_pos"] = (
                protein["chain_M_chain_M_pos"].expand(len(protein["S"]), -1).clone()
            )
            protein["residue_idx"] = (
                protein["residue_idx"].expand(len(protein["S"]), -1).clone()
            )
            protein["chain_encoding_all"] = (
                protein["chain_encoding_all"].expand(len(protein["S"]), -1).clone()
            )
            protein["randn_1"] = (
                protein["randn_1"].expand(len(protein["S"]), -1).clone()
            )

        protein["std_ratio"] = self.std_ratio
        protein["loss_ratio"] = self.loss_ratio
        return protein

    def __getitem__(self, index):

        return self._get_wt_item(index)

    def collect_func(self, batch):
        return batch[0]


class MegaScaleTestDatasets(torch.utils.data.Dataset):
    def __init__(
        self,
    ):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_path = os.path.join(current_dir, "../../../")

        self.s669 = ddgBenchDataset(
            pdb_dir=os.path.join(root_path, "data/dataset/S669/pdb"),
            csv_fname=os.path.join(root_path, "data/dataset/S669/s669_clean_dir.csv"),
            dataset_name="S669",
        )

    def __len__(self):

        return self.s669.__len__()

    def __getitem__(self, index):

        if index < self.s669.__len__():
            return self.s669.__getitem__(index)
        index -= self.s669.__len__()
