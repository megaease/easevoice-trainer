# modified from https://github.com/yangdongchao/SoundStorm/blob/master/soundstorm/s1/AR/data/dataset.py
# reference: https://github.com/lifeiteng/vall-e
import traceback
import os
from typing import Dict, Optional
from typing import List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from ....text import cleaned_text_to_sequence


def batch_sequences(sequences: List[np.ndarray], axis: int = 0, pad_value: int = 0):
    seq = sequences[0]
    ndim = seq.ndim
    if axis < 0:
        axis += ndim
    dtype = seq.dtype
    pad_value = dtype.type(pad_value)
    seq_lengths = [seq.shape[axis] for seq in sequences]
    max_length = np.max(seq_lengths)

    padded_sequences = []
    for seq, length in zip(sequences, seq_lengths):
        padding = (
            [(0, 0)] * axis + [(0, max_length - length)] + [(0, 0)] * (ndim - axis - 1)
        )
        padded_seq = np.pad(seq, padding, mode="constant", constant_values=pad_value)
        padded_sequences.append(padded_seq)
    batch = np.stack(padded_sequences)
    return batch


class Text2SemanticDataset(Dataset):
    """dataset class for text tokens to semantic model training."""

    def __init__(
        self,
        phoneme_path: str,
        semantic_path: str,
        max_sample: Optional[int] = None,
        max_sec: int = 100,
        pad_val: int = 1024,
        # min value of phoneme/sec
        min_ps_ratio: int = 3,
        # max value of phoneme/sec
        max_ps_ratio: int = 25,
    ) -> None:
        super().__init__()

        self.semantic_data = pd.read_csv(
            semantic_path, delimiter="\t", encoding="utf-8"
        )
        self._semantic_path = semantic_path
        self._phoneme_path = phoneme_path
        # get dict
        self.path2 = phoneme_path  # "%s/2-name2text.txt"%exp_dir#phoneme_path
        self.path3 = "%s/3-bert" % (
            os.path.dirname(phoneme_path)
        )  # "%s/3-bert"%exp_dir#bert_dir
        self.path6 = semantic_path  # "%s/6-name2semantic.tsv"%exp_dir#semantic_path
        assert os.path.exists(self.path2)
        assert os.path.exists(self.path6)
        self.phoneme_data = {}
        with open(self.path2, "r", encoding="utf8") as f:
            lines = f.read().strip("\n").split("\n")

        for line in lines:
            tmp = line.split("\t")
            if len(tmp) != 4:
                continue
            self.phoneme_data[tmp[0]] = [tmp[1], tmp[2], tmp[3]]

        # pad for semantic tokens
        self.PAD: int = pad_val
        self.hz = int(os.environ.get("hz", "25hz")[:-2])

        # max seconds of semantic token
        self.max_sec = max_sec
        self.min_ps_ratio = min_ps_ratio
        self.max_ps_ratio = max_ps_ratio

        if max_sample is not None:
            self.semantic_data = self.semantic_data[:max_sample]

        # {idx: (semantic, phoneme)}
        # semantic list, phoneme list
        self.semantic_phoneme = []
        self.item_names = []

        self.inited = False

        if not self.inited:
            self.init_batch()
            self.inited = True
            del self.semantic_data
            del self.phoneme_data

    def init_batch(self):
        semantic_data_len = len(self.semantic_data)
        phoneme_data_len = len(self.phoneme_data.keys())
        print("semantic_data_len:", semantic_data_len)
        print("phoneme_data_len:", phoneme_data_len)
        print(self.semantic_data)
        idx = 0
        num_not_in = 0
        num_deleted_bigger = 0
        num_deleted_ps = 0
        for i in range(semantic_data_len):
            item_name = self.semantic_data.iloc[i, 0]
            try:
                phoneme, word2ph, text = self.phoneme_data[item_name]
            except Exception:
                traceback.print_exc()
                num_not_in += 1
                continue

            semantic_str = self.semantic_data.iloc[i, 1]
            # get token list
            semantic_ids = [int(idx) for idx in semantic_str.split(" ")]  # pyright: ignore
            # (T) -> (1, T)? No, we need len
            # filter too long samples
            if (
                len(semantic_ids) > self.max_sec * self.hz
            ):  # based on tokens to predict total 60s
                num_deleted_bigger += 1
                continue
            phoneme = phoneme.split(" ")

            try:
                phoneme_ids = cleaned_text_to_sequence(phoneme)
            except:
                traceback.print_exc()
                num_not_in += 1
                continue
            # if len(phoneme_ids) >400: change it to semantic/2.5
            if (
                len(phoneme_ids) > self.max_sec * self.hz / 2.5
            ):
                num_deleted_ps += 1
                continue

            ps_ratio = len(phoneme_ids) / (len(semantic_ids) / self.hz)

            if (
                ps_ratio > self.max_ps_ratio or ps_ratio < self.min_ps_ratio
            ):
                num_deleted_ps += 1
                continue

            self.semantic_phoneme.append((semantic_ids, phoneme_ids))
            idx += 1
            self.item_names.append(item_name)

        min_num = 100
        leng = len(self.semantic_phoneme)
        if leng == 0:
            raise ValueError(f"no valid data in {self._semantic_path}, please check the data and try again")

        if leng < min_num:
            tmp1 = self.semantic_phoneme
            tmp2 = self.item_names
            self.semantic_phoneme = []
            self.item_names = []
            for _ in range(max(2, int(min_num / leng))):
                self.semantic_phoneme += tmp1
                self.item_names += tmp2
        if num_not_in > 0:
            print(f"there are {num_not_in} semantic datas not in phoneme datas")
        if num_deleted_bigger > 0:
            print(
                f"deleted {num_deleted_bigger} audios who's duration are bigger than {self.max_sec} seconds"
            )
        if num_deleted_ps > 0:
            print(
                f"deleted {num_deleted_ps} audios who's phoneme/sec are bigger than {self.max_ps_ratio} or smaller than {self.min_ps_ratio}"
            )
        """
        there are 31 semantic datas not in phoneme datas
        deleted 34 audios who's duration are bigger than 54 seconds
        deleted 3190 audios who's phoneme/sec are bigger than 25 or smaller than 3
        dataset.__len__(): 366463

        """
        # 345410 for LibriTTS
        print("dataset.__len__():", self.__len__())

    def __get_item_names__(self) -> List[str]:
        return self.item_names

    def __len__(self) -> int:
        return len(self.semantic_phoneme)

    def __getitem__(self, idx: int) -> Dict:
        semantic_ids, phoneme_ids = self.semantic_phoneme[idx]
        item_name = self.item_names[idx]
        phoneme_ids_len = len(phoneme_ids)
        # semantic tokens target
        semantic_ids_len = len(semantic_ids)

        flag = 0
        path_bert = "%s/%s.pt" % (self.path3, item_name)
        if os.path.exists(path_bert) == True:
            bert_feature = torch.load(path_bert, map_location="cpu")
            assert bert_feature.shape[-1] == len(phoneme_ids)
        else:
            bert_feature = None
        return {
            "idx": idx,
            "phoneme_ids": phoneme_ids,
            "phoneme_ids_len": phoneme_ids_len,
            "semantic_ids": semantic_ids,
            "semantic_ids_len": semantic_ids_len,
            "bert_feature": bert_feature,
        }

    def get_sample_length(self, idx: int):
        semantic_ids = self.semantic_phoneme[idx][0]
        sec = 1.0 * len(semantic_ids) / self.hz
        return sec

    def collate(self, examples: List[Dict]) -> Dict:
        sample_index: List[int] = []
        phoneme_ids: List[torch.Tensor] = []
        phoneme_ids_lens: List[int] = []
        semantic_ids: List[torch.Tensor] = []
        semantic_ids_lens: List[int] = []
        # return

        for item in examples:
            sample_index.append(item["idx"])
            phoneme_ids.append(np.array(item["phoneme_ids"], dtype=np.int64))  # pyright: ignore
            semantic_ids.append(np.array(item["semantic_ids"], dtype=np.int64))  # pyright: ignore
            phoneme_ids_lens.append(item["phoneme_ids_len"])
            semantic_ids_lens.append(item["semantic_ids_len"])

        # pad 0
        phoneme_ids = batch_sequences(phoneme_ids)  # pyright: ignore
        semantic_ids = batch_sequences(semantic_ids, pad_value=self.PAD)  # pyright: ignore

        # # convert each batch to torch.tensor
        phoneme_ids = torch.tensor(phoneme_ids)  # pyright: ignore
        semantic_ids = torch.tensor(semantic_ids)  # pyright: ignore
        phoneme_ids_lens = torch.tensor(phoneme_ids_lens)  # pyright: ignore
        semantic_ids_lens = torch.tensor(semantic_ids_lens)  # pyright: ignore
        bert_padded = torch.FloatTensor(len(examples), 1024, max(phoneme_ids_lens))
        bert_padded.zero_()

        for idx, item in enumerate(examples):
            bert = item["bert_feature"]
            if bert != None:
                bert_padded[idx, :, : bert.shape[-1]] = bert

        return {
            # List[int]
            "ids": sample_index,
            # torch.Tensor (B, max_phoneme_length)
            "phoneme_ids": phoneme_ids,
            # torch.Tensor (B)
            "phoneme_ids_len": phoneme_ids_lens,
            # torch.Tensor (B, max_semantic_ids_length)
            "semantic_ids": semantic_ids,
            # torch.Tensor (B)
            "semantic_ids_len": semantic_ids_lens,
            # torch.Tensor (B, 1024, max_phoneme_length)
            "bert_feature": bert_padded,
        }


if __name__ == "__main__":
    root_dir = "/data/docker/liujing04/gpt-vits/prepare/dump_mix/"
    dataset = Text2SemanticDataset(
        phoneme_path=root_dir + "phoneme_train.npy",
        semantic_path=root_dir + "semantic_train.tsv",
    )

    batch_size = 12
    dataloader = DataLoader(
        dataset, batch_size=batch_size, collate_fn=dataset.collate, shuffle=False
    )
    for i, batch in enumerate(dataloader):
        if i % 1000 == 0:
            print(i)
