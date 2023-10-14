import os
import json
import yaml
import torch
import numpy as np

from transformers import CLIPProcessor
from torch.utils.data import Dataset

from utils.loss import build_targets


class collater:
    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, batch):
        """
            collate func for padding data
             *_n represents non-tensor
            :param cfg: The yaml Path
            :param batch: list of the batch data (dict type)
            :return:
            dict:
            {
                vid_n: the collection of vid,
                pad_n: the pad number for every video (Mainly for calculate the actual length of video,
                        which is different from the max_length below)
                frame_n: the number of the padded frames (using compose with pad_n)
                block_n: num of the block
                text.shape: (bs, max_length1)
                video.shape: (bs, max_length2, hidden_dim(e.g. 768))
                label.shape: (bs, 4)
            }
            """
        # merge the dict
        merged_batch = {key: [d.get(key) for d in batch] for key in set().union(*batch)}

        # frame num
        frame_num_list_n = merged_batch["padded_frames"]

        # anchor
        anchor_n = merged_batch["anchor"]

        # block_size
        block_size = merged_batch["block_size"]

        # actual_frame
        actual_frame = merged_batch["actual_frame"]

        # label
        label = torch.tensor(merged_batch["label"])

        # text
        # text.shape: (bs, max_len)

        max_length = max([len(i["input_ids"]) for i in merged_batch["text"]])

        text_list = torch.tensor([i["input_ids"] + [0] * (max_length - len(i["input_ids"]))
                                  for i in merged_batch["text"]])
        text_mask = torch.tensor([i["attention_mask"] + [0] * (max_length - len(i["attention_mask"]))
                                  for i in merged_batch["text"]])

        # video
        # video.shape: (bs, max_len, 768)
        max_length = max(merged_batch["padded_frames"])

        video_list = torch.stack([torch.cat([i, torch.zeros([max_length - i.shape[0], i.shape[1]])], dim=0)
                                  if i.shape[0] < max_length else i[:max_length, :] for i in
                                  merged_batch["video_feature"]], dim=0)

        video_block_length = [i // block_size[0] for i in merged_batch["padded_frames"]]

        max_video_block_length = max(video_block_length)
        video_block_mask = torch.tensor([[1] * i + [0] * (max_video_block_length - i)
                                         if i < max_video_block_length else [1] * max_video_block_length
                                         for i in video_block_length])

        # label
        # label.shape: (bs, 4)
        label_list = torch.tensor(merged_batch["label"])

        index = torch.arange(len(batch)).unsqueeze(1)

        target_info = build_targets(torch.cat([index, label_list], dim=1), video_block_length, self.cfg)

        # TODO transform the window size and mid-point into classification task  -FIXED

        return {
            "frame_n": frame_num_list_n,
            "text": text_list,
            "text_mask": text_mask,
            "video": video_list,
            "label": label,
            "target": target_info,
            "anchor": anchor_n,
            "video_block_mask": video_block_mask,
            "actual_frame": actual_frame
        }


class DatasetBase(Dataset):
    def __init__(self, path, split):
        self.path = path
        self.split = split
        assert split in ["TRAIN", "VAL", "TEST"], "split must choose in ['TRAIN', 'VAL', 'TEST']!!"

        with open(path, "r") as f:
            self.data_dict = yaml.load(f, Loader=yaml.FullLoader)
        self.data_path = self.data_dict["DATASET"][split]
        self.annotation_path = self.data_dict["DATASET"]["ANNOTATION_PATH"]
        self.video_path = self.data_dict["DATASET"]["VIDEO_CACHE"]
        self.anchor_path = self.data_dict["DATASET"]["ANCHOR_PATH"]
        self.processor = CLIPProcessor.from_pretrained("CLIP_model")

        self.item = list()

    def __len__(self):
        return len(self.item)

    def __getitem__(self, index):
        return self.item[index]

    def read_files(self):
        raise NotImplementedError

    def _startup(self):
        raise NotImplementedError


class Charades_dataset(DatasetBase):
    def __init__(self, path, split):
        super(DatasetBase, self).__init__(path, split)
        self._startup()

    def read_files(self):
        with open(self.data_path, "r") as fin:
            lines = fin.readlines()
            video_dict = dict()
            for line in lines:
                info, sent = line.strip().split("##")
                # drop some illegal sentence
                if eval(info.split()[1]) > eval(info.split()[2]):
                    continue
                video_name = info.split()[0]
                if video_name not in video_dict:
                    video_dict[video_name] = [sent]
                else:
                    video_dict[video_name].append(sent)

            anchors = np.array(
                [eval(i) for i in open(self.anchor_path).read().strip().split()])  # shape: (na, )

            for vid in video_dict.keys():
                labels = []
                with open(os.path.join(self.annotation_path, vid + ".txt"), "r") as f:
                    for idx, line in enumerate(f.readlines()):
                        if idx == 0:
                            num_pad, padded_frames = line.strip().split()
                            num_pad, padded_frames = eval(num_pad), eval(padded_frames)
                            continue
                        if line.strip() != '':
                            labels.append([float(i) for i in line.strip().split()])

                    if len(labels) == 0:
                        continue

                    assert len(labels) == len(video_dict[vid])

                video_feature = torch.load(os.path.join(self.video_path, vid + ".pt"))

                for sen, label in zip(video_dict[vid], labels):
                    # text = self.processor(text=sen, return_tensors="pt")
                    text = self.processor(text=sen)

                    self.item.append({
                        "padded_frames": padded_frames,
                        "text": text,
                        "video_feature": video_feature,
                        "label": label,
                        "anchor": anchors,
                        "block_size": self.data_dict["DATASET"]["BLOCK_SIZE"],
                        "actual_frame": padded_frames - num_pad
                    })
            return anchors

    def _startup(self):
        anchor = self.read_files()
        # merge the dict
        length_label = torch.tensor([d.get("label")[0] for d in self.item])

        int_anchor = torch.LongTensor([int(i) for i in anchor])
        length_label = (length_label.unsqueeze(1) - int_anchor.unsqueeze(0)).view(-1)
        self.length_cls_uni = torch.unique(length_label)


class Tacos_dataset(DatasetBase):
    def __init__(self, path, split):
        super(DatasetBase).__init__(path, split)
        self._startup()

    def read_files(self):
        grounding_data = json.load(open(self.data_path, "r"))

        anchors = np.array(
            [eval(i) for i in open(self.anchor_path).read().strip().split()])  # shape: (na, )

        for vid in grounding_data.keys():
            labels = []
            with open(os.path.join(self.annotation_path, vid + ".txt"), "r") as f:
                for idx, line in enumerate(f.readlines()):
                    if idx == 0:
                        num_pad, padded_frames = line.strip().split()
                        num_pad, padded_frames = eval(num_pad), eval(padded_frames)
                        continue
                    if line.strip() != '':
                        labels.append([float(i) for i in line.strip().split()])

                assert len(labels) == len(grounding_data[vid]["sentences"])

            video_feature = torch.load(os.path.join(self.video_path, vid + ".pt"))

            for sen, label in zip(grounding_data[vid]["sentences"], labels):
                # text = self.processor(text=sen, return_tensors="pt")
                text = self.processor(text=sen)
                # TODO maybe some useless and need to to clean up  --FIXED
                self.item.append({
                    "padded_frames": padded_frames,
                    "text": text,
                    "video_feature": video_feature,
                    "label": label,  # absolute label with list [i_length, f_length, i_point, f_point]
                    "anchor": anchors,
                    "block_size": self.data_dict["DATASET"]["BLOCK_SIZE"],
                    "actual_frame": padded_frames - num_pad
                })
        return anchors

    def _startup(self):
        anchor = self.read_files()
        # merge the dict
        length_label = torch.LongTensor([d.get("label")[0] for d in self.item])
        int_anchor = torch.LongTensor([int(i) for i in anchor])
        length_label = (length_label.unsqueeze(1) - int_anchor.unsqueeze(0)).view(-1)
        self.length_cls_uni = torch.unique(length_label)
