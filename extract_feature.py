import os
import cv2
import math
import json
import yaml
import torch

import pandas as pd

from tqdm import tqdm
from PIL import Image
from scipy.cluster.vq import kmeans
from transformers import CLIPProcessor, CLIPModel

import numpy as np

dataset_path = {
    "Tacos": {
        "text": "/nfsc/aies_core_id10061778_vol1001_dev/backup/xionghaoliang660/datasets/Moment_Retrieval/text_data/TACoS/",
        "videos": "/nfsc/aies_core_id10061778_vol1001_dev/backup/xionghaoliang660/datasets/Moment_Retrieval/TaCoS/videos"},
    "Charades-STA": {
        "text": "/nfsc/aies_core_id10061778_vol1001_dev/backup/xionghaoliang660/datasets/Moment_Retrieval/text_data/Charades-STA",
        "videos": "/nfsc/aies_core_id10061778_vol1001_dev/backup/xionghaoliang660/datasets/Moment_Retrieval/Charades_v1"},
    "ActivityNet": {"text": "",
                    "videos": ""}
}

# The output dir for all data
output_path = {
    "Tacos": "/nfsc/aies_core_id10061778_vol1001_dev/backup/xionghaoliang660/datasets/Moment_Retrieval/TACoS_cache/",
    "Charades-STA": "/nfsc/aies_core_id10061778_vol1001_dev/backup/xionghaoliang660/datasets/Moment_Retrieval/Charades_v1_cache/",
    "ActivityNet": ""
}


class Annotations_builder:
    def __init__(self, path, rebuild_anchor=False):
        self.path = path
        self.rebuild_anchor = rebuild_anchor

        with open(path) as f:
            self.data_dict = yaml.load(f, Loader=yaml.FullLoader)
        self.anchor_length = list()

    def get_tacos_annotations(self):
        if not os.path.exists(self.data_dict["DATASET"]["ANNOTATION_PATH"]):
            os.mkdir(self.data_dict["DATASET"]["ANNOTATION_PATH"])

        for split in ["TRAIN", "VAL", "TEST"]:
            with open(self.data_dict["DATASET"][split], "r") as f:
                annotations = json.load(f)

            # since the feature length may not equal to the frame num in num_frames
            # we will take the first num_frames as the feature
            for vid, video_anno in annotations.items():
                num_pad = video_anno["num_frames"] % self.data_dict["DATASET"]["BLOCK_SIZE"]
                padded_frame = (video_anno["num_frames"] + (self.data_dict["DATASET"]["BLOCK_SIZE"] - num_pad)) \
                    if num_pad > 0 else video_anno["num_frames"]
                with open(os.path.join(self.data_dict["DATASET"]["ANNOTATION_PATH"], vid + '.txt'), "w") as f:
                    f.write(' '.join([str(self.data_dict["DATASET"]["BLOCK_SIZE"] - num_pad), str(padded_frame), "\n"]))
                    for timestamp in video_anno["timestamps"]:
                        if timestamp[0] < timestamp[1]:
                            length = (timestamp[1] - timestamp[0]) / self.data_dict["DATASET"]["BLOCK_SIZE"] / 2
                            if split == "TRAIN":
                                self.anchor_length.append(length)
                            (f_length, i_length) = math.modf(length)
                            f_length, i_length = float(f_length), int(i_length)
                            (f_mid, i_mid) = math.modf((timestamp[0] + (timestamp[1] - timestamp[0]) / 2) /
                                                       self.data_dict["DATASET"]["BLOCK_SIZE"])
                            f_mid, i_mid = float(f_mid), int(i_mid)
                            f.write(' '.join([str(i_length), str(f_length), str(i_mid), str(f_mid), "\n"]))

    def get_charades_annotations(self):
        # TODO add the num_block and pad_block based on the .csv file  -Fixed
        if not os.path.exists(self.data_dict["DATASET"]["ANNOTATION_PATH"]):
            os.mkdir(self.data_dict["DATASET"]["ANNOTATION_PATH"])

        data_dict = {}
        csv_dict = {}
        fps = 16.
        for split in ["TRAIN", "TEST"]:

            csv_file = self.data_dict["DATASET"][split].replace("charades_sta_{}.txt".format(split.lower()),
                                                                "Charades_v1_{}.csv".format(split.lower()))
            csv_data = pd.read_csv(csv_file)
            for i in range(len(csv_data)):
                csv_dict[csv_data["id"][i]] = int(csv_data["length"][i] * 16.)

            with open(self.data_dict["DATASET"][split], "r") as fin:
                lines = fin.readlines()
                video_set = set()
                for line in lines:
                    video_set.add(line.strip().split("##")[0].split()[0])

                for i in list(video_set):
                    data_dict[i] = list()

                for line in lines:
                    file, timestamp_s, timestamp_e = line.strip().split("##")[0].split()
                    timestamp_s = eval(timestamp_s)
                    timestamp_e = eval(timestamp_e)
                    timestamp_s *= fps
                    timestamp_e *= fps
                    if timestamp_s < timestamp_e:
                        length = (timestamp_e - timestamp_s) / self.data_dict["DATASET"]["BLOCK_SIZE"] / 2
                        if split == "TRAIN":
                            self.anchor_length.append(length)
                        (f_length, i_length) = math.modf(length)
                        f_length, i_length = float(f_length), int(i_length)
                        (f_mid, i_mid) = math.modf((timestamp_s + (timestamp_e - timestamp_s) / 2)
                                                   / self.data_dict["DATASET"]["BLOCK_SIZE"])
                        f_mid, i_mid = float(f_mid), int(i_mid)
                        data_dict[file].append([str(i_length), str(f_length), str(i_mid), str(f_mid), "\n"])

        for k, v in data_dict.items():
            with open(os.path.join(self.data_dict["DATASET"]["ANNOTATION_PATH"], k + '.txt'), "w") as f:
                num_pad = csv_dict[k] % self.data_dict["DATASET"]["BLOCK_SIZE"]
                padded_frame = (csv_dict[k] + (self.data_dict["DATASET"]["BLOCK_SIZE"] - num_pad)) if num_pad > 0 else \
                    csv_dict[k]
                f.write(' '.join([str(self.data_dict["DATASET"]["BLOCK_SIZE"] - num_pad), str(padded_frame), "\n"]))
                for i in v:
                    f.write(' '.join(i))

    def build_anchor(self):
        num_anchor = self.data_dict["LOSS"]["NA"]
        anchor_file = self.data_dict["DATASET"]["ANCHOR_PATH"]
        k, dist = kmeans(np.array(self.anchor_length), num_anchor, iter=500)
        np.savetxt(anchor_file, k, delimiter=" ")

    def run(self):
        if "tacos" in self.path.lower():
            self.get_tacos_annotations()
            if self.rebuild_anchor:
                self.build_anchor()
        elif "charades" in self.path.lower():  # Charades
            self.get_charades_annotations()
            if self.rebuild_anchor:
                self.build_anchor()


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, frame_list):
        self.frame_list = frame_list

    def __len__(self):
        return len(self.frame_list)

    def __getitem__(self, index):
        return self.frame_list[index]


def extract_feature(dataset_name):
    assert dataset_name in ["Tacos", "Charades-STA", "ActivityNet"], "The dataset must 'Tacos', " \
                                                                     "'Charades-STA' or 'ActivityNet' !!"

    def build_clip_model(vit_name="openai/clip-vit-base-patch32"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        clip_model = CLIPModel.from_pretrained(vit_name).vision_model
        clip_processor = CLIPProcessor.from_pretrained(vit_name)
        return clip_model.to(device), clip_processor, device

    text_path = dataset_path[dataset_name]["text"]
    videos_path = dataset_path[dataset_name]["videos"]
    if not os.path.exists(output_path[dataset_name]):
        os.mkdir(output_path[dataset_name])

    model, processor, device = build_clip_model()
    if dataset_name == "Tacos":
        for split in ["train", "val", "test"]:
            print("processing the {} data...".format(split))
            grounding_data = json.load(open(os.path.join(text_path, "{}.json".format(split))))
            videos_ids = list(grounding_data.keys())
            for v in videos_ids:
                output_list = []
                frame_list = []
                video_path = os.path.join(videos_path, v)
                frame_num = grounding_data[v]["num_frames"]
                capture = cv2.VideoCapture(video_path)

                if capture.isOpened():
                    with tqdm(total=frame_num, leave=False) as pbar:
                        while True:
                            ret, img = capture.read()
                            if not ret:
                                # assert len(output_list) == frame_num, "the number of frame of {} is not match!!".format(v)
                                dt = SimpleDataset(frame_list)
                                dataloader = torch.utils.data.DataLoader(dt, batch_size=1024, shuffle=False)
                                with torch.no_grad():
                                    for i in dataloader:
                                        img_output = model(i.to(device))
                                        output_list.append(img_output["pooler_output"].cpu())
                                        torch.cuda.empty_cache()
                                tensor_v = torch.cat(output_list, dim=0)

                                torch.save(tensor_v, os.path.join(output_path[dataset_name], "{}.pt".format(v)))
                                del output_list
                                break

                            image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            image = Image.fromarray(image)
                            img_output = processor(images=image, return_tensors="pt")["pixel_values"].squeeze()
                            frame_list.append(img_output)
                            pbar.update(1)
                else:
                    print("打开视频{}失败!!".format(v))

    elif dataset_name == "Charades-STA":
        for split in ["train", "test"]:
            print("processing the {} data...".format(split))
            with open(os.path.join(text_path, "charades_sta_{}.txt".format(split)), 'r') as f:
                lines = f.readlines()
                video_set = set()
                for line in lines:
                    video_set.add(line.strip().split("##")[0].split()[0])
                for v in tqdm(list(video_set), desc="Processing"):
                    output_list = []
                    frame_list = []
                    video_path = os.path.join(videos_path, "{}.mp4".format(v))
                    capture = cv2.VideoCapture(video_path)

                    if capture.isOpened():
                        while True:
                            ret, img = capture.read()
                            if not ret:
                                dt = SimpleDataset(frame_list)
                                dataloader = torch.utils.data.DataLoader(dt, batch_size=1024, shuffle=False)
                                with torch.no_grad():
                                    for i in dataloader:
                                        img_output = model(i.to(device))
                                        output_list.append(img_output["pooler_output"].cpu())
                                        torch.cuda.empty_cache()
                                tensor_v = torch.cat(output_list, dim=0)

                                torch.save(tensor_v, os.path.join(output_path[dataset_name], "{}.pt".format(v)))
                                del output_list
                                break

                            image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            image = Image.fromarray(image)
                            img_output = processor(images=image, return_tensors="pt")["pixel_values"]
                            frame_list.append(img_output)
                    else:
                        print("打开视频{}失败!!".format(v))


if __name__ == '__main__':
    # dataset = 'Tacos'
    # extract_feature(dataset)
    # dataset = 'Charades-STA'
    # extract_feature(dataset)
    Anno_build = Annotations_builder("./config/Tacos.yaml", rebuild_anchor=True)
    Anno_build.run()

    Anno_build = Annotations_builder("./cofig/Charades-STA.yaml", rebuild_anchor=True)
    Anno_build.run()

