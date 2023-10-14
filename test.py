import os

import pickle as pkl

from utils.config import config

def save_score(scores, data, dataset_name, split):
    results = {}
    for i, d in enumerate(data):
        results[d["video"]] = scores[i]
    pkl.dump(results, open(os.path.join(config.RESULT_DIR, dataset_name,
                            "{}_{}_{}.pkl".format(config.MODEL.NAME, config.DATASET.VIS_INPUT_TYPE, split)), "wb"))

def save_to_txt(scores, data, dataset_name, split):
    txt_path = os.path.join(config.RESULT_DIR, dataset_name,
                            "{}_{}_{}.txt".format(config.MODEL.NAME, config.DATASET.VIS_INPUT_TYPE, split))
    rootfolder1 = os.path.dirname(txt_path)
    rootfolder2 = os.path.dirname(rootfolder1)
    if not os.path.exists(rootfolder2):
        print("Make directory %s ..." % rootfolder2)
        os.mkdir(rootfolder2)
    if not os.path.exists(rootfolder1):
        print("Make directory %s ..." % rootfolder1)
        os.mkdir(rootfolder1)
    with open(txt_path, "w") as fb:
        for i, d in enumerate(data):
            fb.write("{} {} == {} {} = {} {} {}\n".format(d["video"], d["description"], d["times"][0], d["times"][1],
                                                          scores[i][0], scores[i][1], scores[i][2]))
            