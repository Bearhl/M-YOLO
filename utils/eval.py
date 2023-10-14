import json
import argparse
import numpy as np
from terminaltables import AsciiTable

from utils.config import config, update_config


def iou(pred, gt):  # require pred and gt is numpy
    # pred.shape: (num_box, 2)  gt.shape: (1, 2)
    inter_left = np.maximum(pred[:, 0], gt[:, 0])
    inter_right = np.minimum(pred[:, 1], gt[:, 1])
    inter = np.maximum(0.0, inter_right - inter_left)
    # When the two regions do not overlap, the union calculation does not matter because the intersection is 0
    union_left = np.minimum(pred[:, 0], gt[:, 0])
    union_right = np.maximum(pred[:, 1], gt[:, 1])
    union = np.maximum(0.0, union_right - union_left)

    overlap = 1.0 * inter / union
    # overlap.shape: (nums, )
    return overlap


def rank(pred, gt):
    return pred.index(gt) + 1


def nms(dets, thresh=0.4, top_k=-1):
    """Pure Python NMS baseline."""
    # det.shape: (num_box, 2)
    if len(dets) == 0:
        return []
    order = np.arange(0, len(dets), 1)
    dets = np.array(dets)
    x1 = dets[:, 0]
    x2 = dets[:, 1]
    lengths = x2 - x1
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if len(keep) == top_k:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1)
        ovr = inter / (lengths[i] + lengths[order[1:]] - inter)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return dets[keep]


def eval(segments, data):
    # segments.shape: (all_data, num_box, 2)  data.shape: (all_data, 1, 2)
    assert len(segments) == len(data)

    tious = [float(i) for i in config.TEST.TIOU.split(',')] if isinstance(config.TEST.TIOU, str) \
        else [config.TEST.TIOU]
    recalls = [int(i) for i in config.TEST.RECALL.split(',')] if isinstance(config.TEST.RECALL, str) \
        else [config.TEST.RECALL]

    eval_result = [[[] for _ in recalls] for _ in tious]
    max_recall = max(recalls)
    average_iou = []
    for batch_seg, batch_data in zip(segments, data):
        for seg, dat in zip(batch_seg, batch_data):
            seg = nms(seg, thresh=config.TEST.NMS_THRESH, top_k=max_recall)
            # FIXME we fix the iou function here  --FIXED
            overlap = iou(seg, np.array(dat.unsqueeze(0)))
            average_iou.append(np.mean(overlap))

            for i, t in enumerate(tious):
                for j, r in enumerate(recalls):
                    eval_result[i][j].append((overlap > t)[:r].any())
    eval_result = np.array(eval_result).mean(axis=-1)
    miou = np.mean(average_iou)

    return eval_result, miou


def eval_predictions(segments, data, verbose=True):
    eval_result, miou = eval(segments, data)
    if verbose:
        print(display_results(eval_result, miou, ''))

    return eval_result, miou


def display_results(eval_result, miou, title=None):
    tious = [float(i) for i in config.TEST.TIOU.split(',')] if isinstance(config.TEST.TIOU, str) else [config.TEST.TIOU]
    recalls = [int(i) for i in config.TEST.RECALL.split(',')] if isinstance(config.TEST.RECALL, str) else [
        config.TEST.RECALL]

    display_data = [['Rank@{},mIoU@{}'.format(i, j) for i in recalls for j in tious] + ['mIoU']]
    eval_result = eval_result * 100
    miou = miou * 100
    display_data.append(['{:.02f}'.format(eval_result[j][i]) for i in range(len(recalls)) for j in range(len(tious))]
                        + ['{:.02f}'.format(miou)])
    table = AsciiTable(display_data, title)
    for i in range(len(tious) * len(recalls)):
        table.justify_columns[i] = 'center'
    return table.table


def write2tensorboard(writer, eval_result, miou, iter, split):
    tious = [float(i) for i in config.TEST.TIOU.split(',')] if isinstance(config.TEST.TIOU, str) else [config.TEST.TIOU]
    recalls = [int(i) for i in config.TEST.RECALL.split(',')] if isinstance(config.TEST.RECALL, str) else [
        config.TEST.RECALL]

    eval_result = eval_result * 100
    miou = miou * 100

    for i in range(len(recalls)):
        for j in range(len(tious)):
            writer.add_scalar('{}/ACC/Rank@{}/mIoU@{}'.format(split, recalls[i], tious[j]),
                              eval_result[j][i],
                              iter)

    writer.add_scalar('{}/MIOU'.format(split), miou, iter)


def parse_args():
    parser = argparse.ArgumentParser(description='Train localization network')

    # general
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    args, rest = parser.parse_known_args()

    # update config
    update_config(args.cfg)

    parser.add_argument('--verbose', default=False, action="store_true", help='print progress bar')
    args = parser.parse_args()

    return args


def reset_config(config, args):
    if args.verbose:
        config.VERBOSE = args.verbose
