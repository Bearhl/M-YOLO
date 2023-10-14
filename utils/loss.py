import math

import yaml
import torch
import numpy as np
import torch.nn as nn


def compute_loss(cfg, con_loss, p, target_info, len_cls_map, len_point_map, mask):
    # p.shape: (bs, na, num_block, 1 + self.len_cls_map + self.point_cls_map + 2)
    device = p.device
    indices, anch, tbox = target_info
    indices, anch, tbox = indices[0], anch[0], tbox[0]

    anch, tbox = torch.tensor(anch).to(device), tbox.to(device)
    len_cls = len(len_cls_map)

    # Define criteria
    BCEcls = nn.BCEWithLogitsLoss().to(device)

    length_ce = nn.CrossEntropyLoss().to(device)
    point_ce = nn.CrossEntropyLoss().to(device)

    length_mse = nn.MSELoss().to(device)
    point_mse = nn.MSELoss().to(device)

    b, a, point_indices = [i.tolist() for i in indices]

    # shape: (bs, na, num_block)
    tcls = torch.zeros_like(p[..., 0], device=device)

    # Objectness classification orient to all prediction
    tcls[b, a, point_indices] = 1
    l_cls = BCEcls(p[..., 0], tcls)

    # other orient to particular prediction
    # ps.shape: (??, 1 + self.len_cls_map + self.point_cls_map + 2)
    ps = p[b, a, point_indices]
    # TODO take the anchor into consideration  --FIXED
    # anch is the absolute value, so it's necessary to split into integer and float -> build the actual boundary label
    anch = torch.tensor([math.modf(i) for i in anch.tolist()]).to(device)
    # anchor + pred = label  -> pred = label - anchor
    length_ce_label = torch.LongTensor([len_cls_map[i] for i in (tbox[:, 0] - anch[:, 1]).tolist()]).to(device)
    l_len1 = length_ce(ps[..., 1: 1 + len_cls], length_ce_label)
    l_len2 = length_mse(torch.sigmoid(ps[..., -2]), tbox[:, 1] - anch[:, 0])

    point_ce_label = torch.LongTensor([len_point_map[i] for i in tbox[:, 2].tolist()]).to(device)
    l_point1 = point_ce(ps[..., 1 + len_cls: -2], point_ce_label)
    l_point2 = point_mse(torch.sigmoid(ps[..., -1]), tbox[:, 3])

    total_loss = cfg["LOSS"]["ALPHA"] * l_cls + cfg["LOSS"]["BETA"] * (l_len1 + l_point1) + \
                 cfg["LOSS"]["SCALE"] * cfg["LOSS"]["BETA"] * (l_len2 + l_point2) + cfg["LOSS"]["GAMMA"] * con_loss

    joint_prob = p * mask.unsqueeze(1).unsqueeze(-1)

    return total_loss, joint_prob


def build_targets(targets, block_length, data_dict):
    # targets shape: (nt, 5)

    # targets (bs, f_length, c_length, f_mid, c_mid) # bs represents the idx of the label
    # TODO dynamically calculate by data_dict["pos_windows"]
    off = torch.tensor([[0, 0], [-2, 0], [-1, 0], [1, 0], [2, 0]])

    with open(data_dict["DATASET"]["ANCHOR_PATH"], "r") as f:
        anchors = np.array([eval(i) for i in f.read().strip().split()])  # shape: (na, )

    indices, tbox, anch = [], [], []
    na, nt = anchors.shape[0], targets.shape[0]  # number of anchors, targets
    ai = torch.arange(na).float().view(na, 1).repeat(1, nt)  # ai.shape: (na, nt)
    targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # target.shape: (na, nt, 6)

    if nt:
        target_windows_size = targets[..., 1:3].sum(2, keepdims=True)  # shape: (na, nt, 1)
        # calculate the radio and filter the anchors
        r = target_windows_size / anchors[:, None, None]  # shape: (na, nt, 1)
        # FIXME: Check whether correct  --FIXED
        j = torch.max(r, 1. / r).max(-1)[0] < data_dict["LOSS"]["ANCHOR_T"]  # j.shape: (na, nt)
        if not torch.all(j):
            j = torch.ones(j.shape).bool().to(j.device)

        # filter the unmatch ratio between targets and anchor that > 4.
        t = targets[j]  # t.shape: (?, 6) -> ? represents the filtered number of the target

        # overlaps
        # [10, 0.02]
        target_point = t[:, 3:5]  # box.shape: (?, 2) -> ? represents the filtered number of the target
        # the boxes near the target also regarded as positive sample
        # pos_windows_size = data_dict["pos_windows"]
        j = torch.ones((5, t.shape[0]), dtype=torch.long).bool()  # j.shape: (5, ?) as we do not filter here

        t = t.repeat((5, 1, 1))[j]  # t.shape: (5, ?, 6) -> (??, 6) -> ?? represents 5 * ?
        # off.shape: (5, 2)  offset.shape: (??, 2)

        offsets = torch.cat([off[:, None] for _ in range(target_point.shape[0])], dim=1)[j]

        boundary = torch.tensor(block_length)[t[:, 0].long()]
        selected_point = torch.sum(t[:, 3: 5], dim=-1) - offsets[:, 0] >= 0
        t = t[selected_point]
        boundary = boundary[selected_point]
        offsets = offsets[selected_point]

        selected_point2 = torch.sum(t[:, 3: 5], dim=-1) - offsets[:, 0] < boundary
        t = t[selected_point2]
        offsets = offsets[selected_point2]

    else:
        t = targets[0]
        offsets = 0

    # Define
    # t.shape: (??, 6) (bs, f_length, c_length, f_mid, c_mid, anchor)
    b = t[:, 0].long()
    b_point = t[:, 3:5]
    b_len = t[:, 1:3]

    point_indices = (b_point - offsets)[:, 0].long()  # shape: (??, )

    a = t[:, 5].long()

    # label_ids, anchor, box indices
    indices.append((b, a, point_indices))
    # anch.shape: (??, 1) save every positive anchor
    anch.append(anchors[a])
    # tbox.shape: (??, 4) -> f_length, c_length, f_mid, c_mid
    # FIXME: Check whether correct  --FIXED
    tbox.append(torch.cat((b_len, torch.cat([offsets[:, 0].unsqueeze(1), t[:, 4].unsqueeze(1)], dim=1)), dim=1))

    return (indices, anch, tbox)
