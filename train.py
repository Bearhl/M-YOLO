import os
import sys
import math
import torch
import pprint
import argparse

import numpy as np
import torch.optim as optim
import torch.distributed as dist
import torch.backends.cudnn as cudnn

from tqdm import tqdm
from torch.utils.data import DataLoader

from utils.model import M_YOLO
from utils.engine import Engine
from utils.loss import compute_loss
from utils.eval import eval_predictions, display_results, write2tensorboard
from utils.utils import create_logger, AverageMeter
from utils.config import config, update_config
from utils.dataset import Charades_dataset, Tacos_dataset, collater

dataset_map = {
    "Tacos": Tacos_dataset,
    "Charades": Charades_dataset
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train Localization network")

    # general
    # For Debug
    parser.add_argument("--cfg", help="experiment configure file name", default="config/Tacos.yaml")
    # parser.add_argument("--cfg", help="experiment configure file name", required=True, type=str)

    args, rest = parser.parse_known_args()

    # update config
    # config.py parameters must include more than .yaml parameters, otherwise an error will be reported
    update_config(args.cfg)

    # training
    parser.add_argument("--gpus", help="gpus", default="0, 1, 2, 3", type=str)
    parser.add_argument("--workers", help="num of dataloader workers", type=int)
    parser.add_argument("--modelDir", help="model path", type=str)
    parser.add_argument("--logDir", help="log path", type=str)
    parser.add_argument("--verbose", default=False, action="store_true", help="print progress bar")
    parser.add_argument("--tag", help="tags shown in log", type=str)
    parser.add_argument("--local_rank", help="local rank", type=int, default=0)
    parser.add_argument("--tensorboardDir", help="tensorboard path", type=str)
    args = parser.parse_args()

    return args


def reset_config(config, args):
    if args.gpus:
        config.GPUS = args.gpus
    if args.workers:
        config.WORKERS = args.workers
    if args.modelDir:
        config.MODEL_DIR = args.modelDir
    if args.logDir:
        config.LOG_DIR = args.logDir
    if args.verbose:
        config.VERBOSE = args.verbose
    if args.tag:
        config.TAG = args.tag
    if args.tensorboardDir:
        config.TENSORBOARD_DIR = args.tensorboardDir


def synchronize(verbose=False):
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    if verbose:
        print("waiting={}".format(dist.get_rank()))
        dist.barrier()
    if verbose:
        print("waiting finished={}".format(dist.get_rank()))


def gather_tensor(data, dim=0, dst=None):
    N = dist.get_world_size()
    if N == 1:
        return data
    is_dst = dst is None or dst == dist.get_rank()
    # get tensor size
    size = torch.tensor(data.shape[dim], device=data.device)
    size_list = [size.clone() for _ in range(N)] if is_dst else None
    if dst is None:
        torch.distributed.all_gather(size_list, size)
    else:
        torch.distributed.gather(tensor=size, gather_list=size_list, dst=dst)
    max_size = max(size.item() for size in size_list)
    shape = list(data.shape)
    shape[dim] = max_size
    tensor_list = [data.new_empty(shape) for _ in range(N)] if is_dst else None
    # pad to same shape
    if data.shape[dim] != max_size:
        shape[dim] = max_size - data.shape[dim]
        tensor = torch.cat([data, data.new_zeros(shape)], dim=dim)
    else:
        tensor = data
    if dst is None:
        torch.distributed.all_gather(tensor_list, tensor)
    else:
        torch.distributed.gather(tensor, tensor_list, dst)
    if is_dst:
        return torch.cat([x.narrow(dim, 0, n.item()) for n, x in zip(size_list, tensor_list)], dim=dim)
    else:
        return None


def main():
    args = parse_args()
    reset_config(config, args)

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED
    torch.multiprocessing.set_sharing_strategy("file_system")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    num_gpus = torch.cuda.device_count()

    if num_gpus > 1:
        distribute = True
    else:
        distribute = False

    if args.local_rank == 0:
        main_work = True
    else:
        main_work = False

    dataset_name = config.DATASET.NAME
    model_name = config.MODEL.NAME

    torch.cuda.set_device(args.local_rank)

    if main_work:
        logger, final_output_dir, time_str, tb_writer = create_logger(config, args.cfg, config.TAG)
        logger.info("\n" + pprint.pformat(args))
        logger.info("\n" + pprint.pformat(config))

    if dataset_name == "Tacos":
        train_dataset = Tacos_dataset(args.cfg, split="TRAIN")
        val_dataset = Tacos_dataset(args.cfg, split="VAL")
        test_dataset = Tacos_dataset(args.cfg, split="TEST")

    else:
        # Charades
        train_dataset = Charades_dataset(args.cfg, split="TRAIN")
        val_dataset = Charades_dataset(args.cfg, split="VAL")
        test_dataset = Charades_dataset(args.cfg, split="TEST")

    length_cls_uni = torch.unique(torch.cat([train_dataset.length_cls_uni, val_dataset.length_cls_uni,
                                             test_dataset.length_cls_uni])).tolist()
    map_length_cls = {i: idx for idx, i in enumerate(length_cls_uni)}
    rev_length_cls = {idx: i for idx, i in enumerate(length_cls_uni)}

    map_point_cls = {
        0: 0,
        -2: 1,
        -1: 2,
        1: 3,
        2: 4
    }

    rev_point_cls = {v: k for k, v in map_point_cls.items()}

    model = M_YOLO(args.cfg, map_length_cls, map_point_cls)

    if distribute:
        torch.distributed.init_process_group(backend="nccl")
        synchronize()

    model = model.cuda(args.local_rank)
    if distribute:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)

    params = model.parameters()

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, params), lr=config.TRAIN.LR, betas=(0.9, 0.999), eps=1e-8,
                           weight_decay=config.TRAIN.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=config.TRAIN.FACTOR,
                                                     patience=config.TRAIN.PATIENCE, verbose=config.VERBOSE)

    def iterator(split):
        def get_sampler(i_dataset, shuffle=True, test=False):
            if test:
                return torch.utils.data.SequentialSampler(i_dataset)
            elif distribute:
                return torch.utils.data.distributed.DistributedSampler(i_dataset, shuffle=shuffle)
            elif shuffle:
                return torch.utils.data.RandomSampler(i_dataset)
            else:
                return torch.utils.data.SequentialSampler(i_dataset)

        collate_fn = collater(config)
        if split == "train":
            sampler = get_sampler(train_dataset)
            dataloader = DataLoader(train_dataset,
                                    batch_size=config.TRAIN.BATCH_SIZE // num_gpus,
                                    pin_memory=True,
                                    sampler=sampler,
                                    collate_fn=collate_fn)
        elif split == "val":
            sampler = get_sampler(val_dataset, shuffle=False, test=True)
            dataloader = DataLoader(val_dataset,
                                    # batch_size=config.TEST.BATCH_SIZE // num_gpus,
                                    batch_size=config.TEST.BATCH_SIZE,
                                    pin_memory=False,
                                    sampler=sampler,
                                    collate_fn=collate_fn)
        elif split == "test":
            sampler = get_sampler(test_dataset, shuffle=False, test=True)
            dataloader = DataLoader(test_dataset,
                                    # batch_size=config.TEST.BATCH_SIZE // num_gpus,
                                    batch_size=config.TEST.BATCH_SIZE,
                                    pin_memory=True,
                                    sampler=sampler,
                                    collate_fn=collate_fn)
        else:
            raise NotImplementedError

        return dataloader

    def network(sample, epoch=0):
        text = sample["text"].cuda(non_blocking=True)
        text_mask = sample["text_mask"].cuda(non_blocking=True)
        video = sample["video"].cuda(non_blocking=True)
        video_block_mask = sample["video_block_mask"].cuda(non_blocking=True)
        target = sample["target"]
        anchor = sample["anchor"]
        actual_frame = sample["actual_frame"]

        prediction, con_loss = model(text, text_mask, video, video_block_mask)
        # TODO calculate the loss --FIXED
        loss_value, joint_prob = compute_loss(config, con_loss, prediction, target,
                                              map_length_cls, map_point_cls, video_block_mask)

        if model.training:
            return loss_value, None
        else:
            # TODO fill the NMS function --FIXED
            sorted_times = get_proposal_result(joint_prob, anchor, actual_frame)
            return loss_value, sorted_times

    def get_proposal_result(joint_prob, anchor, actual_frame):
        # join_prob.shape: (bs, na, n_block, 1 + map_length_cls + map_point_cls + 2)
        out_sorted_times = []
        # anchor.shape: (na, 2)
        anchor = torch.tensor([math.modf(i) for i in anchor[0]]).to(joint_prob.device)

        for idx, i in enumerate(joint_prob):
            # i.shape: (na, n_block, hidden)
            # corresponding to the loss, change together
            score = torch.sigmoid(i[..., 0]).cpu().detach().numpy()
            length_tensor = torch.argmax(i[:, :, 1: 1 + len(map_length_cls)], dim=-1).long()
            flatten_length = length_tensor.flatten()
            index_length = torch.tensor([rev_length_cls[item.item()] for item in flatten_length])
            length_cls = index_length.view(length_tensor.size()).to(joint_prob.device)

            point_tensor = torch.argmax(i[:, :, 1 + len(map_length_cls): -2], dim=-1).long()
            flatten_point = point_tensor.flatten()
            index_point = torch.tensor([rev_point_cls[item.item()] for item in flatten_point])
            point_cls = index_point.view(point_tensor.size()).to(joint_prob.device)

            # shape: (na, n_block)
            # TODO take the anchor into consideration  --FIXED
            actual_len = (length_cls + anchor[:, 1].unsqueeze(1)) + (
                    torch.sigmoid(i[..., -2]) + anchor[:, 0].unsqueeze(1))
            actual_point = point_cls + torch.sigmoid(i[..., -1])
            # shape: (1, na, n_block)
            # TODO filter the lower score previous
            sorted_index = np.dstack(np.unravel_index(np.argsort(score.ravel())[::-1], (i.shape[0], i.shape[1])))
            sorted_index = torch.tensor(sorted_index[0]).to(joint_prob.device)
            flat_index = sorted_index[:, 0] * actual_point.shape[1] + sorted_index[:, 1]

            sorted_point = torch.index_select(actual_point.view(-1), 0, flat_index)
            sorted_length = torch.index_select(actual_len.view(-1), 0, flat_index)

            # TODO filter the score which out of the bound of the video (based on the frame_num and 0) --FIXED

            # FIXME Since we will perform batch dimension concatenation later, there is no way to remove some
            #  vectors for filtering, so we use clamp function for interval limiting --FIXED
            start_frame = torch.clamp(sorted_point - sorted_length, min=0)
            end_frame = torch.clamp(sorted_point + sorted_length, max=actual_frame[idx])

            # (na, block, 2)
            out_sorted_times.append(torch.stack((start_frame, end_frame), dim=1))
        # (bs, na, block, 2)
        return torch.stack(out_sorted_times, dim=0)

    def on_start(state):
        state["test_interval"] = math.ceil(len(train_dataset) / config.TRAIN.BATCH_SIZE * config.TEST.INTERVAL)

        if config.TRAIN.FP16:
            state["scaler"] = torch.cuda.amp.GradScaler()

        if config.TRAIN.FINE_TUNE and not config.TRAIN.CONTINUE:
            loc = "cuda:{}".format(args.local_rank)
            checkpoint = torch.load(config.MODEL.CHECKPOINT, map_location=loc)
            # FIXME Maybe useless ?
            # model.module.load_object_params(checkpoint["model"])
            model.module.load_state_dict(checkpoint["model"])

        if config.MODEL.CHECKPOINT and config.TRAIN.CONTINUE:
            loc = "cuda:{}".format(args.local_rank)
            checkpoint = torch.load(config.MODEL.CHECKPOINT, map_location=loc)
            if "optimizer" in checkpoint:
                model.module.load_state_dict(checkpoint["model"])
                state["optimizer"].load_state_dict(checkpoint["optimizer"])
                state["scheduler"].load_state_dict(checkpoint["scheduler"])
                state["t"] = checkpoint["step"] + 1
                if ("scaler" in checkpoint) and (state["scaler"] is not None) and (checkpoint["scaler"] is not None):
                    state["scaler"].load_state_dict(checkpoint["scaler"])
                state["epoch"] = state["t"] // state["test_interval"]
            else:
                if distribute:
                    model.module.load_state_dict(checkpoint)
                else:
                    model.load_state_dict(checkpoint)
        state["loss_meter"] = AverageMeter()

        tious = [float(i) for i in config.TEST.TIOU.split(",")] if isinstance(config.TEST.TIOU, str) else [
            config.TEST.TIOU]
        recalls = [int(i) for i in config.TEST.RECALL.split(",")] if isinstance(config.TEST.RECALL, str) else [
            config.TEST.RECALL]
        state["best"] = [[0 for _ in recalls] for _ in tious]
        state["best_miou"] = 0

        model.train()
        if config.VERBOSE and main_work:
            state["progress_bar"] = tqdm(total=state["test_interval"])

    def on_start_epoch(state):
        if distribute:
            state["iterator"].sampler.set_epoch(state["epoch"])

    def on_forward(state):
        if state["t"] % state["step_accumulate"] == 0 or state["t"] % state["test_interval"] == 0:
            if state["scaler"] is not None:
                state["scaler"].unscale_(state["optimizer"])
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)

        if distribute:
            dist.all_reduce(state["loss"], op=dist.ReduceOp.SUM)

        state["loss_meter"].update(state["loss"].item() / num_gpus, 1)
        # update the lr of transformer
        if hasattr(model, "adjust_lr"):
            model.adjust_lr(state["optimizer"], state["t"])

    def on_update(state):
        if config.VERBOSE and main_work:
            state["progress_bar"].update(1)

        if state["t"] % state["test_interval"] == 0:
            model.eval()
            if distribute:
                synchronize()

            if config.VERBOSE and main_work:
                state["progress_bar"].close()

                loss_message = "\nepoch: {} iter: {} train loss {:.4f}".format(state["epoch"], state["t"],
                                                                               state["loss_meter"].avg)
                tb_writer.add_scalars("LOSS", {"train": state["loss_meter"].avg}, state["t"])

                table_message = ""

            if not config.DATASET.NO_VAL:

                if main_work:
                    val_state = engine.test(network, iterator("val"), "val")
                    state["scheduler"].step(-val_state["loss_meter"].avg)

                    loss_message += " val loss {:.4f}".format(val_state["loss_meter"].avg)
                    tb_writer.add_scalars("LOSS", {"val": val_state["loss_meter"].avg}, state["t"])
                    val_state["loss_meter"].reset()
                    val_table = display_results(val_state["Rank@N,mIoU@M"], val_state["miou"],
                                                "performance on validation set")
                    write2tensorboard(tb_writer, val_state["Rank@N,mIoU@M"], val_state["miou"], state["t"], "val")
                    table_message += "\n" + val_table

            if main_work:
                test_state = engine.test(network, iterator("test"), "test")
                state["scheduler"].step(-test_state["loss_meter"].avg)

                loss_message += " test loss {:.4f}".format(test_state["loss_meter"].avg)
                tb_writer.add_scalars("LOSS", {"test": test_state["loss_meter"].avg}, state["t"])
                test_state["loss_meter"].reset()
                test_table = display_results(test_state["Rank@N,mIoU@M"], test_state["miou"],
                                             "performance on testing set")
                write2tensorboard(tb_writer, test_state["Rank@N,mIoU@M"], test_state["miou"], state["t"], "test")
                table_message += "\n" + test_table

                message = loss_message + table_message + "\n"
                logger.info(message)
                tb_writer.flush()

                # assert if better result
                # TODO: Check if necessary --FIXED
                save_checkpoint = False
                if test_state["miou"] > state["best_miou"]:
                    state["best_miou"] = test_state["miou"]
                    save_checkpoint = True
                for i_tiou in range(len(state["best"])):
                    for i_recall in range(len(state["best"][i_tiou])):
                        if state["best"][i_tiou][i_recall] < test_state["Rank@N,mIoU@M"][i_tiou][i_recall]:
                            state["best"][i_tiou][i_recall] = test_state["Rank@N,mIoU@M"][i_tiou][i_recall]
                            save_checkpoint = True

                if save_checkpoint:
                    saved_model_filename = os.path.join(config.MODEL_DIR,
                                                        "{}/{}/{}/iter{:06d}-{:.4f}-{:.4f}.pkl".format(
                                                            dataset_name,
                                                            model_name, time_str,
                                                            state["t"], test_state["Rank@N,mIoU@M"][0, 0],
                                                            test_state["Rank@N,mIoU@M"][0, 1]
                                                        ))

                    rootfolder = os.path.dirname(saved_model_filename)
                    if not os.path.exists(rootfolder):
                        print('Make directory %s ...' % rootfolder)
                        os.makedirs(rootfolder, exist_ok=True)

                    save_state = {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        # "scalar": state["scaler"].state_dict() if state["scalar"] is not None else None,
                        "step": state["t"]
                    }
                    torch.save(save_state, saved_model_filename)

            if config.VERBOSE and main_work:
                state["progress_bar"] = tqdm(total=state["test_interval"])
                state["loss_meter"].reset()

            # if distribute:
            #     synchronize()
            model.train()

    def on_end(state):
        if config.VERBOSE and main_work:
            state["progress_bar"].close()
            tb_writer.close()
        if distribute:
            synchronize()

    def on_test_start(state):
        state['loss_meter'] = AverageMeter()
        if main_work:
            state['label_list'] = []
            state['sorted_segments_list'] = []
        if config.VERBOSE and main_work:
            if state['split'] == 'train':
                # state['progress_bar'] = tqdm(total=math.ceil(len(train_dataset) / config.TRAIN.BATCH_SIZE))
                state["progress_bar"] = tqdm(total=len(state["iterator"]))
            elif state['split'] == 'val':
                # state['progress_bar'] = tqdm(total=math.ceil(len(val_dataset) / config.TEST.BATCH_SIZE))
                state['progress_bar'] = tqdm(total=len(state["iterator"]))
            elif state['split'] == 'test':
                # state['progress_bar'] = tqdm(total=math.ceil(len(test_dataset) / config.TEST.BATCH_SIZE))
                state['progress_bar'] = tqdm(total=len(state["iterator"]))
            else:
                raise NotImplementedError

    def on_test_forward(state):
        if config.VERBOSE and main_work:
            state["progress_bar"].update(1)
        # if distribute:
        #     dist.all_reduce(state["loss"], op=dist.ReduceOp.SUM)
        state["loss_meter"].update(state["loss"].item() / num_gpus, 1)

        if distribute:
            sorted_segments = state["output"]  # batch_size * na * n_block * 2
            # TODO Flatten the vector ? --FIXED
            # sorted_segments = torch.flatten(sorted_segments, start_dim=1, end_dim=2)  # batch_size * num_box * 2

            # rebuild label
            label = state["sample"]["label"]  # batch_size, 4 torch.tensor
            length = label[:, 0] + label[:, 1]
            point = label[:, 2] + label[:, 3]
            start = point - length
            end = point + length
            label = torch.stack((start, end), dim=1)  # shape: (bs, 2)
            sorted_segments = torch.cat((label[:, None, :].to(sorted_segments.device), sorted_segments), dim=1)

            # TODO Debug to check whether correct
            # gather_segment = gather_tensor(sorted_segments)  # batch_size * (1 + num_box) * 2
            if main_work:
                state["label_list"].append(sorted_segments[:, 0, :].cpu())
                state["sorted_segments_list"].append(sorted_segments[:, 1:, :].cpu())

        else:
            # FIXME maybe useless ?
            min_idx = min(state["sample"]["index"])
            batch_indexs = [idx - min_idx for idx in state["state"]["index"]]
            sorted_segments = [state["output"][i] for i in batch_indexs]
            state["sorted_segments_list"].extend(sorted_segments)

    def on_test_end(state):
        # if distribute:
        #     synchronize()
        if main_work:
            # output.shape: (all_data, num_box, 2)   label.shape: (all_data, 1, 2)
            # all_output = torch.cat(state["sorted_segment_list"], dim=0).cpu().tolist()
            # all_label = torch.cat(state["label_list"], dim=0).cpu().tolist()

            state['Rank@N,mIoU@M'], state['miou'] = \
                eval_predictions(state["sorted_segments_list"], state["label_list"], verbose=False)

            if config.VERBOSE:
                state["progress_bar"].close()

    engine = Engine()
    engine.hooks["on_start"] = on_start
    engine.hooks["on_start_epoch"] = on_start_epoch
    engine.hooks["on_forward"] = on_forward
    engine.hooks["on_update"] = on_update
    engine.hooks["on_end"] = on_end
    engine.hooks["on_test_start"] = on_test_start
    engine.hooks["on_test_forward"] = on_test_forward
    engine.hooks["on_test_end"] = on_test_end

    engine.train(network,
                 iterator("train"),
                 max_epoch=config.TRAIN.MAX_EPOCH,
                 optimizer=optimizer,
                 scheduler=scheduler,
                 step_accumulate=config.TRAIN.STEP_ACCUMULATE)
    # engine.test(network, iterator("test"), "test")
    if main_work:
        os._exit(0)


if __name__ == '__main__':
    main()
