python -m torch.distributed.launch --nproc_per_node 2 train.py --cfg config/Charades-STA.yaml --gpus 0, 1