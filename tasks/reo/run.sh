## semantic kitti
python -m torch.distributed.launch --nproc_per_node=4 --master_port=63440 --use_env main.py config_smk.yaml

## nuscenes occ3d
# python -m torch.distributed.launch --nproc_per_node=4 --master_port=63420 --use_env main.py config_nus_occ3d.yaml

## nuscenes openocc
# python -m torch.distributed.launch --nproc_per_node=8 --master_port=63410 --use_env main.py config_nus_openocc.yaml