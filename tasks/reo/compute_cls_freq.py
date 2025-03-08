from option import Option
import os
import torch
import pc_processor
import numpy as np
from tqdm import tqdm

# settings = Option("./config_smk.yaml")
settings = Option("./config_nus_openocc.yaml")
# settings = Option("./config_nus_occ3d.yaml")
os.environ["CUDA_VISIBLE_DEVICES"] = settings.gpu

if settings.dataset == "SemanticKITTI":
    valset = pc_processor.dataset.SemanticKitti(
        data_root=settings.data_root,
        use_label_refactor=False,
        split="train"
    )
elif settings.dataset == "nuScenesOcc3D":
    valset = pc_processor.dataset.NuScenesOcc3D(
        data_root=settings.data_root,
        source_data_root=settings.source_data_root,
        split="train",
        use_radar=False
    )
elif settings.dataset == "nuScenesOpenOcc":
    valset = pc_processor.dataset.NuScenesOpenOcc(
        data_root=settings.data_root,
        source_data_root=settings.source_data_root,
        split="train",
        use_radar=False
    )

occ_valset = pc_processor.dataset.OccupancyLoader(
    dataset=valset, img_size=settings.img_size,
    crop_img_size=settings.crop_img_size,
    max_pointcloud_voxels=settings.num_pc_voxels,
    voxel_range=settings.voxel_range,
    grid_size=settings.grid_size,
    use_pcd=settings.use_pcd
)

val_loader = torch.utils.data.DataLoader(
    occ_valset,
    batch_size=8,
    num_workers=8,
    shuffle=False,
    drop_last=False
)
num_batch = len(val_loader)
freq = np.zeros(settings.num_classes)
total_samples = 0
for i, (input_imgs, pcd_voxels, proj_data, fine_query, fine_labels, _) in tqdm(enumerate(val_loader)):
    fine_query_np = fine_query.numpy()
    for j in range(settings.num_classes):
        mask = fine_query_np[:, :, 3] == j
        freq[j] += mask.sum()
    total_samples += fine_query.size(0)
    # break
print("dataset: {}".format(settings.dataset))
print("all cls freq: ", freq)
print("avg cls freq: ", freq/total_samples)
   