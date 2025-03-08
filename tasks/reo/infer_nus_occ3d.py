from option import Option
import os
import torch
import pc_processor
import numpy as np
import time
from tqdm import tqdm

from pc_processor.models import REONet


settings = Option("./config_nus_occ3d.yaml")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# model

model = REONet(
    backbone=settings.backbone, 
    img_size=settings.img_size, 
    use_pcd=settings.use_pcd,
    use_image=settings.use_img,
    num_classes=settings.num_classes,
    num_sem_classes=settings.num_sem_classes,
    pc_range=settings.voxel_range,
    grid_size=settings.grid_size,
    voxel_downscales=settings.voxel_downscales,
    num_images=settings.num_images)

infer_model = os.path.join(settings.infer_path, "checkpoint", settings.infer_model)
if not os.path.isfile(infer_model):
    raise FileNotFoundError("pretrained model not found")

state_dict = torch.load(infer_model, map_location="cpu")
model.load_state_dict(state_dict)

infer_tag = os.path.join(settings.infer_path, "Eval_{}-{}-{}_{}".format(
    settings.dataset, settings.net_type, settings.backbone,
    pc_processor.utils.getTimestamp()
))

save_path = os.path.join(settings.infer_path, infer_tag)
recorder = pc_processor.checkpoint.Recorder(
    settings, save_path, use_tensorboard=False)

if settings.save_infer_result:
    infer_result_folder = os.path.join(save_path, "infer_result")
    if not os.path.isdir(infer_result_folder):
        os.makedirs(infer_result_folder)


model.cuda()
model.eval()

# dataset
valset = pc_processor.dataset.NuScenesOcc3D(
    data_root=settings.data_root,
    source_data_root=settings.source_data_root,
    split="val",
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

val_data_info = valset.data_info

val_loader = torch.utils.data.DataLoader(
    occ_valset,
    batch_size=1,
    num_workers=settings.n_threads,
    shuffle=False,
    drop_last=False
)

# Metric_mIoU
iou_meter = pc_processor.utils.Metric_mIoU_online(
    num_classes=settings.num_classes)
geo_iou_meter = pc_processor.utils.Metric_mIoU_online(
    num_classes=2)

pbar = tqdm(total=len(valset))
for i, (input_imgs, pcd_voxels, _, fine_query, _) in enumerate(val_loader):
    with torch.no_grad():
        input_imgs = input_imgs.cuda()   
        pcd_voxels = pcd_voxels.cuda()   
        fine_query = fine_query.cuda()
        
        sem_preds, geo_preds, _, _ = model(
            input_imgs, 
            query_pose=fine_query[..., :3], 
            pcd_voxels=pcd_voxels)

        pred_rgb = geo_preds[..., :3]
        pred_cls = sem_preds
        query_class = pred_cls[0].argmax(1)

        # save pred, optional
        predictions_np = query_class.cpu().numpy()
        predictions_np = predictions_np.reshape((200, 200, 16))
        scene_index = val_data_info[i]['scene_index']
        sample_token = val_data_info[i]['sample_token']
        
        if settings.save_infer_result:
            pred_save_folder = os.path.join(
                infer_result_folder, scene_index, sample_token)
            if not os.path.isdir(pred_save_folder):
                os.makedirs(pred_save_folder)
            pred_save_path = os.path.join(
                pred_save_folder, "labels.npz")
        
            np.savez(pred_save_path, semantics=predictions_np)

        # save rgb, optional
        rgb_all_np = pred_rgb[0].cpu().numpy()
        if settings.save_infer_result:
            rgb_save_path = os.path.join(
                pred_save_folder, "rgb.npy")
            np.save(rgb_save_path, rgb_all_np)

        # evaluate
        mask_camera = fine_query[0, :, 3].ne(255)
        cls_label = fine_query[0, :, 3][mask_camera].long()
        preds_valid = query_class[mask_camera]

        preds_valid_np = preds_valid.cpu().numpy()
        cls_label_np = cls_label.cpu().numpy()
        cur_miou = iou_meter.addBatch(preds_valid_np, cls_label_np)

        geo_preds_valid_np = preds_valid.ne(settings.empty_idx).long().cpu().numpy()
        geo_label_np = cls_label.ne(settings.empty_idx).long().cpu().numpy()
        cur_geo_iou = geo_iou_meter.addBatch(geo_preds_valid_np, geo_label_np)

    pbar.update(1)
    pbar.set_postfix(
        meanIoU=round(np.nanmean(cur_miou[:17]) * 100, 2),
        geoIoU=round(cur_geo_iou[1] * 100, 2),
        )

pbar.close()

all_miou = iou_meter.getIoU()
all_geo_iou = geo_iou_meter.getIoU()
recorder.logger.info("============ all frames ============")
log_str = ">>> mIoU (0-16): {:0.2f}".format(round(np.nanmean(all_miou[:17]) * 100, 2))
recorder.logger.info(log_str)
log_str = ">>> Geometry IoU: {:0.2f}".format(round(all_geo_iou[1] * 100, 2))
recorder.logger.info(log_str)

recorder.logger.info("============ class iou ============")
for idx in range(settings.num_classes):
    log_str = ">>> class {} IoU = {}".format(
        idx, round(all_miou[idx] * 100, 2))
    recorder.logger.info(log_str)

recorder.logger.info("============ latext string ============")
latext_str = "& {:0.2f} ".format(round(np.nanmean(all_miou[:17]) * 100, 2))
for idx in range(settings.num_classes - 1):
    latext_str += "& {:0.2f} ".format(round(all_miou[idx] * 100, 2))
recorder.logger.info(latext_str)

recorder.logger.info("============ end ============")