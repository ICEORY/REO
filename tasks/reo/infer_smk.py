from option import Option
import os
import torch
import pc_processor
import numpy as np
import time
from tqdm import tqdm
from pc_processor.models import REONet

LATEXT_CLASS_MAP = {
    "road": 9,
    "sidewalk": 11,
    "parking": 10,
    "other-ground":12,
    "building": 13,
    "car": 1,
    "truck": 4,
    "bicycle": 2,
    "motorcycle": 3,
    "other-vehicle": 5,
    "vegetation": 15,
    "trunk": 16,
    "terrain": 17,
    "person": 6,
    "bicyclist": 7,
    "motorcyclist": 8,
    "fence": 14,
    "pole": 18,
    "traffic-sign": 19
}

settings = Option("./config_smk.yaml")
os.environ["CUDA_VISIBLE_DEVICES"] = "0" #  settings.gpu

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
valset = pc_processor.dataset.SemanticKitti(
    data_root=settings.data_root,
    use_label_refactor=False,
    split="val"
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
    batch_size=1,
    num_workers=settings.n_threads,
    shuffle=False,
    drop_last=False
)

class_name_map = valset.class_name_map

class_iou_meter = pc_processor.metrics.IOUEval(
    n_classes=settings.num_classes,
    is_distributed=False, device=torch.device("cpu"))
occupancy_iou_meter = pc_processor.metrics.IOUEval(
    n_classes=2,
    is_distributed=False, device=torch.device("cpu"))

pbar = tqdm(range(len(val_loader)))
t_start = time.time()
for i, (input_imgs, pcd_voxels, _, fine_query, _) in enumerate(val_loader):
    with torch.no_grad():
        input_imgs = input_imgs.cuda()   
        pcd_voxels = pcd_voxels.cuda()   
        fine_query = fine_query.cuda()
        
        sem_preds, geo_preds, _, _= model(
            input_imgs, 
            query_pose=fine_query[..., :3], 
            pcd_voxels=pcd_voxels)

        pred_rgb = geo_preds[:, :, :3]
        pred_cls = sem_preds

        pred_cls_all = pred_cls.squeeze(0).argmax(dim=1)
        valid_label_mask = fine_query[0, :, 3].ne(255)
        cls_label = fine_query[0, :, 3][valid_label_mask].long()
        pred_cls_valid = pred_cls_all[valid_label_mask]
        class_iou_meter.addBatch(pred_cls_valid, cls_label)
        occupancy_iou_meter.addBatch(pred_cls_valid.ge(1).long(), cls_label.ge(1).long())

        _, class_iou = class_iou_meter.getIoU()
        _, class_acc = class_iou_meter.getAcc()
        _, class_recall = class_iou_meter.getRecall()
        
        _, fine_class_iou = occupancy_iou_meter.getIoU()
        _, fine_class_acc = occupancy_iou_meter.getAcc()
        _, fine_class_recall = occupancy_iou_meter.getRecall()
        
        pred_result = torch.cat([
            fine_query[:, :, :3], 
            pred_cls.argmax(dim=2).unsqueeze(2), 
            pred_rgb], dim=2)[0]
        valid_mask = pred_result[:, 3].ge(1)
        pred_result = pred_result[valid_mask]
        query_label = fine_query[0]
        valid_mask = query_label[:, 3].ge(1)
        query_label = query_label[valid_mask]
        valid_mask = query_label[:, 3].ne(255)
        query_label = query_label[valid_mask]

        if settings.save_infer_result:
            np.save(os.path.join(infer_result_folder, "{:04d}_result.npy".format(i)) , {
                "occ_map": pred_result.cpu().numpy(),
                "img": input_imgs[0].cpu().numpy(), 
                "label": query_label.cpu().numpy()
            })
        pbar.update(1)
        pbar.set_postfix(meanIoU=class_iou[1:].mean().item(), OccupyIoU=fine_class_iou[1].item())
        # break

cost_time = time.time() - t_start
recorder.logger.info("cost time: {}".format(cost_time))
# record results 
recorder.logger.info("mean iou: {:0.4f}% | mean acc: {:0.4f}% | mean recall: {:0.4f}%".format(
    class_iou[1:].mean().item()*100, class_acc[1:].mean().item()*100, class_recall[1:].mean().item()*100
))
recorder.logger.info("occupy | iou: {:0.4f}% | acc: {:0.4f}% | recall: {:0.4f}%".format(
    fine_class_iou[1].item()*100, fine_class_acc[1].item()*100, fine_class_recall[1].item()*100))
for i in range(settings.num_classes):
    recorder.logger.info("{} | iou: {:0.4f}% | acc: {:0.4f}% | recall: {:0.4f}%".format(
        class_name_map[i], class_iou[i].item()*100, class_acc[i].item()*100, class_recall[i].item()*100))

recorder.logger.info("==== latext string =====")
latext_str = "& {:0.2f} & {:0.2f} ".format(fine_class_iou[1].item()*100, class_iou[1:].mean().item()*100)
for k,v in LATEXT_CLASS_MAP.items():
    latext_str += "& {:0.2f} ".format(class_iou[v].item()*100)
recorder.logger.info(latext_str)