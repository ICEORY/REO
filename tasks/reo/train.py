import numpy as np
import torch
import torch.nn as nn
import time
from option import Option
import torch.nn as nn
import torch.nn.functional as F
import datetime
import pc_processor
import math
def tensorNormalize(x):
    x_max = x.max()
    x_min = x.min()
    return (x - x_min)/(x_max-x_min)

class Trainer(object):
    def __init__(self, settings: Option, model: nn.Module, recorder=None):
        # init params
        self.settings = settings
        self.recorder = recorder
        self.model = model.cuda()
        self.remain_time = pc_processor.utils.RemainTime(
            self.settings.n_epochs)

        # init data loader
        self.train_loader, self.val_loader, self.train_sampler, self.val_sampler = self._initDataloader()

        # init criterion
        self.criterion = self._initCriterion()
        
        # init optimizer
        self.optimizer = self._initOptimizer()

        # set multi gpu
        if self.settings.n_gpus > 1:
            if self.settings.distributed:
                # sync bn
                self.model = pc_processor.layers.sync_bn.replaceBN(
                    self.model).cuda()
                self.model = nn.parallel.DistributedDataParallel(
                    self.model, device_ids=[self.settings.gpu] , 
                    find_unused_parameters=True)
            else:
                self.model = nn.DataParallel(self.model)
                for k, v in self.criterion.items():
                    self.criterion[k] = nn.DataParallel(v).cuda()
        
        self.scheduler = pc_processor.utils.WarmupCosineLR(
            optimizer=self.optimizer,
            lr=self.settings.lr,
            warmup_steps=self.settings.warmup_epochs *
            len(self.train_loader),
            momentum=self.settings.momentum,
            max_steps=len(self.train_loader) * (self.settings.n_epochs-self.settings.warmup_epochs))
          
        self.semantic_iou_meter = pc_processor.metrics.IOUEval(
            n_classes=self.settings.num_classes,
            is_distributed=self.settings.distributed, device=torch.device("cpu"))

        self.geo_iou_meter = pc_processor.metrics.IOUEval(
            n_classes=2,
            is_distributed=self.settings.distributed, device=torch.device("cpu"))

        self.semantic_2d_iou_meter = pc_processor.metrics.IOUEval(
            n_classes=self.settings.num_sem_classes, 
            ignore=[0], is_distributed=self.settings.distributed, 
            device=torch.device("cpu")
        )
    # ------------------------------------------------------------------
    # functions for initialization
    # ------------------------------------------------------------------
    def _initOptimizer(self):
        # check params
        adam_params = []
        adam_params.append({"params": self.model.parameters()})

        adam_opt = torch.optim.AdamW(
            params=adam_params, lr=self.settings.lr,
            weight_decay=self.settings.weight_decay,
            amsgrad=True
        )
        return adam_opt

    def _initDataloader(self):
        if self.settings.dataset == "SemanticKITTI":
            trainset = pc_processor.dataset.SemanticKitti(
                data_root=self.settings.data_root,
                use_label_refactor=False,
                split="train"
            )
            valset = pc_processor.dataset.SemanticKitti(
                data_root=self.settings.data_root,
                use_label_refactor=False,
                split="val"
            )
        elif self.settings.dataset == "nuScenesOcc3D":
            trainset = pc_processor.dataset.NuScenesOcc3D(
                data_root=self.settings.data_root,
                source_data_root=self.settings.source_data_root,
                split="train",
                use_radar=False
            )
            valset = pc_processor.dataset.NuScenesOcc3D(
                data_root=self.settings.data_root,
                source_data_root=self.settings.source_data_root,
                split="val",
                use_radar=False
            )
        elif self.settings.dataset == "nuScenesOpenOcc":
            trainset = pc_processor.dataset.NuScenesOpenOcc(
                data_root=self.settings.data_root,
                source_data_root=self.settings.source_data_root,
                split="train",
                use_radar=False
            )
            valset = pc_processor.dataset.NuScenesOpenOcc(
                data_root=self.settings.data_root,
                source_data_root=self.settings.source_data_root,
                split="val",
                use_radar=False
            )
        else:
            raise NotImplementedError(self.settings.dataset)

        occ_trainset = pc_processor.dataset.OccupancyLoader(
            dataset=trainset, img_size=self.settings.img_size,
            crop_img_size=self.settings.crop_img_size,
            max_pointcloud_voxels=self.settings.num_pc_voxels,
            voxel_range=self.settings.voxel_range,
            grid_size=self.settings.grid_size,
            use_pcd=self.settings.use_pcd,
            use_multisweeps=self.settings.use_multi_sweeps
        )

        occ_valset = pc_processor.dataset.OccupancyLoader(
            dataset=valset, img_size=self.settings.img_size,
            crop_img_size=self.settings.crop_img_size,
            max_pointcloud_voxels=self.settings.num_pc_voxels,
            voxel_range=self.settings.voxel_range,
            grid_size=self.settings.grid_size,
            use_pcd=self.settings.use_pcd,
            use_multisweeps=self.settings.use_multi_sweeps
        )

        self.class_name_map = trainset.class_name_map
        if self.settings.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                occ_trainset, shuffle=True, drop_last=True)
            val_sampler = torch.utils.data.distributed.DistributedSampler(
                occ_valset, shuffle=False, drop_last=False)
            
            train_loader = torch.utils.data.DataLoader(
                occ_trainset,
                batch_size=self.settings.batch_size[0],
                num_workers=self.settings.n_threads,
                drop_last=True,
                sampler=train_sampler
            )

            val_loader = torch.utils.data.DataLoader(
                occ_valset,
                batch_size=self.settings.batch_size[1],
                num_workers=self.settings.n_threads,
                drop_last=False,
                sampler=val_sampler
            )
            return train_loader, val_loader, train_sampler, val_sampler

        else:
            train_loader = torch.utils.data.DataLoader(
                occ_trainset,
                batch_size=self.settings.batch_size[0],
                num_workers=self.settings.n_threads,
                shuffle=True,
                drop_last=True)

            val_loader = torch.utils.data.DataLoader(
                occ_valset,
                batch_size=self.settings.batch_size[1],
                num_workers=self.settings.n_threads,
                shuffle=False,
                drop_last=False
            )
            return train_loader, val_loader, None, None

    def computeClsWeight(self, cls_freq, ignore_zero=False, ignore_idx=0):
        if ignore_zero:
            cls_freq[0] = 0
        cls_freq = cls_freq / cls_freq.sum()
        cls_weight = 1 / (cls_freq + 1e-8)
        if ignore_zero:
            cls_weight[ignore_idx] = 0
        alpha = np.log(1+cls_weight)
        if ignore_zero:
            alpha[ignore_idx] = 0
        alpha = alpha / alpha.max()
        if self.recorder is not None:
            self.recorder.logger.info("class weights: {}".format(alpha))
        return alpha

    def _initCriterion(self):
        criterion = {}
        # loss for texture cues
        criterion["rgb_loss"] = nn.SmoothL1Loss(beta=0.05, reduction="none")

        # loss for semantic occupancy
        alpha = self.computeClsWeight(
            np.array(self.settings.cls_freq), 
            ignore_zero=False, ignore_idx=self.settings.empty_idx)
        criterion["sem_focal_loss"] = pc_processor.loss.FocalSoftmaxLoss( 
            self.settings.num_classes, gamma=2, softmax=False, alpha=alpha)
        criterion["sem_dice_loss"] = pc_processor.loss.InvertDiceLoss()
        
        if self.settings.dataset == "nuScenesOcc3D":
            alpha = self.computeClsWeight(np.array([
                self.settings.cls_freq[self.settings.num_classes-1], 
                sum(self.settings.cls_freq[:self.settings.num_classes-1])]))
        else:
            alpha = self.computeClsWeight(np.array([
                self.settings.cls_freq[0], sum(self.settings.cls_freq[1:])]))

        criterion["geo_focal_loss"] = pc_processor.loss.FocalSoftmaxLoss( 
            2, gamma=2, softmax=False, alpha=alpha)
        criterion["geo_dice_loss"] = pc_processor.loss.InvertDiceLoss()

        # loss for 2d semantic segmentation
        alpha = self.computeClsWeight(
            np.array(self.settings.cls_freq[:self.settings.num_sem_classes]), ignore_zero=True,
            ignore_idx=0)
        criterion["sem2d_focal_loss"] = pc_processor.loss.FocalSoftmaxLoss(
            self.settings.num_sem_classes, gamma=2, alpha=alpha, softmax=False)
        criterion["sem2d_dice_loss"] = pc_processor.loss.InvertDiceLoss()
        
        # loss for depth
        criterion["depth_loss"] = nn.SmoothL1Loss(beta=0.05, reduction="none")

        # set device
        for _, v in criterion.items():
            v.cuda()
            
        return criterion

    def _computeProjectionAuxLosses(
            self, sem_pred, rgb_pred, depth_pred, sem_label, rgb_label, depth_label, label_mask):
        proj_focal_loss = self.criterion["sem2d_focal_loss"](sem_pred, sem_label, mask=label_mask)
        proj_dice_loss = self.criterion["sem2d_dice_loss"](sem_pred, sem_label, mask=label_mask)

        proj_depth_loss = self.criterion["depth_loss"](
            depth_pred*label_mask.unsqueeze(1), depth_label).sum()/(label_mask.sum()+1e-6)
        proj_rgb_loss = self.criterion["rgb_loss"](
            rgb_pred*label_mask.unsqueeze(1), rgb_label).sum()/(label_mask.sum()*3+1e-6)

        loss = proj_focal_loss*self.settings.w_focal + \
               proj_dice_loss*self.settings.w_dice + \
               proj_depth_loss*self.settings.w_depth + \
               proj_rgb_loss*self.settings.w_rgb
            
        loss_dict = {
            "FocalLoss": proj_focal_loss,
            "DiceLoss": proj_dice_loss,
            "DepthLoss": proj_depth_loss,
            "RGBLoss": proj_rgb_loss
        }
        return loss, loss_dict
    
    def _compute3DGeoLoss(self, geo_pred, geo_label, rgb_pred, rgb_label, rgb_mask):
        # compute rgb_loss for 3d cues
        rgb_loss = self.criterion["rgb_loss"](rgb_pred * rgb_mask, rgb_label).sum()/(rgb_mask.sum()*3+1e-6)

        # compute geometry loss
        pred_softmax = F.softmax(geo_pred, dim=1)
        geo_focal_loss = self.criterion["geo_focal_loss"](pred_softmax, geo_label)
        geo_dice_loss = self.criterion["geo_dice_loss"](pred_softmax, geo_label)

        loss = geo_focal_loss*self.settings.w_focal+\
                geo_dice_loss*self.settings.w_dice +\
                self.settings.w_rgb*rgb_loss
                
        loss_dict = {
            "FocalLoss": geo_focal_loss,
            "DiceLoss": geo_dice_loss,
            "RGBLoss": rgb_loss,
        }
        return loss, loss_dict
    
    def _compute3DLoss(self, sem_pred, sem_label):
        # compute semantic loss
        pred_softmax = F.softmax(sem_pred, dim=1)
        sem_focal_loss = self.criterion["sem_focal_loss"](pred_softmax, sem_label)
        sem_dice_loss = self.criterion["sem_dice_loss"](pred_softmax, sem_label)
        
        loss = sem_focal_loss*self.settings.w_focal+\
                sem_dice_loss*self.settings.w_dice

        loss_dict = {
            "FocalLoss": sem_focal_loss,
            "DiceLoss": sem_dice_loss,
        }
        return loss, loss_dict

    # ---------------------------------------------------------------------
    # 
    # ---------------------------------------------------------------------
    def run(self, epoch, mode="Train"):
        if self.settings.distributed:
            torch.distributed.barrier()
        if mode == "Train":
            dataloader = self.train_loader
            self.model.train()
            if self.settings.distributed:
                self.train_sampler.set_epoch(epoch)

        elif mode == "Validation":
            dataloader = self.val_loader
            self.model.eval()
        else:
            raise ValueError("invalid mode: {}".format(mode))

        self.geo_iou_meter.reset()
        self.semantic_iou_meter.reset()
        self.semantic_2d_iou_meter.reset()

        loss_meter_dict = {}
        total_iter = len(dataloader)
        t_start = time.time()
        rgb_rmse_meter = pc_processor.utils.AverageMeter()
        rgb_rmse_meter.reset()
        depth_rmse_meter = pc_processor.utils.AverageMeter()
        depth_rmse_meter.reset()
        proj_rgb_rmse_meter = pc_processor.utils.AverageMeter()
        proj_rgb_rmse_meter.reset()

        for i, (input_imgs, pcd_voxels, proj_data, fine_query, radar_points) in enumerate(dataloader):
            # ======================================================
            t_process_start = time.time()
            input_imgs = input_imgs.cuda()
            pcd_voxels = pcd_voxels.cuda()
            fine_query = fine_query.cuda()
            proj_data = proj_data.cuda()
            
            proj_label_mask = proj_data[:, 3].gt(0)
            proj_sem_label = proj_data[:, 5].long()
            proj_sem_mask = proj_sem_label.gt(0)
            proj_label_mask = proj_label_mask * proj_sem_mask

            proj_depth_label = proj_data[:, 4:5]*proj_label_mask.unsqueeze(1).float()
            proj_rgb_label = proj_data[:, 6:9]*proj_label_mask.unsqueeze(1).float()
            proj_rgb_label = proj_rgb_label * proj_label_mask.unsqueeze(1)

            proj_loss_dict = None
            geo_loss_dict = None
            spatial_loss_dict = None

            if mode == "Train":
                # forward
                sem_preds, geo_preds, proj_pred, voxel_preds = self.model(
                    input_imgs, 
                    query_pose=fine_query[..., :3], 
                    aug_imgs=proj_data[:, :3], 
                    pcd_voxels=pcd_voxels,
                    radar_points=radar_points)
                
                # compute entropy      
                voxel_pred_softmax = voxel_preds.softmax(dim=1)
                full_pred_log = torch.log(voxel_pred_softmax.clamp(min=1e-8))
                num_classes = full_pred_log.size(1)
                full_pred_entropy = - (voxel_pred_softmax * full_pred_log).sum(1) / math.log(num_classes)
                
                rgb_label = fine_query[..., 4:7] * fine_query[..., :, 3:4].lt(255).float() * fine_query[..., :, 3:4].gt(0).float()
                classify_label = fine_query[..., 3:4].view(-1, 1).long()
                label_mask = classify_label.lt(255).squeeze(1) # 255 means ignore
                classify_label = classify_label[label_mask]
                geo_label = classify_label.ne(self.settings.empty_idx).long()
                rgb_mask = rgb_label.abs().sum(2).gt(1e-6).unsqueeze(2)
                
                if self.settings.use_img:
                    proj_sem_pred = proj_pred[:, :self.settings.num_sem_classes]
                    proj_depth_pred = proj_pred[:, self.settings.num_sem_classes:self.settings.num_sem_classes+1]
                    proj_rgb_pred = proj_pred[:, self.settings.num_sem_classes+1:]
                    
                    # compute entropy
                    proj_sem_log = torch.log(proj_sem_pred.clamp(min=1e-8))
                    proj_sem_entropy = - (proj_sem_pred*proj_sem_log).sum(1)/math.log(self.settings.num_sem_classes)

                    # compute proj loss
                    proj_loss, proj_loss_dict = self._computeProjectionAuxLosses(
                        sem_pred=proj_sem_pred, sem_label=proj_sem_label,
                        rgb_pred=proj_rgb_pred, rgb_label=proj_rgb_label,
                        depth_pred=proj_depth_pred, depth_label=proj_depth_label,
                        label_mask=proj_label_mask
                    )
                # -------------------------------
                # compute geo loss
                pred_geo_cls = geo_preds[..., 3:]
                pred_rgb = geo_preds[..., :3]

                pred_geo_cls = pred_geo_cls.view(-1, 2)
                pred_geo_cls = pred_geo_cls[label_mask]
                geo_loss, geo_loss_dict = self._compute3DGeoLoss(
                    geo_pred=pred_geo_cls, geo_label=geo_label,
                    rgb_pred=pred_rgb, rgb_label=rgb_label, rgb_mask=rgb_mask
                )

                # compute 3d occ result
                pred_cls = sem_preds.view(-1, self.settings.num_classes)
                pred_cls = pred_cls[label_mask]

                spatial_loss, spatial_loss_dict = self._compute3DLoss(
                    sem_pred=pred_cls, sem_label=classify_label,
                )

                loss = 0
                if self.settings.use_img:
                    if self.settings.use_proj_loss:
                        loss +=  proj_loss + spatial_loss + geo_loss 
                    else:
                        loss += spatial_loss + geo_loss 
                else:
                    loss += spatial_loss + geo_loss
                
                if self.settings.n_gpus > 1:
                    loss = loss.mean()
               
                # backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            # ======================================================    
            else:
                with torch.no_grad():
                    # forward
                    sem_preds, geo_preds, proj_pred, voxel_preds = self.model(
                        input_imgs,
                        query_pose=fine_query[..., :3], 
                        aug_imgs=proj_data[:, :3],
                        pcd_voxels=pcd_voxels,
                        radar_points=radar_points)
                    
                    # compute entropy        
                    voxel_pred_softmax = voxel_preds.softmax(dim=1)
                    full_pred_log = torch.log(voxel_pred_softmax.clamp(min=1e-8))
                    num_classes = full_pred_log.size(1)
                    full_pred_entropy = - (voxel_pred_softmax * full_pred_log).sum(1) / math.log(num_classes)

                    rgb_label = fine_query[..., 4:7] * fine_query[..., 3:4].lt(255).float() * fine_query[..., 3:4].gt(0).float()
                    classify_label = fine_query[..., 3:4].view(-1, 1).long()
                    label_mask = classify_label.lt(255).squeeze(1) # 255 means ignore
                    classify_label = classify_label[label_mask]
                    
                    geo_label = classify_label.ne(self.settings.empty_idx).long()
                    rgb_mask = rgb_label.abs().sum(2).gt(1e-6).unsqueeze(2)
                    
                    if self.settings.use_img:
                        proj_sem_pred = proj_pred[:, :self.settings.num_sem_classes]
                        proj_depth_pred = proj_pred[:, self.settings.num_sem_classes:self.settings.num_sem_classes+1]
                        proj_rgb_pred = proj_pred[:, self.settings.num_sem_classes+1:]
                        
                        # compute entropy
                        proj_sem_log = torch.log(proj_sem_pred.clamp(min=1e-8))
                        proj_sem_entropy = - (proj_sem_pred*proj_sem_log).sum(1)/math.log(self.settings.num_sem_classes)

                        # compute proj loss
                        proj_loss, proj_loss_dict = self._computeProjectionAuxLosses(
                            sem_pred=proj_sem_pred, sem_label=proj_sem_label,
                            rgb_pred=proj_rgb_pred, rgb_label=proj_rgb_label,
                            depth_pred=proj_depth_pred, depth_label=proj_depth_label,
                            label_mask=proj_label_mask
                        )

                    # -------------------------------
                    # compute geo loss
                    pred_geo_cls = geo_preds[..., 3:]
                    pred_rgb = geo_preds[..., :3]

                    pred_geo_cls = pred_geo_cls.view(-1, 2)
                    pred_geo_cls = pred_geo_cls[label_mask]
                    geo_loss, geo_loss_dict = self._compute3DGeoLoss(
                        geo_pred=pred_geo_cls, geo_label=geo_label,
                        rgb_pred=pred_rgb, rgb_label=rgb_label, rgb_mask=rgb_mask
                    )
                    
                    # compute 3d occ result
                    pred_cls = sem_preds.view(-1, self.settings.num_classes)
                    pred_cls = pred_cls[label_mask]

                    spatial_loss, spatial_loss_dict = self._compute3DLoss(
                        sem_pred=pred_cls, sem_label=classify_label
                    )

                    loss = 0
                    if self.settings.use_img:
                        if self.settings.use_proj_loss:
                            loss +=  proj_loss + spatial_loss + geo_loss 
                        else:
                            loss += spatial_loss + geo_loss 
                    else:
                        loss += spatial_loss + geo_loss
                    
                    if self.settings.n_gpus > 1:
                        loss = loss.mean()

            with torch.no_grad():
                if rgb_mask.sum() > 0:
                    rgb_rmse = ((pred_rgb.view(-1, 3) * rgb_mask.view(-1, 1) - rgb_label.view(-1, 3) * rgb_mask.view(-1, 1)
                                ).pow(2).sum()/ rgb_mask.sum() / 3).sqrt()
                    rgb_rmse_meter.update(rgb_rmse.cpu().item())

                self.semantic_iou_meter.addBatch(pred_cls.argmax(dim=1), classify_label)
                occupancy_label = classify_label.ne(self.settings.empty_idx).long()
                pred_occupancy = pred_cls.argmax(dim=1).ne(self.settings.empty_idx).long()
                self.geo_iou_meter.addBatch(pred_occupancy, occupancy_label)

                if self.settings.use_img:
                    depth_rmse = ((proj_depth_pred*proj_label_mask.unsqueeze(1)-proj_depth_label).pow(2).sum()/proj_label_mask.sum()).sqrt()
                    depth_rmse_meter.update(depth_rmse.cpu().item())
                    proj_rgb_rmse = ((proj_rgb_pred*proj_label_mask.unsqueeze(1)-proj_rgb_label).pow(2).sum()/proj_label_mask.sum()/3).sqrt()
                    proj_rgb_rmse_meter.update(proj_rgb_rmse.cpu().item())
                    self.semantic_2d_iou_meter.addBatch(proj_sem_pred.argmax(dim=1), proj_sem_label)

            total_loss = loss.cpu().item()
            if "TotalLoss" not in loss_meter_dict.keys():
                loss_meter_dict["TotalLoss"] = pc_processor.utils.AverageMeter()
                loss_meter_dict["TotalLoss"].reset()
            loss_meter_dict["TotalLoss"].update(total_loss)
            
            for k, v in spatial_loss_dict.items():
                key_name = "3D{}".format(k)
                if key_name not in loss_meter_dict.keys():
                    loss_meter_dict[key_name] = pc_processor.utils.AverageMeter()
                    loss_meter_dict[key_name].reset()
                loss_meter_dict[key_name].update(v.cpu().item())

            for k, v in geo_loss_dict.items():
                key_name = "3DGeo{}".format(k)
                if key_name not in loss_meter_dict.keys():
                    loss_meter_dict[key_name] = pc_processor.utils.AverageMeter()
                    loss_meter_dict[key_name].reset()
                loss_meter_dict[key_name].update(v.cpu().item())

            if self.settings.use_img:
                for k, v in proj_loss_dict.items():
                    key_name = "2D{}".format(k)
                    if key_name not in loss_meter_dict.keys():
                        loss_meter_dict[key_name] = pc_processor.utils.AverageMeter()
                        loss_meter_dict[key_name].reset()
                    loss_meter_dict[key_name].update(v.cpu().item())
            
            _, geo_iou = self.geo_iou_meter.getIoU()
            _, geo_acc = self.geo_iou_meter.getAcc()
            _, geo_recall = self.geo_iou_meter.getRecall()

            _, sem_iou = self.semantic_iou_meter.getIoU()
            _, sem_acc = self.semantic_iou_meter.getAcc()
            _, sem_recall = self.semantic_iou_meter.getRecall()

            if self.settings.use_img:
                mean_sem_2d_iou, cls_sem_2d_iou = self.semantic_2d_iou_meter.getIoU()
                mean_sem_2d_acc, cls_sem_2d_acc = self.semantic_2d_iou_meter.getAcc()
                mean_sem_2d_recall, cls_sem_2d_recall = self.semantic_2d_iou_meter.getRecall()

            if mode == "Train":
                self.scheduler.step()
        
            # timer logger ----------------------------------------
            t_process_end = time.time()

            data_cost_time = t_process_start - t_start
            process_cost_time = t_process_end - t_process_start

            self.remain_time.update(cost_time=(time.time()-t_start), mode=mode)
            remain_time = datetime.timedelta(
                seconds=self.remain_time.getRemainTime(
                    epoch=epoch, iters=i, total_iter=total_iter, mode=mode
                ))
            t_start = time.time()

            for g in self.optimizer.param_groups:
                lr = g["lr"]
                break
            log_str = ">>> {} E[{:02d}|{:02d}] I[{:04d}|{:04d}] DT[{:.2f}] PT[{:.2f}] ".format(
                mode, self.settings.n_epochs, epoch+1, total_iter, i+1, data_cost_time, process_cost_time)
            
            if self.settings.use_img:
                log_str += "LR {:0.6f} Loss {:0.2f} RGBRMSE {:0.4f} DepthRMSE {:0.4f} 2DmIoU {:0.4f} | ".format(
                    lr, total_loss, 
                    proj_rgb_rmse_meter.avg,
                    depth_rmse_meter.avg,
                    mean_sem_2d_iou.item(),
                ) 
            else:
                log_str += "LR {:0.5f} Loss {:0.4f} ".format(
                    lr, total_loss,
                ) 
            log_str += "RMSE {:0.4f} SCIoU {:0.4f} 3DmIoU {:0.4f} ".format(
                rgb_rmse_meter.avg,
                geo_iou[1].item(), 
                sem_iou[:self.settings.num_classes-1].mean().item() if self.settings.dataset=="nuScenesOcc3D" else sem_iou[1:].mean().item()
            )

            log_str += "RT {}".format(remain_time)

            if self.recorder is not None:
                self.recorder.logger.info(log_str)

            if self.settings.is_debug:
                break
        
        # -----------------------------------------------------------------------
        if self.recorder is not None:
            # img log 
            if epoch % self.settings.print_frequency == 0:
                for idx in range(full_pred_entropy.size(1)):
                    self.recorder.tensorboard.add_image("{}_Entropy_{:02d}".format(
                        mode, idx
                    ), full_pred_entropy[0:1, idx, :, :], epoch)

                self.recorder.tensorboard.add_image("{}_Image".format(mode), input_imgs[0, 0], epoch)

                if self.settings.use_img:
                    self.recorder.tensorboard.add_image("{}_ProjEntropy".format(mode), tensorNormalize(proj_sem_entropy[0:1]), epoch)
                    self.recorder.tensorboard.add_image("{}_ProjDepth".format(mode), tensorNormalize(proj_depth_pred[0:1, 0]), epoch)
                    self.recorder.tensorboard.add_image("{}_ProjImagePred".format(mode), proj_rgb_pred[0], epoch)
                
            # scalar log
            ## loss
            for k, v in loss_meter_dict.items():
                self.recorder.tensorboard.add_scalar(
                    tag="{}_{}".format(mode, k), scalar_value=v.avg, global_step=epoch)
            ## lr        
            self.recorder.tensorboard.add_scalar(
                tag="{}_lr".format(mode), scalar_value=lr, global_step=epoch)

            ## mtloss
            if mode == "Train" and self.use_mtloss:
                sigma = self.mt_loss.module.sigma
                for idx in range(sigma.size(0)):
                    self.recorder.tensorboard.add_scalar(
                        tag="{}_LossWeight_{}".format(mode, idx), scalar_value=sigma[idx].item(), global_step=epoch)
            ## mse 
            self.recorder.tensorboard.add_scalar(
                tag="{}_RGBRMSE".format(mode), scalar_value=rgb_rmse_meter.avg, global_step=epoch)
            
            if self.settings.dataset == "nuScenesOcc3D":
                self.recorder.tensorboard.add_scalar(
                    tag="{}_SSCMeanAcc".format(mode), scalar_value=sem_acc[:self.settings.num_classes-1].mean().item(), 
                    global_step=epoch)
                self.recorder.tensorboard.add_scalar(
                    tag="{}_SSCMeanRecall".format(mode), scalar_value=sem_recall[:self.settings.num_classes-1].mean().item(), 
                    global_step=epoch)
                self.recorder.tensorboard.add_scalar(
                    tag="{}_SSCMeanIoU".format(mode), scalar_value=sem_iou[:self.settings.num_classes-1].mean().item(), 
                    global_step=epoch)

            else:
                self.recorder.tensorboard.add_scalar(
                    tag="{}_SSCMeanAcc".format(mode), scalar_value=sem_acc[1:].mean().item(), global_step=epoch)
                self.recorder.tensorboard.add_scalar(
                    tag="{}_SSCMeanRecall".format(mode), scalar_value=sem_recall[1:].mean().item(), global_step=epoch)
                self.recorder.tensorboard.add_scalar(
                    tag="{}_SSCMeanIoU".format(mode), scalar_value=sem_iou[1:].mean().item(), global_step=epoch)

            self.recorder.tensorboard.add_scalar(
                tag="{}_EmptyAcc".format(mode), scalar_value=geo_acc[0].item(), global_step=epoch)
            self.recorder.tensorboard.add_scalar(
                tag="{}_EmptyRecall".format(mode), scalar_value=geo_recall[0].item(), global_step=epoch)
            self.recorder.tensorboard.add_scalar(
                tag="{}_EmptyIoU".format(mode), scalar_value=geo_iou[0].item(), global_step=epoch)

            self.recorder.tensorboard.add_scalar(
                tag="{}_SCAcc".format(mode), scalar_value=geo_acc[1].item(), global_step=epoch)
            self.recorder.tensorboard.add_scalar(
                tag="{}_SCRecall".format(mode), scalar_value=geo_recall[1].item(), global_step=epoch)
            self.recorder.tensorboard.add_scalar(
                tag="{}_SCIoU".format(mode), scalar_value=geo_iou[1].item(), global_step=epoch)
            
            for cls_idx in range(self.settings.num_classes):
                cls_name = self.class_name_map[cls_idx]
                # 3D semantic
                self.recorder.tensorboard.add_scalar(
                    tag="{}_{:02d}{}_Acc".format(mode, cls_idx, cls_name), 
                    scalar_value=sem_acc[cls_idx].item(), global_step=epoch)
                self.recorder.tensorboard.add_scalar(
                    tag="{}_{:02d}{}_Recall".format(mode, cls_idx, cls_name), 
                    scalar_value=sem_recall[cls_idx].item(), global_step=epoch)
                self.recorder.tensorboard.add_scalar(
                    tag="{}_{:02d}{}_IoU".format(mode, cls_idx, cls_name), 
                    scalar_value=sem_iou[cls_idx].item(), global_step=epoch)
                
            
            if self.settings.use_img:    
                for cls_idx in range(self.settings.num_sem_classes):
                    # 2D semantic
                    cls_name = self.class_name_map[cls_idx]
                    self.recorder.tensorboard.add_scalar(
                        tag="{}_2D_{:02d}{}_Acc".format(mode, cls_idx, cls_name), 
                        scalar_value=cls_sem_2d_acc[cls_idx].item(), global_step=epoch)
                    self.recorder.tensorboard.add_scalar(
                        tag="{}_2D_{:02d}{}_Recall".format(mode, cls_idx, cls_name), 
                        scalar_value=cls_sem_2d_recall[cls_idx].item(), global_step=epoch)
                    self.recorder.tensorboard.add_scalar(
                        tag="{}_2D_{:02d}{}_IoU".format(mode, cls_idx, cls_name), 
                        scalar_value=cls_sem_2d_iou[cls_idx].item(), global_step=epoch)
                    
                self.recorder.tensorboard.add_scalar(
                    tag="{}_DepthRMSE".format(mode), scalar_value=depth_rmse_meter.avg, global_step=epoch)
                self.recorder.tensorboard.add_scalar(
                    tag="{}_2DRGBRMSE".format(mode), scalar_value=proj_rgb_rmse_meter.avg, global_step=epoch)
                self.recorder.tensorboard.add_scalar(
                    tag="{}_2DSemMeanAcc".format(mode), scalar_value=mean_sem_2d_acc, global_step=epoch)
                self.recorder.tensorboard.add_scalar(
                    tag="{}_2DSemMeanRecall".format(mode), scalar_value=mean_sem_2d_recall, global_step=epoch)
                self.recorder.tensorboard.add_scalar(
                    tag="{}_2DSemMeanIoU".format(mode), scalar_value=mean_sem_2d_iou, global_step=epoch)

        if self.settings.dataset == "nuScenesOcc3D":
            ssc_miou = sem_iou[:self.settings.num_classes-1].mean().item()
        else:
            ssc_miou = sem_iou[1:].mean().item()
        result_dict = {
            "SCIoU": geo_iou[1].item(),
            "SSCMeanIoU": ssc_miou,
            "last": 0
        }
        if self.recorder is not None:
            self.recorder.logger.info("{} SCIoU {:0.4f} SSCMeanIoU: {:0.4f}".format(
                mode, geo_iou[1].item(), ssc_miou))
        return result_dict