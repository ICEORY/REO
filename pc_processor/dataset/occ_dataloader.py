import torch
import numpy as np
from torch.utils.data import Dataset
import cv2
from torchvision import transforms
import random 

from pc_processor.dataset import occupancy_api
from .common import GridVoxel

class OccupancyLoader(Dataset):
    def __init__(self, 
                 dataset,
                 img_size=(352, 1184), 
                 crop_img_size=(256, 1024),
                 max_pointcloud_voxels=10240, 
                 grid_size=0.2,
                 voxel_range=(0, 51.2, -25.6, 25.6, -2, 4.4),
                 use_pcd=False,
                 use_multisweeps=False):
        self.dataset = dataset
        self.is_train = dataset.is_train
        self.img_size = img_size
        self.crop_img_size = crop_img_size
        self.grid_size = grid_size
        self.voxel_range = voxel_range
        self.voxels = GridVoxel(grid_size=grid_size, voxel_range=voxel_range)
        self.max_pointcloud_voxels = max_pointcloud_voxels
        self.use_pcd = use_pcd
        self.use_multisweeps = use_multisweeps

        if self.is_train:
            self.img_sem_aug = transforms.Compose([
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomRotation(15),
                transforms.RandomCrop(size=self.crop_img_size, pad_if_needed=True)
            ])
        else:
            self.img_sem_aug = transforms.CenterCrop(size=self.crop_img_size)

        self.img_jitter = transforms.ColorJitter(0.4, 0.4, 0.4)

        print("loading dataset with {} samples: ".format(len(self.dataset)))

    def processProjData(self, img_data, pcd_data, index, cam_idx):
        raw_img_h = img_data.shape[0]
        raw_img_w = img_data.shape[1]
        xy_index, keep_mask = self.dataset.mapLidar2Camera(
            index=index, pointcloud=pcd_data[:, :3], 
            img_h=raw_img_h, img_w=raw_img_w, cam_idx=cam_idx)
        
        if self.is_train:
            img_scale = np.random.uniform(low=1.0, high=1.5)
        else:
            img_scale = 1
        img_aux = cv2.resize(
            img_data.copy(), dsize=(int(self.crop_img_size[1]*img_scale), int(self.crop_img_size[0]*img_scale)))

        aux_img_h = img_aux.shape[0]
        aux_img_w = img_aux.shape[1]

        h_scale = aux_img_h/raw_img_h 
        w_scale = aux_img_w/raw_img_w 
        x_data = np.floor(xy_index[:, 0]*h_scale).astype(np.int32)
        y_data = np.floor(xy_index[:, 1]*w_scale).astype(np.int32)
        
        proj_depth = np.zeros((aux_img_h, aux_img_w), dtype=np.float32)
        proj_label = np.zeros((aux_img_h, aux_img_w), dtype=np.int32)
        proj_mask = np.zeros((aux_img_h, aux_img_w), dtype=np.int32)
        proj_rgb = np.zeros((aux_img_h, aux_img_w, 3), dtype=np.float32)

        sem_label = self.dataset.loadSemLabel(index)
        sem_label = sem_label[keep_mask]
        keep_points = pcd_data[keep_mask]
        keep_depth = np.linalg.norm(keep_points, axis=1)

        proj_depth[x_data, y_data] = keep_depth
        
        proj_label[x_data, y_data] = sem_label
        proj_mask[x_data, y_data] = 1
        proj_rgb[x_data, y_data] = img_aux[x_data, y_data]
            
        proj_tensor = torch.cat((
            torch.from_numpy(img_aux).permute(2, 0, 1).float(), # 0-2
            torch.from_numpy(proj_mask).float().unsqueeze(0), # 3
            torch.from_numpy(proj_depth).unsqueeze(0), # 4
            torch.from_numpy(proj_label).float().unsqueeze(0), # 5
            torch.from_numpy(proj_rgb).permute(2, 0, 1).float() # 6-8
        ), dim=0)

        # apply data augmentation to projected data
        if self.img_sem_aug is not None:
            proj_tensor = self.img_sem_aug(proj_tensor)

        mask_depth = proj_tensor[4].masked_select(proj_tensor[3].eq(1))
        max_depth = mask_depth.max()
        min_depth = mask_depth.min()
        proj_tensor[4] = (proj_tensor[4] - min_depth)/(max_depth-min_depth)*proj_tensor[3]

        return proj_tensor

    def processPCDVoxels(self, pcd_data):
        pcd_voxels = occupancy_api.pointcloudVoxelize(pcd_data.astype(np.float32), self.voxels.grid_size)
        # compute depth
        pcd_range = np.linalg.norm(pcd_voxels[:, :3], axis=1, keepdims=True)
        # x,y,z,i,density, depth
        pcd_voxels = np.concatenate([pcd_voxels, pcd_range], axis=1)
        # normalize
        pcd_voxels[:, 4] = np.log(pcd_voxels[:, 4]+1)

        if self.max_pointcloud_voxels == -1:
            return torch.from_numpy(pcd_voxels)
        max_voxels= self.max_pointcloud_voxels
        num_voxels = pcd_voxels.shape[0]
        if num_voxels >= max_voxels:
            choice_idx = torch.from_numpy(
                np.random.choice(
                    np.arange(num_voxels), size=max_voxels, replace=False)
            ).long()
            pad_voxels = pcd_voxels[choice_idx]
        elif num_voxels < max_voxels:
            try:
                num_pad_voxels = max_voxels - num_voxels
                choice_idx = torch.from_numpy(
                    np.random.choice(
                        np.arange(num_voxels), size=num_pad_voxels, replace=True)
                ).long()   
                noise_voxels = pcd_voxels[choice_idx]
                if noise_voxels.ndim == 1:
                    noise_voxels = noise_voxels.reshape(1, -1)
                if self.is_train:
                    noise_voxels[:, :3] += (np.random.random((num_pad_voxels, 3)) - 0.5)*self.grid_size*0.1
                pad_voxels = np.concatenate([pcd_voxels, noise_voxels], axis=0)
            except:
                print(choice_idx.shape, pcd_voxels.shape, noise_voxels.shape)
                assert False
        
        # voxel permute
        if self.is_train:
            choice_idx = np.random.choice(
                np.arange(max_voxels), size=max_voxels, replace=False)
            pad_voxels = pad_voxels[choice_idx]

        pcd_voxels = torch.from_numpy(pad_voxels)

        return pad_voxels

    def __getitem__(self, index):
        # if self.is_train:
        #     img_data = self.dataset.loadImage(index, self.img_jitter)
        # else:
        img_data = self.dataset.loadImage(index)
        pcd_data = self.dataset.loadPCD(index)
        if isinstance(img_data, list):
            num_cams = len(img_data)
            if num_cams > 2:
                sem_img_index = random.randint(0, num_cams-1)
            else:
                sem_img_index = 0
            sem_img = img_data[sem_img_index]
            resize_img_list = []
            for img in img_data:
                resize_img_list.append(
                    torch.from_numpy(
                        cv2.resize(img.copy(), dsize=(self.img_size[1], self.img_size[0]))
                    ).unsqueeze(0) # N, H, W, C
                )
            img_tensor = torch.cat(resize_img_list, dim=0).permute(0, 3, 1, 2).float()

        else:
            num_cams = 1
            sem_img_index = 0
            sem_img = img_data

            img_data_resize = cv2.resize(
                img_data.copy(), dsize=(self.img_size[1], self.img_size[0]))
            img_tensor = torch.from_numpy(
                img_data_resize
            ).permute(2, 0, 1).float() # C, H, W
            img_tensor = img_tensor.unsqueeze(0) # N, C, H, W

        proj_data_tensor = self.processProjData(
            img_data=sem_img,
            pcd_data=pcd_data,
            index=index,
            cam_idx=sem_img_index
        )

        if isinstance(img_data, list) and len(img_data) == 2:
            voxel_label, rgb_result = self.dataset.loadOccLabel(
                pcd_data=pcd_data, image_data=img_data[0], voxels=self.voxels, index=index)

        else:
            voxel_label, rgb_result = self.dataset.loadOccLabel(
                pcd_data=pcd_data, image_data=img_data, voxels=self.voxels, index=index)

        if self.use_multisweeps:
            ms_pcd_data = self.dataset.loadMultiSweepPCD(index)
        else:
            ms_pcd_data = pcd_data

        if self.use_pcd:
            pcd_voxel_tensor = self.processPCDVoxels(ms_pcd_data)
        else:
            pcd_voxel_tensor = torch.zeros(10, 6)
            
        query_points = self.voxels.query_points
        query_tensor = torch.from_numpy(
            np.concatenate((
                query_points, 
                voxel_label, 
                rgb_result), axis=1)
        ).float()

        if hasattr(self.dataset, "use_radar") and self.dataset.use_radar:
            radar_points = self.dataset.loadRadar(index)
            radar_tensor = torch.from_numpy(
                np.concatenate(radar_points, axis=0)
            )
        else:
            radar_tensor = torch.zeros(5, 10, 4)
        
        if self.is_train:
            # speedup training on OpenOccupancy
            max_points = min(query_tensor.size(0), 2048000)
            choice_idx = np.random.choice(
                np.arange(query_tensor.size(0)), size=max_points, replace=False)
            query_tensor = query_tensor.index_select(dim=0, index=torch.from_numpy(choice_idx))

        return img_tensor, pcd_voxel_tensor, proj_data_tensor, query_tensor, radar_tensor 
        # return pcd_voxel_tensor, torch.from_numpy(pcd_data)

    def __len__(self):
        return len(self.dataset)