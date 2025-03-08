import random
import numpy as np

class GridVoxel(object):
    def __init__(self, grid_size=0.2, voxel_range=(0, 51.2, -25.6, 25.6, -2, 4.4)) -> None:
        self.bev_h = int((voxel_range[1] - voxel_range[0])/grid_size)
        self.bev_w = int((voxel_range[3] - voxel_range[2])/grid_size)
        self.bev_l = int((voxel_range[5] - voxel_range[4])/grid_size)

        query = np.arange(self.bev_h*self.bev_w*self.bev_l)
        query_x = query//(self.bev_w*self.bev_l)*grid_size+voxel_range[0]+grid_size/2
        query_y = query%(self.bev_w*self.bev_l)//self.bev_l*grid_size+voxel_range[2]+grid_size/2
        query_z = query%self.bev_l*grid_size+voxel_range[4]+grid_size/2
        self.query_points = np.concatenate((
            query_x.reshape(-1, 1),
            query_y.reshape(-1, 1),
            query_z.reshape(-1, 1),
        ), axis=1).astype(np.float32)

        self.grid_size = grid_size
        self.voxel_range = voxel_range
    
    def computeVoxelIndex(self, points):
        x_idx = ((points[:, 0]-self.grid_size/2-self.voxel_range[0])/self.grid_size).astype(np.int32)
        y_idx = ((points[:, 1]-self.grid_size/2-self.voxel_range[2])/self.grid_size).astype(np.int32)
        z_idx = ((points[:, 2]-self.grid_size/2-self.voxel_range[4])/self.grid_size).astype(np.int32)
        xyz_idx = np.concatenate([
            x_idx.reshape(-1, 1),
            y_idx.reshape(-1, 1),
            z_idx.reshape(-1, 1)
        ], axis=1).astype(np.int32)
        mask_h = np.logical_and(
            xyz_idx[:, 0] < self.bev_h,
            xyz_idx[:, 0] >= 0
        )
        mask_w = np.logical_and(
            xyz_idx[:, 1] < self.bev_w,
            xyz_idx[:, 1] >= 0
        )
        mask_l = np.logical_and(
            xyz_idx[:, 2] < self.bev_l,
            xyz_idx[:, 2] >= 0
        )
        valid_mask = np.logical_and(mask_h, mask_w)
        valid_mask = np.logical_and(valid_mask, mask_l)
            
        return xyz_idx, valid_mask

    def computeOccupancyMask(self, occupancy_data):
        mask = np.zeros(self.query_points.shape[0])
        x_idx = ((occupancy_data[:, 0]-self.grid_size/2-self.voxel_range[0])/self.grid_size).astype(np.int32)
        y_idx = ((occupancy_data[:, 1]-self.grid_size/2-self.voxel_range[2])/self.grid_size).astype(np.int32)
        z_idx = ((occupancy_data[:, 2]-self.grid_size/2-self.voxel_range[4])/self.grid_size).astype(np.int32)
        idx = x_idx * self.bev_w*self.bev_l + y_idx*self.bev_l + z_idx
        mask[idx] = 1
        return mask
    
    def getQueryPointsWithNoise(self):
        noise_xyz = (np.random.rand(self.query_points.shape[0], 3)-0.5)*0.1 * self.grid_size
        return self.query_points + noise_xyz