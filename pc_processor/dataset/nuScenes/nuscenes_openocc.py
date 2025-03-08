import os
import torch
import numpy as np
from PIL import Image
import pickle
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud, LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
from pc_processor.dataset import occupancy_api
import numba as nb


map_name_from_general_to_segmentation_class = {
    'noise': 'ignore',
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.wheelchair': 'ignore',
    'human.pedestrian.stroller': 'ignore',
    'human.pedestrian.personal_mobility': 'ignore',
    'human.pedestrian.police_officer': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'animal': 'ignore',
    'vehicle.car': 'car',
    'vehicle.motorcycle': 'motorcycle',
    'vehicle.bicycle': 'bicycle',
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.truck': 'truck',
    'vehicle.construction': 'construction_vehicle',
    'vehicle.emergency.ambulance': 'ignore',
    'vehicle.emergency.police': 'ignore',
    'vehicle.trailer': 'trailer',
    'movable_object.barrier': 'barrier',
    'movable_object.trafficcone': 'traffic_cone',
    'movable_object.pushable_pullable': 'ignore',
    'movable_object.debris': 'ignore',
    'static_object.bicycle_rack': 'ignore',
    'flat.driveable_surface': 'driveable_surface',
    'flat.other': 'other_flat',
    'flat.sidewalk': 'sidewalk',
    'flat.terrain': 'terrain',
    'static.manmade': 'manmade',
    'static.vegetation': 'vegetation',
    'static.other': 'ignore',
    'vehicle.ego': 'ignore'
}

map_name_from_segmentation_class_to_segmentation_index = {
    'ignore': 0,
    'barrier': 1,
    'bicycle': 2,
    'bus': 3,
    'car': 4,
    'construction_vehicle': 5,
    'motorcycle': 6,
    'pedestrian': 7,
    'traffic_cone': 8,
    'trailer': 9,
    'truck': 10,
    'driveable_surface': 11,
    'other_flat': 12,
    'sidewalk': 13,
    'terrain': 14,
    'manmade': 15,
    'vegetation': 16
}

@nb.jit('u1[:,:,:](u1[:,:,:],i4[:,:])', nopython=True, cache=True, parallel=False)
def nb_process_label(processed_label, sorted_label_voxel_pair):
    label_size = 256
    counter = np.zeros((label_size,), dtype=np.uint16)
    counter[sorted_label_voxel_pair[0, 3]] = 1
    cur_sear_ind = sorted_label_voxel_pair[0, :3]
    for i in range(1, sorted_label_voxel_pair.shape[0]):
        cur_ind = sorted_label_voxel_pair[i, :3]
        if not np.all(np.equal(cur_ind, cur_sear_ind)):
            processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
            counter = np.zeros((label_size,), dtype=np.uint16)
            cur_sear_ind = cur_ind
        counter[sorted_label_voxel_pair[i, 3]] += 1
    processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
    return processed_label

class NuScenesOpenOcc(object):
    def __init__(self, 
                 data_root,
                 source_data_root,
                 split="train",
                 use_radar=False) -> None:
        
        self.data_root = data_root
        self.source_data_root = source_data_root
        self.label_root = os.path.join(data_root, "nuScenes-Occupancy-v0.1")
        
        self.use_radar  =use_radar
        self.nusc = NuScenes(
            version='v1.0-trainval', dataroot=source_data_root, verbose=False)
        
        if split == "train":
            self.is_train = True
            pkl_file = os.path.join(data_root, "nuscenes_occ_infos_train.pkl")
        else:
            self.is_train = False
            pkl_file = os.path.join(data_root, "nuscenes_occ_infos_val.pkl")
        
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        self.nusc_infos = data['infos']

        self.class_name_map = [
            'free', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',  
            'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',  
            'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade', 'vegetation'
        ]   # 0-17
        
        self.map_name_from_general_index_to_segmentation_index = {}
        for index in self.nusc.lidarseg_idx2name_mapping:
            self.map_name_from_general_index_to_segmentation_index[index] = \
                map_name_from_segmentation_class_to_segmentation_index[
                    map_name_from_general_to_segmentation_class[self.nusc.lidarseg_idx2name_mapping[index]]]
        
        self.camera_channnels = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']

        self.num_cams = len(self.camera_channnels)

    def loadImage(self, index, img_jitter=None):
        info = self.nusc_infos[index]
        img_data_list = []
        for _, cam_info in info['cams'].items():
            cam_path = cam_info['data_path']
            cam_path = cam_path.replace("./data/nuscenes", self.source_data_root)
            img_pil = Image.open(cam_path)
            if img_jitter is not None:
                img_pil = img_jitter(img_pil)
            img_data_list.append(np.array(img_pil) / 255.)
        return img_data_list

    def loadMultiSweepPCD(self, index, num_sweeps=10):
        # load multi frame
        info = self.nusc_infos[index]
        lidar_path = info['lidar_path'].replace("./data/nuscenes", self.source_data_root)      
        pcd_data = np.fromfile(lidar_path, dtype=np.float32, count=-1).reshape([-1, 5])
        pcd_data = pcd_data[:, :4]
        sweep_points_list = [pcd_data]
        if len(info['sweeps']) <= num_sweeps:
            choices = np.arange(len(info['sweeps']))
        else:
            choices = np.random.choice(len(info['sweeps']), num_sweeps, replace=False)
        
        for idx in choices:
            sweep = info['sweeps'][idx]
            sweep_path = sweep['data_path'].replace("./data/nuscenes", self.source_data_root)
            points_sweep = np.fromfile(sweep_path, dtype=np.float32, count=-1).reshape([-1, 5])
            points_sweep[:, :3] = points_sweep[:, :3] @ sweep['sensor2lidar_rotation'].T
            points_sweep[:, :3] += sweep['sensor2lidar_translation']
            points_sweep = points_sweep[:, :4]
            sweep_points_list.append(points_sweep)
        points = np.ascontiguousarray(np.concatenate(sweep_points_list, axis=0))
        return points
    
    def loadPCD(self, index):
        info = self.nusc_infos[index]
        lidar_path = info['lidar_path']   
        lidar_path = lidar_path.replace("./data/nuscenes", self.source_data_root)
        pcd_data = np.fromfile(lidar_path, dtype=np.float32).reshape((-1, 5))
        pcd_data = pcd_data[:, :4]
        return pcd_data

    def loadRadar(self, index):
        # radar data
        radar_type_list = ['RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT', 'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT']
        radar_ls = []
        sample_token = self.nusc_infos[index]['token']
        sample_data = self.nusc.get('sample', sample_token)
        for radar_type in radar_type_list:
            radar_data = self.nusc.get('sample_data', sample_data['data'][radar_type])
            radar_path = os.path.join(self.source_data_root, radar_data['filename'])
            pcd_radar = RadarPointCloud.from_file(radar_path)
            pcd_radar = pcd_radar.points.T
            pcd_radar = np.concatenate([
                pcd_radar[:, 0:2],
                pcd_radar[:, 6:10],
            ], axis=1)

            max_radar_points = 256
            num_points = pcd_radar.shape[0]
            if num_points > max_radar_points:
                choice_idx = torch.from_numpy(
                    np.random.choice(
                        np.arange(num_points), size=max_radar_points, replace=False)
                ).long()
                pad_radar = pcd_radar[choice_idx]
            elif num_points < max_radar_points:
                pad_radar = np.zeros((max_radar_points, pcd_radar.shape[1])).astype(np.float32)
                pad_radar[:num_points] = pcd_radar
            else:
                pad_radar = pcd_radar

            # radar permute
            if self.is_train:
                choice_idx = torch.from_numpy(
                np.random.choice(
                    np.arange(max_radar_points), size=max_radar_points, replace=False)
                ).long()
                pad_radar = pad_radar[choice_idx]
            
            radar_ls.append(np.expand_dims(pad_radar, axis=0))
        return radar_ls
    
    def loadOccLabel(self, pcd_data, image_data, voxels, index):
        info = self.nusc_infos[index]
        rel_path = 'scene_{}/occupancy/{}.npy'.format(info['scene_token'], info['lidar_token'])
        label = np.load(os.path.join(self.label_root, rel_path))  #  [z y x cls]
        occ_label = label[..., -1:]
        occ_label[occ_label==0] = 255       # noise --> 255
        occ_xyz_grid = label[..., [2,1,0]]  # z y x  -->  x y z
        label_voxel_pair = np.concatenate([occ_xyz_grid, occ_label], axis=-1)
        label_voxel_pair = label_voxel_pair[np.lexsort((occ_xyz_grid[:, 0], occ_xyz_grid[:, 1], occ_xyz_grid[:, 2])), :].astype(np.int32)
        voxel_label = np.zeros([512, 512, 40], dtype=np.uint8)
        voxel_label = nb_process_label(voxel_label, label_voxel_pair)  
        voxel_label = voxel_label.reshape(-1, 1)
        voxel_label = np.ascontiguousarray(voxel_label)  

        query_points = voxels.query_points
        rgb_result_all = np.zeros((query_points.shape[0], 3)).astype(np.float32)
        for i, (_, cam_info) in enumerate(info['cams'].items()):
            # RGB LABEL
            lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
            lidar2cam_t = cam_info[
                'sensor2lidar_translation'] @ lidar2cam_r.T
            lidar2cam_rt = np.eye(4)
            lidar2cam_rt[:3, :3] = lidar2cam_r.T
            lidar2cam_rt[3, :3] = -lidar2cam_t
            intrinsic = cam_info['cam_intrinsic']
            viewpad = np.eye(4)
            viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
            lidar2img_rt = (lidar2cam_rt @ viewpad.T)
            lidar2img_rt = np.ascontiguousarray(lidar2img_rt.T)   # transpose for mappingOccupancyRGB

            occupancy_api.mappingOccupancyRGB(
                query_points.astype(np.float32), 
                lidar2img_rt.astype(np.float32),  
                voxel_label.astype(np.float32),
                image_data[i].astype(np.float32), 
                np.array(voxels.voxel_range).astype(np.float32),
                voxels.grid_size, rgb_result_all
            )

        return voxel_label, rgb_result_all

    def loadSemLabel(self, index):
        info = self.nusc_infos[index]
        lidar_sample_token = info['lidar_token']

        lidarseg_path = os.path.join(
            self.source_data_root,
            self.nusc.get('lidarseg', lidar_sample_token)['filename'])
        annotated_data = np.fromfile(
            lidarseg_path, dtype=np.uint8).reshape((-1, 1))  # label
    
        sem_label = np.vectorize(self.map_name_from_general_index_to_segmentation_index.__getitem__)(
            annotated_data)  # n, 1
        assert sem_label.shape[-1] == 1
        sem_label = sem_label[:, 0]
        return sem_label
    
    def mapLidar2Camera(self, index, pointcloud, img_h, img_w, cam_idx=0):
        info = self.nusc_infos[index]
        lidar_sample_token = info['lidar_token']
        sample_token = info['token']
        pointsensor = self.nusc.get('sample_data', lidar_sample_token)

        assert pointsensor['is_key_frame'], \
            'Error: Only pointclouds which are keyframes have lidar segmentation labels. Rendering aborted.'
        assert pointsensor['sensor_modality'] == 'lidar', 'Error: Can only render lidarseg labels for lidar, ' \
            'not %s!' % pointsensor['sensor_modality']

        # Projects a pointcloud into a camera image along with the lidarseg labels
        cam_sample_token = self.nusc.get('sample', sample_token)['data'][self.camera_channnels[cam_idx]]
        cam = self.nusc.get('sample_data', cam_sample_token)

        pcl_path = os.path.join(self.source_data_root, pointsensor['filename'])
        pc = LidarPointCloud.from_file(pcl_path)

        # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
        # First step: transform the pointcloud to the ego vehicle frame for the timestamp of the sweep.
        cs_record = self.nusc.get(
            'calibrated_sensor', pointsensor['calibrated_sensor_token'])
        pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
        pc.translate(np.array(cs_record['translation']))

        # Second step: transform from ego to the global frame.
        poserecord = self.nusc.get('ego_pose', pointsensor['ego_pose_token'])
        pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
        pc.translate(np.array(poserecord['translation']))

        # Third step: transform from global into the ego vehicle frame for the timestamp of the image.
        poserecord = self.nusc.get('ego_pose', cam['ego_pose_token'])
        pc.translate(-np.array(poserecord['translation']))
        pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)

        # Fourth step: transform from ego into the camera.
        cs_record = self.nusc.get(
            'calibrated_sensor', cam['calibrated_sensor_token'])
        pc.translate(-np.array(cs_record['translation']))
        pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)

        # Fifth step: actually take a "picture" of the point cloud.
        # Grab the depths (camera frame z axis points away from the camera).
        depths = pc.points[2, :]

        # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
        points = view_points(pc.points[:3, :], np.array(
            cs_record['camera_intrinsic']), normalize=True)

        # Remove points that are either outside or behind the camera. Leave a margin of 1 pixel for aesthetic reasons.
        # Also make sure points are at least 1m in front of the camera to avoid seeing the lidar points on the camera
        # casing for non-keyframes which are slightly out of sync.
        mask = np.ones(depths.shape[0], dtype=bool)
        mask = np.logical_and(mask, depths > 0.5)
        mask = np.logical_and(mask, points[0, :] > 1)
        mask = np.logical_and(mask, points[0, :] < img_w - 1)
        mask = np.logical_and(mask, points[1, :] > 1)
        mask = np.logical_and(mask, points[1, :] < img_h - 1)

        mapped_points = points.transpose(1, 0)  # n, 3
        mapped_points = np.fliplr(mapped_points[:, :2])

        # fliplr so that indexing is row, col and not col, row
        return mapped_points[mask, :], mask  # (n, 3) (n, )

    def __len__(self):
        return len(self.nusc_infos)

