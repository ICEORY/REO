import os
import torch
import numpy as np
import json
from PIL import Image
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud, LidarPointCloud
from nuscenes.utils.geometry_utils import view_points


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

class NuScenesOcc3D(object):
    def __init__(self, 
                 data_root,
                 source_data_root,
                 split="train",
                 use_radar=False) -> None:
        
        self.data_root = data_root
        self.source_data_root = source_data_root
        self.use_radar  =use_radar
        self.nusc = NuScenes(
            version='v1.0-trainval', dataroot=source_data_root, verbose=False)

        anno_path = os.path.join(data_root, "annotations.json")
        with open(anno_path, 'r') as anno_file:
            self.annotations = json.load(anno_file)
        
        if split == "train":
            self.is_train = True
            self.scene_list = self.annotations['train_split']
        else:
            self.is_train = False
            self.scene_list = self.annotations['val_split']

        self.data_info = []
        for scene_index in self.scene_list:
            for sample_token in self.annotations['scene_infos'][scene_index].keys():
                sample_info = {
                    'sample_token': sample_token,
                    'scene_index': scene_index
                }
                self.data_info.append(sample_info)

        self.class_name_map = [
            'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',  
            'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',  
            'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade', 'vegetation', 'free'
        ]   # 0-18
        
        self.map_name_from_general_index_to_segmentation_index = {}
        for index in self.nusc.lidarseg_idx2name_mapping:
            self.map_name_from_general_index_to_segmentation_index[index] = \
                map_name_from_segmentation_class_to_segmentation_index[
                    map_name_from_general_to_segmentation_class[self.nusc.lidarseg_idx2name_mapping[index]]]
        
        self.camera_channnels = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']

        self.num_cams = len(self.camera_channnels)

    def loadImage(self, index, img_jitter=None):
        scene_index = self.data_info[index]['scene_index']
        sample_token = self.data_info[index]['sample_token']
        sample_info = self.annotations['scene_infos'][scene_index][sample_token]
        img_data_list = []
        for _, cam_info in sample_info['camera_sensor'].items():
            img_pil = Image.open(os.path.join(self.data_root, "imgs", cam_info['img_path']))
            if img_jitter is not None:
                img_pil = img_jitter(img_pil)
            img_data_list.append(np.array(img_pil) / 255.)
        return img_data_list
    
    def loadMultiSweepPCD(self, index, num_sweeps=10):
        raise NotImplementedError
    
    def loadPCD(self, index):
        sample_token = self.data_info[index]['sample_token']
        sample_data = self.nusc.get('sample', sample_token)
        lidar_data = self.nusc.get('sample_data', sample_data['data']['LIDAR_TOP'])
        lidar_path = os.path.join(self.source_data_root, lidar_data['filename'])
        pcd_data = np.fromfile(lidar_path, dtype=np.float32).reshape((-1, 5))
        pcd_data = pcd_data[:, :4]
        return pcd_data

    def loadRadar(self, index):
        # radar data
        radar_type_list = ['RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT', 'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT']
        radar_ls = []
        sample_token = self.data_info[index]['sample_token']
        sample_data = self.nusc.get('sample', sample_token)
        for radar_type in radar_type_list:
            radar_data = self.nusc.get('sample_data', sample_data['data'][radar_type])
            radar_path = os.path.join(self.source_data_root, radar_data['filename'])
            pcd_radar = RadarPointCloud.from_file(radar_path)
            pcd_radar = pcd_radar.points.T
            # x y z dyn_prop id rcs vx vy vx_comp vy_comp is_quality_valid ambig_state x_rms y_rms invalid_state pdh0 vx_rms vy_rms
            
            mask = pcd_radar[:, 14] == 0
            pcd_radar = pcd_radar[mask]
            mask = pcd_radar[:, 3] < 7
            pcd_radar = pcd_radar[mask]
            pcd_radar = np.concatenate([
                pcd_radar[:, 0:2],
                pcd_radar[:, 8:10],
            ], axis=1)

            max_radar_points = 256
            num_points = pcd_radar.shape[0]
            if num_points >= max_radar_points:
                choice_idx = torch.from_numpy(
                    np.random.choice(
                        np.arange(num_points), size=max_radar_points, replace=False)
                ).long()
                pad_radar = pcd_radar[choice_idx]
            else:
                num_pad_points = max_radar_points - num_points
                choice_idx = torch.from_numpy(
                    np.random.choice(
                        np.arange(num_points), size=num_pad_points, replace=True)
                ).long()
                pad_radar = pcd_radar[choice_idx]
                pad_radar = np.concatenate([pcd_radar, pad_radar], axis=0)

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
        scene_index = self.data_info[index]['scene_index']
        sample_token = self.data_info[index]['sample_token']
        sample_info = self.annotations['scene_infos'][scene_index][sample_token]
        gt = np.load(os.path.join(self.data_root, sample_info['gt_path']))
        voxel_label = gt['semantics']
        mask_camera = gt['mask_camera']
        voxel_label = voxel_label.reshape(-1, 1)
        mask_camera = mask_camera.reshape(-1, 1)
        voxel_label[np.isclose(mask_camera, 0)] = 255    # mask_camera

        # compute RGB guidence
        query_points = voxels.query_points
        rgb_mask = mask_camera.reshape(-1) == 1
        rgb_result_all = np.zeros((query_points.shape[0], 3)).astype(np.float32)
        rgb_result = rgb_result_all[rgb_mask]
        
        xyz_proj = query_points[rgb_mask]
        ego_pose = sample_info['ego_pose']

        for i, (_, cam_info) in enumerate(sample_info['camera_sensor'].items()):
            cur_img = image_data[i]

            cam_ego_pose = cam_info['ego_pose']
            cam_extrinsic = cam_info['extrinsic']
            cam_intrinsics = cam_info['intrinsics']
            pixel_coors, mapped_mask = self.mapOccQuery2Camera(
                xyz_proj, ego_pose, cam_ego_pose, cam_extrinsic, cam_intrinsics,
                img_h=cur_img.shape[0], img_w=cur_img.shape[1])
            pixel_u = pixel_coors[:, 1].astype(np.int32)    # 1600
            pixel_v = pixel_coors[:, 0].astype(np.int32)    # 900
            rgb_result[mapped_mask] = cur_img[pixel_v, pixel_u, :]

        rgb_result_all[rgb_mask] = rgb_result

        return voxel_label, rgb_result_all

    def loadSemLabel(self, index):
        sample_token = self.data_info[index]['sample_token']
        lidar_sample_token = self.nusc.get('sample', sample_token)['data']['LIDAR_TOP']

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
        sample_token = self.data_info[index]['sample_token']
        lidar_sample_token = self.nusc.get('sample', sample_token)['data']['LIDAR_TOP']
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
    
    @staticmethod
    def translate(xyz_T, trans: np.ndarray) -> None:
        """
        Applies a translation to the point cloud.
        :param xyz_T: <np.float: 3, n>. PointCloud.
        :param trans: <np.float: 3, 1>. Translation in x, y, z.
        """
        for i in range(3):
            xyz_T[i, :] = xyz_T[i, :] + trans[i]
        return xyz_T

    @staticmethod
    def rotate(xyz_T, rot_matrix: np.ndarray) -> None:
        """
        Applies a rotation.
        :param xyz_T: <np.float: 3, n>. PointCloud.
        :param rot_matrix: <np.float: 3, 3>. Rotation matrix.
        """
        return np.dot(rot_matrix, xyz_T[:3, :])
    
    def mapOccQuery2Camera(self, xyz, ego_pose, cam_ego_pose, cam_extrinsic, cam_intrinsics, img_h, img_w):
        xyz_T = xyz.T

        xyz_T = self.rotate(xyz_T, Quaternion(ego_pose['rotation']).rotation_matrix)
        xyz_T = self.translate(xyz_T, np.array(ego_pose['translation']))

        xyz_T = self.translate(xyz_T, -np.array(cam_ego_pose['translation']))
        xyz_T = self.rotate(xyz_T, Quaternion(cam_ego_pose['rotation']).rotation_matrix.T)

        xyz_T = self.translate(xyz_T, -np.array(cam_extrinsic['translation']))
        xyz_T = self.rotate(xyz_T, Quaternion(cam_extrinsic['rotation']).rotation_matrix.T)

        depths = xyz_T[2, :]
        
        viewpad = np.eye(4)
        cam_intrinsics = np.array(cam_intrinsics)
        viewpad[:cam_intrinsics.shape[0], :cam_intrinsics.shape[1]] = cam_intrinsics
        xyz_T = np.concatenate((xyz_T, np.ones((1, xyz_T.shape[1]))))
        xyz_T = np.dot(viewpad, xyz_T)
        xyz_T = xyz_T[:3, :]
        xyz_T = xyz_T / xyz_T[2:3, :].repeat(3, 0).reshape(3, xyz_T.shape[1])

        mask = np.ones(depths.shape[0], dtype=bool)
        mask = np.logical_and(mask, depths > 0)
        mask = np.logical_and(mask, xyz_T[0, :] > 1)
        mask = np.logical_and(mask, xyz_T[0, :] < img_w - 1)
        mask = np.logical_and(mask, xyz_T[1, :] > 1)
        mask = np.logical_and(mask, xyz_T[1, :] < img_h - 1)

        mapped_points = xyz_T.transpose(1, 0)  # n, 3
        mapped_points = np.fliplr(mapped_points[:, :2])

        return mapped_points[mask], mask

    def __len__(self):
        return len(self.data_info)
