"""
Some of the code in this file is taken from https://github.com/cv-rits/LMSCNet/blob/main/LMSCNet/data/io_data.py
"""

import os
import yaml
import numpy as np
from PIL import Image
from pc_processor.dataset import occupancy_api

def unpack(compressed):
    ''' given a bit encoded voxel grid, make a normal voxel grid out of it.  '''
    uncompressed = np.zeros(compressed.shape[0] * 8, dtype=np.uint8)
    uncompressed[::8] = compressed[:] >> 7 & 1
    uncompressed[1::8] = compressed[:] >> 6 & 1
    uncompressed[2::8] = compressed[:] >> 5 & 1
    uncompressed[3::8] = compressed[:] >> 4 & 1
    uncompressed[4::8] = compressed[:] >> 3 & 1
    uncompressed[5::8] = compressed[:] >> 2 & 1
    uncompressed[6::8] = compressed[:] >> 1 & 1
    uncompressed[7::8] = compressed[:] & 1

    return uncompressed

def get_remap_lut(path):
    '''
    remap_lut to remap classes of semantic kitti for training...
    :return:
    '''

    dataset_config = yaml.safe_load(open(path, 'r'))

    # make lookup table for mapping
    maxkey = max(dataset_config['learning_map'].keys())

    # +100 hack making lut bigger just in case there are unknown labels
    remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
    remap_lut[list(dataset_config['learning_map'].keys())] = list(dataset_config['learning_map'].values())

    # in completion we have to distinguish empty and invalid voxels.
    # Important: For voxels 0 corresponds to "empty" and not "unlabeled".
    remap_lut[remap_lut == 0] = 255  # map 0 to 'invalid'
    remap_lut[0] = 0  # only 'empty' stays 'empty'.

    return remap_lut

# from scpnet
def label_rectification(grid_ind, voxel_label, instance_label, 
                        dynamic_classes=[1,4,5,6,7,8],
                        voxel_shape=(256,256,32),
                        ignore_class_label=255):
    
    segmentation_label = voxel_label[grid_ind[:,0], grid_ind[:,1], grid_ind[:,2]]
    
    for c in dynamic_classes:
        voxel_pos_class_c = (voxel_label==c).astype(int)
        instance_label_class_c = instance_label[segmentation_label==c].squeeze(1)
        
        if len(instance_label_class_c) == 0:
            pos_to_remove = voxel_pos_class_c
        
        elif len(instance_label_class_c) > 0 and np.sum(voxel_pos_class_c) > 0:
            mask_class_c = np.zeros(voxel_shape, dtype=int)
            point_pos_class_c = grid_ind[segmentation_label==c]
            uniq_instance_label_class_c = np.unique(instance_label_class_c)
            
            for i in uniq_instance_label_class_c:
               point_pos_instance_i = point_pos_class_c[instance_label_class_c==i]
               x_max, y_max, z_max = np.amax(point_pos_instance_i, axis=0)
               x_min, y_min, z_min = np.amin(point_pos_instance_i, axis=0)
               
               mask_class_c[x_min:x_max,y_min:y_max,z_min:z_max] = 1
        
            pos_to_remove = (voxel_pos_class_c - mask_class_c) > 0
        
        voxel_label[pos_to_remove] = ignore_class_label
            
    return voxel_label

class SemanticKitti(object):
    def __init__(
            self, 
            data_root, 
            use_label_refactor=False,
            split="train",
        ):

        self.data_root = data_root
        self.use_label_refactor = use_label_refactor
        self.split = split

        if split == "train":
            self.is_train = True
        else:
            self.is_train = False

        splits = {
            "train": ["00", "01", "02", "03", "04", "05", "06", "07", "09", "10"],
            "val": ["08"],
            "test": ["11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21"],
        }
        self.sequences = splits[split]

        self.img_file_list = []
        self.proj_matrix = {}
        for seq in self.sequences:
            current_img_file_list = os.listdir(os.path.join(self.data_root, "sequences", seq, "image_2"))
            current_img_file_list.sort()
            for img_file_name in current_img_file_list:
                voxel_file_path = os.path.join(
                    self.data_root, "sequences", seq, "voxels", img_file_name.replace(".png", ".label"))
                voxel_valid_path = os.path.join(
                    self.data_root, "sequences", seq, "voxels", img_file_name.replace(".png", ".invalid"))
                if os.path.isfile(voxel_file_path) and os.path.isfile(voxel_valid_path):
                    self.img_file_list.append(
                        os.path.join(seq, "image_2", img_file_name)
                    )
            calib_path = os.path.join(self.data_root, "sequences", seq, "calib.txt")
            calib = self.read_calib(calib_path)
            proj_matrix = np.matmul(calib["P2"], calib["Tr"])
            self.proj_matrix[seq] = proj_matrix

        self.remap_lut = get_remap_lut("../../pc_processor/dataset/semantic_kitti/semantic-kitti.yaml")
        smk_config = yaml.safe_load(open("../../pc_processor/dataset/semantic_kitti/semantic-kitti.yaml", "r"))
        learning_map = smk_config["learning_map"]
        max_key = 0
        for k, v in learning_map.items():
            if k > max_key:
                max_key = k
        # +100 hack making lut bigger just in case there are unknown labels
        self.class_map_lut = np.zeros((max_key + 100), dtype=np.int32)
        for k, v in learning_map.items():
            self.class_map_lut[k] = v
        
        self.class_name_map = smk_config["mapped_class_name"]
        

    def loadImage(self, index, img_jitter=None):
        img_file_path = self.img_file_list[index]
        # load img
        img_pil = Image.open(os.path.join(self.data_root, "sequences", img_file_path))
        if img_jitter is not None:
            img_pil = img_jitter(img_pil)
        img_np = np.array(img_pil) / 255.
        return img_np

        # img_pil_2 = Image.open(os.path.join(self.data_root, "sequences", img_file_path.replace("image_2", "image_3")))
        # if img_jitter is not None:
        #     img_pil = img_jitter(img_pil_2)
        # img_np_2 = np.array(img_pil_2) / 255.
        # return [img_np, img_np_2]
    
    def loadMultiSweepPCD(self, index, num_sweeps=10):
        raise NotImplementedError
    
    def loadPCD(self, index):
        img_file_path_split = self.img_file_list[index].split("/")
        seq_id = img_file_path_split[0]
        frame_id = img_file_path_split[2].strip(".png")
        pcd_data_path = os.path.join(self.data_root, "sequences", seq_id, "velodyne", frame_id+".bin")
        pcd = np.fromfile(pcd_data_path, dtype=np.float32).reshape(-1, 4)
        return pcd
    
    def loadOccLabel(self, pcd_data, image_data, voxels, index):
        img_file_path_split = self.img_file_list[index].split("/")
        seq_id = img_file_path_split[0]
        frame_id = img_file_path_split[2].strip(".png")
        voxel_label_path = os.path.join(self.data_root, "sequences", seq_id, "voxels", frame_id+".label")
        voxel_invalid_path = os.path.join(self.data_root, "sequences", seq_id, "voxels", frame_id+".invalid")
        voxel_label = np.fromfile(voxel_label_path, dtype=np.uint16)
        voxel_label = self.remap_lut[voxel_label].astype(np.float32)
        voxel_invalid = np.fromfile(voxel_invalid_path, dtype=np.uint8)
        voxel_invalid = unpack(voxel_invalid)
        voxel_label[np.isclose(voxel_invalid, 1)] = 255

        if self.use_label_refactor:
            refact_label_path = os.path.join(self.data_root, "sequences", seq_id, "voxels_refact", frame_id+".npy")
            refact_label_found = False
            if os.path.isfile(refact_label_path):
                try:
                    refact_label = np.load(refact_label_path)
                    refact_label_found = True
                except:
                    refact_label_found = False
            if not refact_label_found:
                # label refactorized (following scpnet)
                voxel_grid, mask = voxels.computeVoxelIndex(pcd_data[:, :3])
                
                inst_label = inst_label[mask].reshape(-1, 1)
                voxel_grid = voxel_grid[mask]
                voxel_label = voxel_label.reshape([256, 256, 32])
                voxel_label = label_rectification(voxel_grid, voxel_label, inst_label)
                save_root = os.path.join(self.data_root, "sequences", seq_id, "voxels_refact")
                if not os.path.isdir(save_root):
                    os.makedirs(save_root)
                np.save(refact_label_path, voxel_label)
            else:
                voxel_label = refact_label

        voxel_label = voxel_label.reshape(-1, 1)

        query_points = voxels.query_points
        proj_matrix_pad = np.eye(4)
        proj_matrix_pad[:3] = self.getProjMatrix(index)
        rgb_result = np.zeros((query_points.shape[0], 3)).astype(np.float32)

        occupancy_api.mappingOccupancyRGB(
            query_points.astype(np.float32), 
            proj_matrix_pad.astype(np.float32),  
            voxel_label.astype(np.float32),
            image_data.astype(np.float32), 
            np.array(voxels.voxel_range).astype(np.float32),
            voxels.grid_size, rgb_result
        )

        return voxel_label, rgb_result
    
    def loadSemLabel(self, index):
        img_file_path_split = self.img_file_list[index].split("/")
        seq_id = img_file_path_split[0]
        frame_id = img_file_path_split[2].strip(".png")
        pcd_label_path = os.path.join(self.data_root, "sequences", seq_id, "labels", frame_id+".label")
        label = np.fromfile(pcd_label_path, dtype=np.int32)

        sem_label = label & 0xFFFF  # semantic label in lower half
        inst_label = label >> 16  # instance id in upper half
        sem_label = self.class_map_lut[sem_label]
        return sem_label

    def getProjMatrix(self, index):
        img_file_path_split = self.img_file_list[index].split("/")
        seq_id = img_file_path_split[0]
        proj_matrix = self.proj_matrix[seq_id]
        return proj_matrix
    
        # proj_matrix = np.eye(4)
        # if self.is_train:
        #     noise_calib = np.random.normal(0, 2**(-10), size=(3, 4))
        #     proj_matrix[:3] = self.proj_matrix[seq_id] + noise_calib
        # else:
        #     proj_matrix[:3] = self.proj_matrix[seq_id]
        # return proj_matrix[:3]
    
    def mapLidar2Camera(self, index, pointcloud, img_h, img_w, cam_idx=0):
        # img_file_path_split = self.img_file_list[index].split("/")
        # seq_id = img_file_path_split[0]
        # proj_matrix = self.proj_matrix[seq_id]
        proj_matrix = self.getProjMatrix(index)
        pointcloud_hcoord = np.concatenate([pointcloud, np.ones(
            [pointcloud.shape[0], 1], dtype=np.float32)], axis=1)
        mapped_points = (proj_matrix @ pointcloud_hcoord.T).T  # n, 3
        valid_mask = mapped_points[:, 2] > 0
        # scale 2D points
        mapped_points = mapped_points[:, :2] / \
                        np.expand_dims(mapped_points[:, 2], axis=1)  # n, 2
        # fliplr so that indexing is row, col and not col, row
        mapped_points = np.fliplr(mapped_points)
        keep_idx_pts = (mapped_points[:, 1] > 0) * (mapped_points[:, 1] < img_w) * (
                mapped_points[:, 0] > 0) * (mapped_points[:, 0] < img_h) * valid_mask
        return mapped_points[keep_idx_pts], keep_idx_pts
    
    @staticmethod
    def read_calib(calib_path):
        """
        :param calib_path: Path to a calibration text file.
        :return: dict with calibration matrices.
        """
        calib_all = {}
        with open(calib_path, 'r') as f:
            for line in f.readlines():
                if line == '\n':
                    break
                key, value = line.split(':', 1)
                calib_all[key] = np.array([float(x) for x in value.split()])
        # reshape matrices
        calib_out = {}
        # 3x4 projection matrix for left camera
        calib_out['P2'] = calib_all['P2'].reshape(3, 4)
        calib_out['Tr'] = np.identity(4)  # 4x4 matrix
        calib_out['Tr'][:3, :4] = calib_all['Tr'].reshape(3, 4)
        return calib_out
    
    def __len__(self):
        return len(self.img_file_list)