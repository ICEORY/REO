import numpy as np
import os
from tqdm import tqdm
import argparse
import logging


class Metric_mIoU_online():
    def __init__(self, num_classes=18):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))
        self.cnt = 0
    
    def hist_info(self, n_cl, pred, gt):
        """
        build confusion matrix
        empty classes:0
        non-empty class: 1-16
        free voxel class: 17
        
        Args:
            n_cl (int): num_classes_occupancy
            pred (1-d array): pred_occupancy_label
            gt (1-d array): gt_occupancu_label
            
        Returns:
            tuple:(hist, correctly number_predicted_labels, num_labelled_sample) 
        """
        assert pred.shape == gt.shape
        k = (gt >= 0) & (gt < n_cl)  # exclude 255
        labeled = np.sum(k)
        correct = np.sum((pred[k] == gt[k]))

        return (
            np.bincount(
                n_cl * gt[k].astype(int) + pred[k].astype(int), minlength=n_cl ** 2
            ).reshape(n_cl, n_cl),
            correct,
            labeled,
        )

    def per_class_iu(self, hist):
        return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

    def compute_mIoU(self, pred, label, n_classes):
        hist = np.zeros((n_classes, n_classes))
        new_hist, correct, labeled = self.hist_info(n_classes, pred.flatten(), label.flatten()) 
        hist += new_hist
        mIoUs = self.per_class_iu(hist)
        return mIoUs, hist
    
    def addBatch(self, pred, label):
        _miou, _hist = self.compute_mIoU(pred, label, self.num_classes)
        self.hist += _hist
        self.cnt += 1
        return _miou

    def getIoU(self):
        mIoU = self.per_class_iu(self.hist)
        print(f'===> per class IoU of {self.cnt} samples:')
        for ind_class in range(self.num_classes):
            print(f'===> class {ind_class} IoU = ' + str(round(mIoU[ind_class] * 100, 2)))
        print(f'===> mIoU of {self.cnt} samples (0-17): ' + str(round(np.nanmean(mIoU) * 100, 2)))
        print(f'===> mIoU of {self.cnt} samples (0-16): ' + str(round(np.nanmean(mIoU[:17]) * 100, 2)))
        return mIoU


class Metric_mIoU():
    def __init__(self,
                 gt_path,
                 pred_path,
                 save_dir='.',
                 num_classes=18,
                 use_lidar_mask=False,
                 use_image_mask=False,
                 ):

        self.gt_path = gt_path
        self.pred_path = pred_path
        self.save_dir = save_dir
        self.use_lidar_mask = use_lidar_mask
        self.use_image_mask = use_image_mask
        self.num_classes = num_classes
        # self.point_cloud_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
        # self.occupancy_size = [0.4, 0.4, 0.4]
        # self.voxel_size = 0.4
        # self.occ_xdim = int((self.point_cloud_range[3] - self.point_cloud_range[0]) / self.occupancy_size[0])
        # self.occ_ydim = int((self.point_cloud_range[4] - self.point_cloud_range[1]) / self.occupancy_size[1])
        # self.occ_zdim = int((self.point_cloud_range[5] - self.point_cloud_range[2]) / self.occupancy_size[2])
        # self.voxel_num = self.occ_xdim * self.occ_ydim * self.occ_zdim
        self.logger = self._initLogger()

    def _initLogger(self):
        logger = logging.getLogger("eval_perframe")
        logger.propagate = False
        file_formatter = logging.Formatter('%(message)s')
        file_handler = logging.FileHandler(os.path.join(self.save_dir, "eval_perframe.log"))    # file log
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)
        return logger

    def hist_info(self, n_cl, pred, gt):
        """
        build confusion matrix
        empty classes:0
        non-empty class: 1-16
        free voxel class: 17
        
        Args:
            n_cl (int): num_classes_occupancy
            pred (1-d array): pred_occupancy_label
            gt (1-d array): gt_occupancu_label
            
        Returns:
            tuple:(hist, correctly number_predicted_labels, num_labelled_sample) 
        """
        assert pred.shape == gt.shape
        k = (gt >= 0) & (gt < n_cl)  # exclude 255
        labeled = np.sum(k)
        correct = np.sum((pred[k] == gt[k]))

        return (
            np.bincount(
                n_cl * gt[k].astype(int) + pred[k].astype(int), minlength=n_cl ** 2
            ).reshape(n_cl, n_cl),
            correct,
            labeled,
        )

    def per_class_iu(self, hist):

        return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

    def compute_mIoU(self, pred, label, n_classes):
        hist = np.zeros((n_classes, n_classes))
        new_hist, correct, labeled = self.hist_info(n_classes, pred.flatten(), label.flatten()) 
        hist += new_hist
        mIoUs = self.per_class_iu(hist)
        # for ind_class in range(n_classes):
        #     print(str(round(mIoUs[ind_class] * 100, 2)))
        # print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)))
        return round(np.nanmean(mIoUs) * 100, 2), hist
    
    def get_dicts(self, ):
        gts_dict = {}
        for scene in os.listdir(self.gt_path):
            for frame in os.listdir(os.path.join(self.gt_path, scene)):
                scene_token = frame
                gts_dict[scene_token] = os.path.join(self.gt_path, scene, frame, 'labels.npz')
        print('number of gt samples = {}'.format(len(gts_dict)))

        preds_dict = {}
        for scene in os.listdir(self.pred_path):
            for frame in os.listdir(os.path.join(self.pred_path, scene)):
                scene_token = frame
                preds_dict[scene_token] = os.path.join(self.pred_path, scene, frame, 'labels.npz')
        print('number of pred samples = {}'.format(len(preds_dict)))
        return gts_dict, preds_dict
    
    def __call__(self):
        gts_dict, preds_dict = self.get_dicts()

        # assert set(gts_dict.keys()) == set(preds_dict.keys()) # DEBUG_TMP
        num_samples = len(preds_dict.keys())
        # _mIoU = 0.
        cnt = 0
        hist = np.zeros((self.num_classes, self.num_classes))
        
        for scene_token in tqdm(preds_dict.keys()):
            cnt += 1
            gt = np.load(gts_dict[scene_token])
            pred = np.load(preds_dict[scene_token])
            semantics_gt, mask_lidar, mask_camera = gt['semantics'], gt['mask_lidar'], gt['mask_camera']
            semantics_pred = pred['semantics']
            mask_lidar = mask_lidar.astype(bool)
            mask_camera = mask_camera.astype(bool)

            # # eval range: 0-10
            # mask_camera[:75, :, :] = 0
            # mask_camera[125:, :, :] = 0
            # mask_camera[:, :75, :] = 0
            # mask_camera[:, 125:, :] = 0

            # # eval range: 10-20
            # mask_camera[:50, :, :] = 0
            # mask_camera[150:, :, :] = 0
            # mask_camera[:, :50, :] = 0
            # mask_camera[:, 150:, :] = 0
            # mask_camera[75:125, 75:125, :] = 0

            # # eval range: 20-30
            # mask_camera[:25, :, :] = 0
            # mask_camera[175:, :, :] = 0
            # mask_camera[:, :25, :] = 0
            # mask_camera[:, 175:, :] = 0
            # mask_camera[50:150, 50:150, :] = 0
            
            # # eval range: 30-40
            # mask_camera[25:175, 25:175, :] = 0

            if self.use_image_mask:
                masked_semantics_gt = semantics_gt[mask_camera]
                masked_semantics_pred = semantics_pred[mask_camera]
            elif self.use_lidar_mask:
                masked_semantics_gt = semantics_gt[mask_lidar]
                masked_semantics_pred = semantics_pred[mask_lidar]
            else:
                masked_semantics_gt = semantics_gt
                masked_semantics_pred = semantics_pred

            # pred = np.random.randint(low=0, high=17, size=masked_semantics.shape)

            _, _hist = self.compute_mIoU(masked_semantics_pred, masked_semantics_gt, self.num_classes)
            hist += _hist
            # _mIoU += _miou

            # log: record the miou of each frame
            miou_cur_frame = self.per_class_iu(_hist)
            log_str = ">>> mIoU (0-16): {}, prediction_path: {} ".format(
                round(np.nanmean(miou_cur_frame[:17]) * 100, 2), preds_dict[scene_token])
            self.logger.info(log_str)

        # mIoU_avg = _mIoU / num_samples
        mIoU = self.per_class_iu(hist)
        assert cnt == num_samples, 'some samples are not included in the miou calculation'
        print(f'===> per class IoU of {cnt} samples:')
        for ind_class in range(self.num_classes):
            print(f'===> class {ind_class} IoU = ' + str(round(mIoU[ind_class] * 100, 2)))
            
        # print(f'===> mIoU of {cnt} samples: ' + str(round(np.nanmean(mIoU) * 100, 2)))
        # print(f'===> sample-wise averaged mIoU of {cnt} samples: ' + str(round(np.nanmean(mIoU_avg), 2)))
        print(f'===> mIoU of {cnt} samples (0-17): ' + str(round(np.nanmean(mIoU) * 100, 2)))
        print(f'===> mIoU of {cnt} samples (0-16): ' + str(round(np.nanmean(mIoU[:17]) * 100, 2)))
        return mIoU


def eval_nuscene(gt_path, pred_path, save_dir=None):
    miou = Metric_mIoU(
        gt_path=gt_path,
        pred_path=pred_path,
        save_dir=save_dir,
        num_classes=18,
        use_lidar_mask=False,
        use_image_mask=True,
    )
    metric = miou()


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--gt_path', help='path to gt voxels')
    # parser.add_argument('--pred_path', help='path to gt voxels')
    # args = parser.parse_args()
    
    # eval_nuscene(args.gt_path, args.pred_path)

    gt_path = "/mnt/dataset/Occ3D_nuScenes/trainval/gts/"
    pred_path = "/home/chensitao/chensitao/code/QueryBasedOccupancy/experiments/OccFormer/log_QueryOccNet-resnet50_bs4E25lr0.001_nuscenes-RGB-20231109-useimg/infer_results"
    log_save_dir = "/home/chensitao/chensitao/code/QueryBasedOccupancy/experiments/OccFormer/log_QueryOccNet-resnet50_bs4E25lr0.001_nuscenes-RGB-20231109-useimg/log"
    eval_nuscene(gt_path, pred_path, log_save_dir)
