import os 
import yaml
import sys 
import shutil
import numpy as np
sys.path.insert(0, "../../")

import pc_processor

class Option(object):
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = yaml.safe_load(open(config_path, "r"))

        # ---------------------------- general options -----------------
        self.save_path = self.config["save_path"] # log path
        self.seed = self.config["seed"] # manually set RNG seed
        self.gpu = self.config["gpu"] # GPU id to use, e.g. "0,1,2,3"
        self.rank = 0 # rank of distributed thread
        self.world_size = 1 # 
        self.distributed = False # 
        self.n_gpus = len(self.gpu.split(",")) # # number of GPUs to use by default
        self.dist_backend = "nccl"
        self.dist_url = "env://"

        self.print_frequency = self.config["print_frequency"]  # print frequency (default: 10)
        self.n_threads = self.config["n_threads"] # number of threads used for data loading
        self.experiment_id = self.config["experiment_id"] # identifier for experiment
       
        # --------------------------- data config ------------------------
        self.dataset = self.config["dataset"]
        self.data_root = self.config["data_root"]
        self.source_data_root = self.config["source_data_root"]
        self.grid_size = self.config["grid_size"]
        self.img_size = self.config["img_size"]
        self.crop_img_size = self.config["crop_img_size"]
        self.use_pcd = self.config["use_pcd"]
        self.use_img = self.config["use_img"]
        self.use_radar = self.config["use_radar"]
        self.num_pc_voxels = self.config["num_pc_voxels"]
        self.voxel_range = self.config["voxel_range"]
        self.num_images = self.config["num_images"]
        self.voxel_downscales = self.config["voxel_downscales"] 
        self.use_multi_sweeps = self.config["use_multi_sweeps"]

        # --------------- train config ------------------------
        self.n_epochs = self.config["n_epochs"]  # number of total epochs
        self.batch_size = self.config["batch_size"]  # mini-batch size
        
        self.lr = self.config["lr"] # initial learning rate
        self.warmup_epochs = self.config["warmup_epochs"]
        
        self.momentum = self.config["momentum"]
        self.weight_decay = self.config["weight_decay"]

        self.val_only = self.config["val_only"]
        self.is_debug = self.config["is_debug"]
        self.val_frequency = self.config["val_frequency"]
        self.cls_freq = self.config["cls_freq"]

        self.use_proj_loss = self.config["use_proj_loss"]
        self.w_depth = self.config["w_depth"]
        self.w_rgb = self.config["w_rgb"]
        self.w_dice = self.config["w_dice"]
        self.w_focal = self.config["w_focal"]

        # --------------------------- model options -----------------------
        self.net_type = self.config["net_type"]
        self.backbone = self.config["backbone"]
        self.num_classes = self.config["num_classes"]
        self.num_sem_classes = self.config["num_sem_classes"]
        self.empty_idx = self.config["empty_idx"]
        self.imgnet_pretrained = self.config["imgnet_pretrained"]
        self.pretrained_model_root = self.config["pretrained_model_root"]

        # --------------------------- checkpoit model ----------------------
        self.checkpoint = self.config["checkpoint"]
        self.pretrained_model = self.config["pretrained_model"]
        self.img_pretrained_model = self.config.get("img_pretrained_model", None)
        
        # --------------------------- inference ----------------------
        self.save_infer_result = self.config["save_infer_result"]
        self.infer_path = self.config["infer_path"]
        self.infer_model = self.config["infer_model"]

        self._prepare()

    def _prepare(self):
        # check settings
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            batch_size = self.batch_size[0] * self.n_gpus
        else:
            batch_size = self.batch_size[0]

        # folder name: log_dataset_nettype_batchsize-lr__experimentID
        modality = ""
        if self.use_img:
            modality += "Cam"
        if self.use_pcd:
            modality += "Lidar"
        if self.use_radar:
            modality += "Radar"
        self.save_path = os.path.join(self.save_path, "log_{}-{}-{}-{}_bs{}E{}lr{}_{}".format(
            self.net_type, self.dataset, modality, self.backbone, batch_size, 
            self.n_epochs, self.lr, self.experiment_id
        ))

    def check_path(self):
        if pc_processor.utils.is_main_process():
            if os.path.exists(self.save_path):
                print("file exist: {}".format(self.save_path))
                action = input("Select Action: d(delete) / q(quit): ").lower().strip()
                if action == "d":
                    shutil.rmtree(self.save_path)
                else:
                    raise OSError("Directory exits: {}".format(self.save_path))
            
            if not os.path.isdir(self.save_path):
                os.makedirs(self.save_path)
