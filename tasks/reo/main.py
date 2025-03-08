import argparse
import datetime
from option import Option
import os
import torch
import time
# from train_seg import Trainer
from train import Trainer

import pc_processor
from pc_processor.models import REONet

class Experiment(object):
    def __init__(self, settings: Option):
        self.settings = settings
        # init gpu
        os.environ["CUDA_VISIBLE_DEVICES"] = self.settings.gpu
        pc_processor.utils.init_distributed_mode(self.settings)
        if self.settings.distributed:
            torch.distributed.barrier()

        # set random seed
        if self.settings.distributed:
            rank = self.settings.rank
        else:
            rank = 0
        torch.manual_seed(self.settings.seed+rank)
        torch.cuda.manual_seed(self.settings.seed+rank)
        if self.settings.distributed:
            torch.cuda.set_device(self.settings.gpu)
        else:
            torch.cuda.set_device(0)
        torch.backends.cudnn.benchmark = True

        # init checkpoint
        if not self.settings.distributed or (self.settings.rank == 0):
            self.settings.check_path()
            self.recorder = pc_processor.checkpoint.Recorder(
                self.settings, self.settings.save_path)
        else:
            self.recorder = None

        self.epoch_start = 0
        # init model
        if self.settings.net_type == "REONet":
            self.model = REONet(
                backbone=self.settings.backbone, 
                img_size=self.settings.img_size, 
                use_pcd=self.settings.use_pcd,
                use_image=self.settings.use_img,
                use_radar=self.settings.use_radar,
                num_classes=self.settings.num_classes,
                num_sem_classes=self.settings.num_sem_classes,
                pc_range=self.settings.voxel_range,
                grid_size=self.settings.grid_size,
                voxel_downscales=self.settings.voxel_downscales,
                num_images=self.settings.num_images,
                empty_idx=self.settings.empty_idx,
                pretrained_model_root=self.settings.pretrained_model_root,
                imgnet_pretrained=self.settings.imgnet_pretrained)
        else:
            raise NotImplementedError("invalid net_type: {}".format(self.settings.net_type))

        # init trainer
        self.trainer = Trainer(
            self.settings, self.model, self.recorder)
        
        # load checkpoint
        self._loadCheckpoint()

    def _loadCheckpoint(self):
        assert self.settings.pretrained_model is None or self.settings.checkpoint is None, "cannot use pretrained weight and checkpoint at the same time"
        if self.settings.img_pretrained_model is not None:
            if not os.path.isfile(self.settings.img_pretrained_model):
                raise FileNotFoundError("img pretrained model not found: {}".format(
                    self.settings.img_pretrained_model))
            state_dict = torch.load(
                self.settings.img_pretrained_model, map_location="cpu")
            new_state_dict = self.model.state_dict()
            for k, v in state_dict.items():
                if "aux_decoder" not in k and "feature_extractor" not in k:
                    continue

                if k in new_state_dict.keys():
                    if new_state_dict[k].size() == v.size():
                        new_state_dict[k] = v
                    else:
                        print("diff size: ", k, v.size())
                else:
                    print("diff key: ", k)
            self.model.load_state_dict(new_state_dict)
            # self.model.load_state_dict(state_dict)
            if self.recorder is not None:
                self.recorder.logger.info(
                    "loading img pretrained weight from: {}".format(self.settings.pretrained_model))
                
        if self.settings.pretrained_model is not None:
            if not os.path.isfile(self.settings.pretrained_model):
                raise FileNotFoundError("pretrained model not found: {}".format(
                    self.settings.pretrained_model))
            state_dict = torch.load(
                self.settings.pretrained_model, map_location="cpu")
            new_state_dict = self.model.state_dict()
            for k, v in state_dict.items():
                if k in new_state_dict.keys():
                    if new_state_dict[k].size() == v.size():
                        new_state_dict[k] = v
                    else:
                        print("diff size: ", k, v.size())
                else:
                    print("diff key: ", k)
            self.model.load_state_dict(new_state_dict)
            # self.model.load_state_dict(state_dict)
            if self.recorder is not None:
                self.recorder.logger.info(
                    "loading pretrained weight from: {}".format(self.settings.pretrained_model))

        if self.settings.checkpoint is not None:
            if not os.path.isfile(self.settings.checkpoint):
                raise FileNotFoundError(
                    "checkpoint file not found: {}".format(self.settings.checkpoint))
            checkpoint_data = torch.load(
                self.settings.checkpoint, map_location="cpu")
            self.model.load_state_dict(checkpoint_data["model"])
            self.trainer.optimizer.load_state_dict(
                checkpoint_data["optimizer"])
            self.epoch_start = checkpoint_data["epoch"] + 1

    def run(self):
        t_start = time.time()
        if self.settings.val_only:
            self.trainer.run(0, mode="Validation")
            return
        
        best_val_result = None
        # # update lr after backward (required by pytorch)
        for epoch in range(self.epoch_start, self.settings.n_epochs):
            self.trainer.run(epoch, mode="Train")
            if epoch % self.settings.val_frequency == 0 or epoch == self.settings.n_epochs-1:
                val_result = self.trainer.run(epoch, mode="Validation")
                if self.recorder is not None:
                    if best_val_result is None:
                        best_val_result = val_result
                    for k, v in val_result.items():
                        if v >= best_val_result[k]:
                            self.recorder.logger.info(
                                "get better {} model: {}".format(k, v))
                            saved_path = os.path.join(
                                self.recorder.checkpoint_path, "best_{}_model.pth".format(k))
                            best_val_result[k] = v
                            torch.save(self.model.state_dict(), saved_path)

            # save checkpoint
            if self.recorder is not None:
                saved_path = os.path.join(
                    self.recorder.checkpoint_path, "checkpoint.pth")
                checkpoint_data = {
                    "model": self.model.state_dict(),
                    "optimizer": self.trainer.optimizer.state_dict(),
                    "epoch": epoch,
                }

                torch.save(checkpoint_data, saved_path)
                # log
                if best_val_result is not None:
                    log_str = ">>> Best Result: "
                    for k, v in best_val_result.items():
                        log_str += "{}: {} ".format(k, v)
                    self.recorder.logger.info(log_str)
        cost_time = time.time() - t_start
        if self.recorder is not None:
            self.recorder.logger.info("==== total cost time: {}".format(
                datetime.timedelta(seconds=cost_time)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment Options")
    parser.add_argument("config_path", type=str, metavar="config_path",
                        help="path of config file, type: string")
    parser.add_argument("--id", type=int, metavar="experiment_id", required=False,
                        help="id of experiment", default=0)
    args = parser.parse_args()
    exp = Experiment(Option(args.config_path))
    print("===init env success===")
    exp.run()
