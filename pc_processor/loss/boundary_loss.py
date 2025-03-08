###
## copy from : https://github.com/yiskw713/boundary_loss_for_remote_sensing/blob/master/boundary_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

import time
def one_hot(label, n_classes, requires_grad=False):
    """Return One Hot Label"""
    one_hot_label = torch.eye(
        n_classes, device=label.device, requires_grad=requires_grad)[label]
    one_hot_label = one_hot_label.transpose(1, 3).transpose(2, 3)

    return one_hot_label


class BoundaryLoss(nn.Module):
    """Boundary Loss proposed in:
    Alexey Bokhovkin et al., Boundary Loss for Remote Sensing Imagery Semantic Segmentation
    https://arxiv.org/abs/1905.07852
    """

    def __init__(self, theta0=3, theta=5):
        super().__init__()

        self.theta0 = theta0
        self.theta = theta

    def forward(self, pred, gt, mask=None):
        """
        Input:
            - pred: the output from model (before softmax)
                    shape (N, C, H, W)
            - gt: ground truth map
                    shape (N, H, w)
        Return:
            - boundary loss, averaged over mini-bathc
        """
        
        n, c, _, _ = pred.shape

        # softmax so that predicted map can be distributed in [0, 1]
        pred = torch.softmax(pred, dim=1)

        # one-hot vector of ground truth
        one_hot_gt = one_hot(gt, c)

        # boundary map
        gt_b = F.max_pool2d(
            1 - one_hot_gt, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        gt_b -= 1 - one_hot_gt

        pred_b = F.max_pool2d(
            1 - pred, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        pred_b -= 1 - pred

        # extended boundary map
        gt_b_ext = F.max_pool2d(
            gt_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)

        pred_b_ext = F.max_pool2d(
            pred_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)

        # reshape
        gt_b = gt_b.view(n, c, -1)
        pred_b = pred_b.view(n, c, -1)
        gt_b_ext = gt_b_ext.view(n, c, -1)
        pred_b_ext = pred_b_ext.view(n, c, -1)

        # Precision, Recall
        if mask is not None:
            mask = mask.view(n, 1, -1)
        else:
            mask = 1
        P = torch.sum(pred_b * gt_b_ext * mask, dim=2) / (torch.sum(pred_b * mask, dim=2) + 1e-7)
        R = torch.sum(pred_b_ext * gt_b * mask, dim=2) / (torch.sum(gt_b * mask, dim=2) + 1e-7)

        # Boundary F1 Score
        BF1 = 2 * P * R / (P + R + 1e-7)

        # summing BF1 Score for each class and average over mini-batch
        # loss = torch.mean(1 - BF1)
        # ignore zero idx
        loss = torch.mean(1 - BF1[:, 1:])
        return loss


# for debug
# if __name__ == "__main__":
#     import torch.optim as optim
#     from torchvision.models import segmentation

#     device = 'cuda' if torch.cuda.is_available() else 'cpu'

#     img = torch.randn(8, 3, 224, 224).to(device)
#     gt = torch.randint(0, 10, (8, 224, 224)).to(device)

#     model = segmentation.fcn_resnet50(num_classes=10).to(device)

#     optimizer = optim.Adam(model.parameters(), lr=0.0001)
#     criterion = BoundaryLoss()

#     y = model(img)

#     loss = criterion(y['out'], gt)

#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

#     print(loss)