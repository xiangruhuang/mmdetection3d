import mmcv
import numpy as np
import torch
from mmcv.parallel import DataContainer as DC
from mmcv.runner import auto_fp16
from os import path as osp

from mmdet3d.core import show_seg_result
from mmseg.models import SEGMENTORS
from ..builder import build_backbone, build_head, build_neck
from .base import Base3DSegmentor
from mmdet3d.ops import knn

@SEGMENTORS.register_module()
class DGCNN(Base3DSegmentor):
    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(PointNet2Seg, self).__init__()
        self.backbone = build_backbone(backbone)
        self.k = k
        self.decode_head = build_head(decode_head)

    def backbone(self, points, features):
        """
        Args:
            features (torch.Tensor, shape=[1, N, D]): feature vectors.

        """
        adj = knn(self.k, points, points)[0] # [1, k, N]
        for i in range(self.k):
            diff = torch.cat(
                       [features[0], 
                        features[0] - features[0, :, adj[i]]],
                       dim=0)


    def forward_train(self, points, img_metas, gt_labels):
        res = []
        for p in points:
            out_dict = self.backbone(p.unsqueeze(0))
            res.append(self.decode_head(out_dict)) # [1, 2, N]
        y = torch.cat(gt_labels, dim=0) # [N]
        res = torch.cat(res, dim=-1).squeeze(0).T # [N, 2]
        pred = res.argmax(-1)
        acc = (pred == y).float()

        losses = self.decode_head.losses(res, y)
        losses.update(dict(acc=acc))

        return losses

    def simple_test(self, points, img_metas, gt_labels=None, rescale=True):
        """Simple test with single scene.

        Args:
            points (list[torch.Tensor]): List of points of shape [N, 3+C].
            img_metas (list[dict]): Meta information of each sample.
            rescale (bool): Whether transform to original number of points.
                Will be used for voxelization based segmentors.
                Defaults to True.

        Returns:
            list[dict]: The output prediction result with following keys:

                - semantic_mask (Tensor): Segmentation mask of shape [N].
        """
        if not self.training:
            self.train()
        # 3D segmentation requires per-point prediction, so it's impossible
        # to use down-sampling to get a batch of scenes with same num_points
        # therefore, we only support testing one scene every time
        points = [points]
        res = []
        for p in points:
            out_dict = self.backbone(p[:, :3].unsqueeze(0))
            prob=self.decode_head(out_dict) # [1,2,N]
            pred=prob[0].T.argmax(-1).detach().cpu() # [2, N]
            res.append(
                dict(prob=prob.detach().cpu(), pred=pred)
            )

        return res

    def aug_test(self, points, img_metas, gt_labels=None, rescale=True):
        """Simple test with single scene.

        Args:
            points (list[torch.Tensor]): List of points of shape [N, 3+C].
            img_metas (list[dict]): Meta information of each sample.
            rescale (bool): Whether transform to original number of points.
                Will be used for voxelization based segmentors.
                Defaults to True.

        Returns:
            list[dict]: The output prediction result with following keys:

                - semantic_mask (Tensor): Segmentation mask of shape [N].
        """
        if not self.training:
            self.train()
        # 3D segmentation requires per-point prediction, so it's impossible
        # to use down-sampling to get a batch of scenes with same num_points
        # therefore, we only support testing one scene every time
        res = []
        for p in points:
            out_dict = self.backbone(p[:, :3].unsqueeze(0))
            res.append(self.decode_head(out_dict)[0].T.argmax(-1)) # [1, 2, N]

        return res
