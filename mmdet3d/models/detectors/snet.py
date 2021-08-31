import torch
from mmcv.runner import force_fp32
from torch.nn import functional as F

from mmdet.models import DETECTORS, build_backbone, build_head, build_neck
from .base import Base3DDetector

@DETECTORS.register_module()
class SNet(Base3DDetector):

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 seg_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,):
        super(SNet, self).__init__()
        self.backbone = build_backbone(backbone)
        if seg_head is not None:
            self.seg_head = build_head(seg_head)
        weight = torch.as_tensor([0.1, 1.0, 1.0, 4.0])
        self.ce_loss = torch.nn.CrossEntropyLoss(weight)
        #bbox_head.update(train_cfg=train_cfg)
        #bbox_head.update(test_cfg=test_cfg)
        #self.bbox_head = build_head(bbox_head)

    def extract_feat(self, points, img_metas):
        """Extract features from points."""
        
        feat_dicts = []
        for i in range(len(points)):
            feat_dict = self.backbone(points[i].unsqueeze(0))
            feat_dicts.append(feat_dict)

        #voxels, num_points, coors = self.voxelize(points)
        #voxel_features = self.voxel_encoder(voxels, num_points, coors)
        #batch_size = coors[-1, 0].item() + 1
        #x = self.middle_encoder(voxel_features, coors, batch_size)
        #x = self.backbone(x)
        #if self.with_neck:
        #    x = self.neck(x)
        return feat_dicts

    def forward_train(self,
                      points,
                      img_metas,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      pts_segment_mask=None,
                      gt_bboxes_ignore=None):
        """Training forward function.

        Args:
            points (list[torch.Tensor]): Point cloud of each sample.
            img_metas (list[dict]): Meta information of each sample
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        x = self.extract_feat(points, img_metas)
        outs = []
        for xi in x:
            outs.append(self.seg_head(xi))
        outs = torch.cat(outs, dim=-1).squeeze(0).transpose(0, 1).softmax(-1)
        pts_segment_mask = torch.cat(pts_segment_mask)
        losses = dict()
        losses['loss_seg'] = self.ce_loss(outs, pts_segment_mask.long())
        pred = outs.argmax(-1)
        names = ['BG', 'Car', 'Ped', 'Cyc']
        for i, name in enumerate(names):
            P = (pred == i)
            T = (pts_segment_mask == i)
            losses[f'{name}@P'] = P.float().sum()
            losses[f'{name}@T'] = T.float().sum()
            losses[f'{name}@TP'] = (T & P).float().sum()
        #loss_inputs = outs + (gt_bboxes_3d, gt_labels_3d, img_metas)
        #losses = self.bbox_head.loss(
        #    *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    def simple_test(self, points, img_metas, imgs=None, rescale=False):
        """Test function without augmentaiton."""
        import ipdb; ipdb.set_trace()
        x = self.extract_feat(points, img_metas)
        outs = self.bbox_head(x)
        bbox_list = self.bbox_head.get_bboxes(
            *outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def aug_test(self, points, img_metas, imgs=None, rescale=False):
        """Test function with augmentaiton."""
        import ipdb; ipdb.set_trace()
        feats = self.extract_feats(points, img_metas)

        # only support aug_test for one sample
        aug_bboxes = []
        for x, img_meta in zip(feats, img_metas):
            outs = self.bbox_head(x)
            bbox_list = self.bbox_head.get_bboxes(
                *outs, img_meta, rescale=rescale)
            bbox_list = [
                dict(boxes_3d=bboxes, scores_3d=scores, labels_3d=labels)
                for bboxes, scores, labels in bbox_list
            ]
            aug_bboxes.append(bbox_list[0])

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes = merge_aug_bboxes_3d(aug_bboxes, img_metas,
                                            self.bbox_head.test_cfg)

        return [merged_bboxes]
