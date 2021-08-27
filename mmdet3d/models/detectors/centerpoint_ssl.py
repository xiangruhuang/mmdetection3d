import torch

from mmdet3d.core import bbox3d2result, merge_aug_bboxes_3d
from mmdet.models import DETECTORS
from .mvx_two_stage import MVXTwoStageDetector
import polyscope as ps
from mmcv.cnn import ConvModule
from torch_geometric.nn import radius
from torch_scatter import scatter

@DETECTORS.register_module()
class CenterPointSSL(MVXTwoStageDetector):
    """Base class of Multi-modality VoxelNet."""

    def __init__(self,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 ssl_mlps=None):
        super(CenterPointSSL,
              self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)

        if ssl_mlps is not None:
            ssl_channels = ssl_mlps['ssl_channels']
            if ssl_mlps.get('norm_cfg', None) is None:
                norm_cfg = dict(type='BN2d')
            else:
                norm_cfg = ssl_mlps['norm_cfg']
            self.ssl_weights = []
            self.ssl_mlps = []
            for s, (ssl_channel, ssl_weight) in enumerate(ssl_channels):
                mlp = torch.nn.Sequential()
                for i in range(len(ssl_channel) - 2):
                    mlp.add_module(
                        f'sslmlp{i}',
                        ConvModule(
                            ssl_channel[i],
                            ssl_channel[i+1],
                            kernel_size=(1, 1),
                            stride=(1, 1),
                            conv_cfg=dict(type='Conv2d'),
                            norm_cfg=norm_cfg,
                            bias=True,
                            ))
                i = len(ssl_channel) - 2
                mlp.add_module(
                        f'sslmlp{i}',
                        ConvModule(
                            ssl_channel[i],
                            ssl_channel[i + 1],
                            kernel_size=(1, 1),
                            stride=(1, 1),
                            conv_cfg=dict(type='Conv2d'),
                            norm_cfg=None,
                            act_cfg=None,
                            bias=True))
                self.ssl_mlps.append(mlp)
                self.ssl_weights.append(ssl_weight)
            self.ssl_mlps = torch.nn.ModuleList(self.ssl_mlps)
            self.ce_loss = torch.nn.CrossEntropyLoss(
                weight=torch.as_tensor([1e-2, 1.0]))

    def extract_pts_feat(self, pts, img_feats, img_metas):
        """Extract features of points."""
        if not self.with_pts_bbox:
            return None
        voxels, num_points, coors = self.voxelize(pts)

        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1
        x, ef = self.pts_middle_encoder(voxel_features, coors, batch_size,
                    return_encode_features=True)
        
        voxel_points = voxels.view(-1, voxels.shape[-1])
        x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        return x, ef

    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None):
        """Forward function for point cloud branch.

        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        outs = self.pts_bbox_head(pts_feats)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs)
        return losses
    
    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      motion_mask_3d=None,
                      use_obj_labels=None):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        #ps.set_up_dir('z_up')
        #ps.init()
        #for i, pi in enumerate(points):
        #    ps.register_point_cloud(f'pts-{i}',
        #        pi[:, :3].detach().cpu(), radius=2e-4)
        #import ipdb; ipdb.set_trace()
        #size_x, size_y = self.pts_voxel_layer.voxel_size[:2]
        #lx, ly, lz = self.pts_voxel_layer.point_cloud_range[:3]
        #for task_id, heatmap in enumerate(heatmaps):
        #    for scene_id, scene_heatmap in enumerate(heatmap):
        #        # scalar, [1, 128, 128]
        #        y, x = torch.where(scene_heatmap[0] > 0) # tuple
        #        x = lx + x*8*size_x
        #        y = ly + y*8*size_y
        #        z = torch.ones_like(x)
        #        coors = torch.stack([x,y,z], dim=-1)
        #        ps.register_point_cloud(
        #            f'heatmap-task{task_id}-scene{scene_id}',
        #            coors.detach().cpu(), radius=1e-3, enabled=False)
        #box_connection = [[0, 1], [0, 3], [0, 4], [1, 2], [1, 5], [2, 3],
        #                  [2, 6], [3, 7], [4, 5], [4, 7], [5, 6], [6, 7]]
        #box_connection = torch.as_tensor(box_connection, dtype=torch.long)
        #box_connection = box_connection.view(-1, 2)
        #for scene_id, (bboxes, labels) in enumerate(zip(gt_bboxes_3d, gt_labels_3d)):
        #    corners = bboxes.corners
        #    edges = [box_connection + 8*i for i in range(corners.shape[0])]
        #    edges = torch.cat(edges, dim=0)
        #    colors = torch.randn(3, 3)
        #    edge_colors = colors[labels].repeat(12, 1, 1).transpose(0, 1).reshape(-1, 3)
        #    box_net = ps.register_curve_network(f'boxes-scene{scene_id}',
        #        corners.detach().cpu().view(-1, 3),
        #        edges.detach().cpu().view(-1, 2), radius=4e-4)
        #    box_net.add_color_quantity('edges', edge_colors, defined_on='edges', enabled=True)
        #ps.show()
            
        losses = dict()
        if (use_obj_labels is None):
            img_feats, (pts_feats, middle_feats) = self.extract_feat(
                points, img=img, img_metas=img_metas)
            if pts_feats:
                losses_pts = self.forward_pts_train(pts_feats, gt_bboxes_3d,
                                                    gt_labels_3d, img_metas,
                                                    gt_bboxes_ignore)
                losses.update(losses_pts)
            if img_feats:
                losses_img = self.forward_img_train(
                    img_feats,
                    img_metas=img_metas,
                    gt_bboxes=gt_bboxes,
                    gt_labels=gt_labels,
                    gt_bboxes_ignore=gt_bboxes_ignore,
                    proposals=proposals)
                losses.update(losses_img)
        elif use_obj_labels.float().sum() > 0:
            points_s = [p for p, u in zip(points, use_obj_labels) if u]
            gt_bboxes_3d = [p for p, u in zip(gt_bboxes_3d, use_obj_labels) if u]
            gt_labels_3d = [p for p, u in zip(gt_labels_3d, use_obj_labels) if u]
            if img is not None:
                img = [im for im, u in zip(img, use_obj_labels) if u]
            if img_metas is not None:
                img_metas = [im for im, u in zip(img_metas, use_obj_labels) if u]
            if gt_bboxes_ignore is not None:
                gt_bboxes_ignore = [im for im, u in zip(gt_bboxes_ignore, use_obj_labels) if u]
            img_feats, (pts_feats, middle_feats) = self.extract_feat(
                points_s, img=img, img_metas=img_metas)
            if pts_feats:
                losses_pts = self.forward_pts_train(pts_feats, gt_bboxes_3d,
                                                    gt_labels_3d, img_metas,
                                                    gt_bboxes_ignore)
                losses.update(losses_pts)
        else:
            losses['loss_fake'] = torch.nn.Parameter(torch.zeros(1), requires_grad=True).to(points[0].device)
        if motion_mask_3d is not None:
            points_m = [p for p, m in zip(points, motion_mask_3d) if m.sum()>0]
            masks_m = [m for m in motion_mask_3d if m.sum()>0]
            if len(points_m) > 0:
                img_feats, (pts_feats, middle_feats) = self.extract_feat(
                    points_m, img=img, img_metas=img_metas)

                out_factors = [1, 1, 2, 4, 8, 8]
                voxel_size_x, voxel_size_y, voxel_size_z = self.pts_voxel_layer.voxel_size[:3]
                min_x, min_y, min_z = self.pts_voxel_layer.point_cloud_range[:3]
                moving_points = []
                for i, (points, mask) in enumerate(zip(points_m, masks_m)):
                    mp = points[torch.where(mask > 0.5)[0]][:, :3]
                    #x = ( x - min_x ) / out_factors[i] / voxel_size_x
                    #y = ( y - min_y ) / out_factors[i] / voxel_size_y
                    #z = ( z - min_z ) / out_factors[i] / voxel_size_z
                    b = torch.zeros_like(mp[:, 0]) + i*10
                    mp = torch.cat([b.unsqueeze(-1), mp], dim=-1)
                    moving_points.append(mp)
                moving_points = torch.cat(moving_points, dim=0)
                    
                #ps.init()
                #ps.set_up_dir('z_up')
                #for i, pi in enumerate(points_m):
                #    ps.register_point_cloud(f'p-{i}', pi.detach().cpu()[:, :3], radius=2e-4)
                for i, (middle_feat, out_f) in enumerate(zip(middle_feats, out_factors)):
                    if self.ssl_weights[i] == 0:
                        continue
                    indices = middle_feat.indices
                    b, z, x, y = indices.T
                    x = x * voxel_size_x * out_f + min_x
                    y = y * voxel_size_y * out_f + min_y
                    z = z * voxel_size_z * out_f + min_z
                    vox_pos = torch.stack([b*10, y, x, z], dim=-1)
                    e0, e1 = radius(moving_points, vox_pos, (0.15**2+0.2**2+0.15**2)**0.5/2.0)
                    degree = scatter(torch.ones_like(e0), index=e0, dim=0, dim_size=vox_pos.shape[0])
                    moving_index = torch.where(degree >= 3)[0]
                    out = self.ssl_mlps[i](middle_feat.features.unsqueeze(-1).unsqueeze(-1))[:, :, 0, 0]
                    gt_labels = torch.zeros(out.shape[0], dtype=torch.long).to(out.device)
                    gt_labels[moving_index] = 1
                    losses[f'loss_ssl{i}'] = self.ce_loss(out, gt_labels) * self.ssl_weights[i]
                    #vox_pos = vox_pos[moving_index]
                    #for j in range(len(points_m)):
                    #    vox_pos_j = vox_pos[(vox_pos[:, 0] == j)][:, 1:]
                    #    ps.register_point_cloud(f'moving voxels-{j}', vox_pos_j.detach().cpu(), radius=5e-4)
                    #ps.show()

        return losses


    def simple_test_pts(self, x, img_metas, rescale=False):
        """Test function of point cloud branch."""
        outs = self.pts_bbox_head(x)
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def aug_test_pts(self, feats, img_metas, rescale=False):
        """Test function of point cloud branch with augmentaiton.

        The function implementation process is as follows:

            - step 1: map features back for double-flip augmentation.
            - step 2: merge all features and generate boxes.
            - step 3: map boxes back for scale augmentation.
            - step 4: merge results.

        Args:
            feats (list[torch.Tensor]): Feature of point cloud.
            img_metas (list[dict]): Meta information of samples.
            rescale (bool): Whether to rescale bboxes. Default: False.

        Returns:
            dict: Returned bboxes consists of the following keys:

                - boxes_3d (:obj:`LiDARInstance3DBoxes`): Predicted bboxes.
                - scores_3d (torch.Tensor): Scores of predicted boxes.
                - labels_3d (torch.Tensor): Labels of predicted boxes.
        """
        # only support aug_test for one sample
        outs_list = []
        for x, img_meta in zip(feats, img_metas):
            outs = self.pts_bbox_head(x)
            # merge augmented outputs before decoding bboxes
            for task_id, out in enumerate(outs):
                for key in out[0].keys():
                    if img_meta[0]['pcd_horizontal_flip']:
                        outs[task_id][0][key] = torch.flip(
                            outs[task_id][0][key], dims=[2])
                        if key == 'reg':
                            outs[task_id][0][key][:, 1, ...] = 1 - outs[
                                task_id][0][key][:, 1, ...]
                        elif key == 'rot':
                            outs[task_id][0][
                                key][:, 1,
                                     ...] = -outs[task_id][0][key][:, 1, ...]
                        elif key == 'vel':
                            outs[task_id][0][
                                key][:, 1,
                                     ...] = -outs[task_id][0][key][:, 1, ...]
                    if img_meta[0]['pcd_vertical_flip']:
                        outs[task_id][0][key] = torch.flip(
                            outs[task_id][0][key], dims=[3])
                        if key == 'reg':
                            outs[task_id][0][key][:, 0, ...] = 1 - outs[
                                task_id][0][key][:, 0, ...]
                        elif key == 'rot':
                            outs[task_id][0][
                                key][:, 0,
                                     ...] = -outs[task_id][0][key][:, 0, ...]
                        elif key == 'vel':
                            outs[task_id][0][
                                key][:, 0,
                                     ...] = -outs[task_id][0][key][:, 0, ...]

            outs_list.append(outs)

        preds_dicts = dict()
        scale_img_metas = []

        # concat outputs sharing the same pcd_scale_factor
        for i, (img_meta, outs) in enumerate(zip(img_metas, outs_list)):
            pcd_scale_factor = img_meta[0]['pcd_scale_factor']
            if pcd_scale_factor not in preds_dicts.keys():
                preds_dicts[pcd_scale_factor] = outs
                scale_img_metas.append(img_meta)
            else:
                for task_id, out in enumerate(outs):
                    for key in out[0].keys():
                        preds_dicts[pcd_scale_factor][task_id][0][key] += out[
                            0][key]

        aug_bboxes = []

        for pcd_scale_factor, preds_dict in preds_dicts.items():
            for task_id, pred_dict in enumerate(preds_dict):
                # merge outputs with different flips before decoding bboxes
                for key in pred_dict[0].keys():
                    preds_dict[task_id][0][key] /= len(outs_list) / len(
                        preds_dicts.keys())
            bbox_list = self.pts_bbox_head.get_bboxes(
                preds_dict, img_metas[0], rescale=rescale)
            bbox_list = [
                dict(boxes_3d=bboxes, scores_3d=scores, labels_3d=labels)
                for bboxes, scores, labels in bbox_list
            ]
            aug_bboxes.append(bbox_list[0])

        if len(preds_dicts.keys()) > 1:
            # merge outputs with different scales after decoding bboxes
            merged_bboxes = merge_aug_bboxes_3d(aug_bboxes, scale_img_metas,
                                                self.pts_bbox_head.test_cfg)
            return merged_bboxes
        else:
            for key in bbox_list[0].keys():
                bbox_list[0][key] = bbox_list[0][key].to('cpu')
            return bbox_list[0]

    def aug_test(self, points, img_metas, imgs=None, rescale=False):
        """Test function with augmentaiton."""
        img_feats, pts_feats = self.extract_feats(points, img_metas, imgs)
        bbox_list = dict()
        if pts_feats and self.with_pts_bbox:
            pts_bbox = self.aug_test_pts(pts_feats, img_metas, rescale)
            bbox_list.update(pts_bbox=pts_bbox)
        return [bbox_list]
