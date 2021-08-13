import numpy as np
from mmcv import is_tuple_of
from mmcv.utils import build_from_cfg
import torch

from mmdet3d.core import VoxelGenerator
from mmdet3d.core.bbox import box_np_ops
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import RandomFlip
from ..builder import OBJECTSAMPLERS
from .data_augment_utils import noise_per_object_v3_
from mmdet3d.ops import Points_Sampler
from torch_geometric.nn import knn, fps, radius, knn_interpolate
from mmdet3d.ops import points_in_boxes_cpu
import polyscope as ps 
import open3d as o3d
import os
#from pytorch3d.transforms.rotation_conversions import \
#    matrix_to_quaternion, quaternion_to_matrix, \
#    quaternion_to_axis_angle, axis_angle_to_quaternion
from geop import icp_reweighted, batched_icp
from geop import matrix_to_axis_angle, axis_angle_to_matrix
import geop.geometry.util as gutil
from torch_scatter import scatter
from torch_geometric.data import Data
from torch_geometric.transforms import GridSampling
import time

@PIPELINES.register_module()
class EstimateMotionMask(object):
    def __init__(self,
                 sweeps_num=10,
                 points_loader=None,
                 points_range_filter=None,
                 visualize=False,
                 eval_stats=True):
        if points_loader is not None:
            self.points_loader = build_from_cfg(points_loader, PIPELINES)
        if points_range_filter is not None:
            self.points_range_filter = build_from_cfg(points_range_filter, PIPELINES)
        self.counter = 0
        self.stats = [0, 0]
        self.visualize = visualize
        self.eval_stats = eval_stats
        self.sweeps_num = sweeps_num

    def _load_points(self, pts_filename):
        inp = dict(pts_filename=pts_filename)
        return self.points_loader(inp)['points']

    def _range_filter(self, points):
        inp = dict(points=points)
        return self.points_range_filter(inp)['points']

    def graph_cut_overseg(self, points, l, r, num_clusters):
        clusters = GridSampling(0.2)(Data(pos=points)).pos
        #ps_clusters = ps.register_point_cloud('clusters', clusters,
        #                                      radius=4e-4, enabled=False)
        e0_r, e1_r = radius(clusters, clusters, 0.5, max_num_neighbors=1280)
        from scipy.sparse import csr_matrix
        import scipy
        G = csr_matrix((np.ones_like(e0_r.numpy()), (e0_r.numpy(), e1_r.numpy())),
                        shape=(clusters.shape[0], clusters.shape[0]))
        num_comp, comp_labels = scipy.sparse.csgraph.connected_components(G, directed=False)
        e0, e1 = knn(clusters, points, 1)
        return comp_labels[e1]

    def weighted_scalar_aggr(self, x, w, index):
        """
        Args:
            y (torch.Tensor, shape=[M, D])
            x (torch.Tensor, shape=[N]): src
            w (torch.Tensor, shape=[N]): weights
            index (torch.Tensor, shape=[N]): target index

        Returns:
            y (torch.Tensor, shape=[M])
        """
        num_cluster = index.max()+1
        top = scatter(x*w, index, dim=-1, dim_size=num_cluster, reduce='sum')
        bottom = scatter(w, index, dim=-1, dim_size=num_cluster, reduce='sum')
        return top / bottom

    def weighted_vector_aggr(self, x, w, index):
        """
        Args:
            x (torch.Tensor, shape=[N, D]): src
            w (torch.Tensor, shape=[N]): weights
            index (torch.Tensor, shape=[N]): target index

        Returns:
            y (torch.Tensor, shape=[M, D])
        """
        num_cluster = index.max()+1
        top = scatter(x*w.unsqueeze(-1), index, dim=0,
                      dim_size=num_cluster, reduce='sum')
        bottom = scatter(w, index, dim=-1, dim_size=num_cluster,
                         reduce='sum')
        return top / bottom.unsqueeze(-1)

    def estimate_per_cluster(self, points, ref_points,
                             point2cluster, transform):
        point2cluster = torch.as_tensor(point2cluster, dtype=torch.long)
        R, t = gutil.unpack(transform)
        R, t = R[point2cluster], t[point2cluster]
        new_pos = R.matmul(points.unsqueeze(-1)).squeeze(-1) + t

        e0_new, e1_new = knn(ref_points, new_pos, 1)
        residual = (new_pos - ref_points[e1_new]).norm(p=2, dim=-1)
        weight = (-(residual / 0.05)**2).exp()
        motion = (new_pos - points) # [N, 3]

        motion = self.weighted_vector_aggr(motion, weight, point2cluster)
        residual = self.weighted_scalar_aggr(residual, torch.ones_like(weight),
                                             point2cluster)

        return motion, residual, new_pos

    def recursive_segment(self, points, ref_points_list, segments):
        t0 = time.time()
        e0, e1 = radius(points, points, r=0.4, max_num_neighbors=64)
        diff = (points[e0] - points[e1])
        ppT = (diff.unsqueeze(-1) * diff.unsqueeze(-2)).view(-1, 9)
        ppT_per_point = scatter(
            ppT, e0, dim=0,
            dim_size=points.shape[0], reduce='sum').view(-1, 3, 3)
        eigvals = np.linalg.eigvalsh(ppT_per_point.numpy())
        valid_idx = np.where(eigvals[:, 1] > 0.03)[0]
        valid_idx = torch.as_tensor(valid_idx, dtype=torch.long)
        ref_list, normals_list = [], []
        if self.visualize:
            ps.set_up_dir('z_up')
            ps.init()
            ps.remove_all_structures()
            ps_points = ps.register_point_cloud('points', points[valid_idx], radius=2e-4)
            #for i in range(3):
            #    ps_points.add_scalar_quantity(f'eig-{i}', eigvals[:, i])
            #for ths in np.logspace(-6, 0, 20):
            #    ps_points.add_scalar_quantity(f'eig-1 < {ths}', eigvals[:, 1] < ths)

            for key in segments.keys():
                ps.register_point_cloud(f'seg-{key}', points[segments[key]],
                                        radius=4e-4, enabled=False)
        if self.eval_stats:
            obj_mask = (points[:, 0] < -1e10)
            obj_mask[:] = False
            for key in segments.keys():
                obj_mask[segments[key]] = True
            obj_mask = obj_mask[valid_idx]

        points = points[valid_idx]
        for i, ref_points in enumerate(ref_points_list):
            ref = o3d.geometry.PointCloud()
            ref.points = o3d.utility.Vector3dVector(ref_points)
            ref.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.4,
                max_nn=20))
            if self.visualize:
                ps_ref = ps.register_point_cloud(f'ref-{i}', ref_points,
                                                 radius=2e-4, enabled=False)
                ps_ref.add_vector_quantity('normals', np.array(ref.normals))
            normals_list.append(np.array(ref.normals))
            ref_list.append(ref)

        #print(f'est normal={time.time()-t0}'); t0 = time.time()
        point2cluster = self.graph_cut_overseg(points, 0, 300, 300)
        num_cluster = point2cluster.max()+1
        num_points_per_cluster = scatter(
            torch.ones_like(points[:, 0]),
            torch.as_tensor(point2cluster, dtype=torch.long), dim=0,
            dim_size=num_cluster, reduce='sum')
        max_z = scatter(
            points[:, 2],
            torch.as_tensor(point2cluster, dtype=torch.long), dim=0,
            dim_size=num_cluster, reduce='max')
        min_z = -scatter(
            -points[:, 2],
            torch.as_tensor(point2cluster, dtype=torch.long), dim=0,
            dim_size=num_cluster, reduce='max')

        if self.visualize:
            comp_colors = torch.randn(max(1000, num_cluster), 3).cuda()
            comp_idx = fps(comp_colors, ratio=(num_cluster+1) / 1000.0)
            comp_colors = comp_colors[comp_idx].detach().cpu()
            ps_points.add_color_quantity('connected components',
                                         comp_colors[point2cluster])

        #print(f'graph cut={time.time()-t0}'); t0 = time.time()

        motion_list = []
        residual_list = []
        transform = torch.eye(4).repeat(num_cluster, 1, 1).to(points)
        for i, ref in enumerate(ref_list):
            #print(f'working on component-{i}')
            ref_points = ref_points_list[i]
            e0_knn, e1_knn = knn(ref_points, points, 1)
            transf_this = torch.zeros(num_cluster, 6)
            transform = batched_icp(
                points, ref_points, np.array(ref.normals), point2cluster,
                init_transf = transform)
            motion, residual, moved_points = self.estimate_per_cluster(
                points, ref_points, point2cluster, transform)
            ps.register_point_cloud(f'moved-points-{i}', moved_points, radius=2e-4, enabled=False)
            #print(f'icp={time.time()-t0}'); t0 = time.time()
            motion = motion / (i+1.0)
            motion_list.append(motion)
            residual_list.append(residual)
            if self.visualize:
                motion_p = motion[point2cluster]
                residual_p = residual[point2cluster]
                velocity = motion.norm(p=2, dim=-1)
                velocity_p = velocity[point2cluster]
                ps_points.add_vector_quantity(f'motion-per-cluster-{i}', motion_p)
                ps_points.add_scalar_quantity(f'residual-per-cluster-{i}', residual_p)
                ps_points.add_scalar_quantity(f'velocity-per-cluster-{i}', velocity_p)
        
        motion_stack = torch.stack(motion_list, dim=-1) # [M, 3, F]
        residual_stack = torch.stack(residual_list, dim=-1) # [M, F]
        weight_stack = (-(residual_stack/0.05)**2).exp() # [M, F]

        if self.visualize:
            for i in range(weight_stack.shape[-1]):
                ps_points.add_scalar_quantity(f'weight of ref-frame-{i}',
                                              weight_stack[point2cluster, i])
        
        weight_stack, selected_frames = weight_stack.sort(descending=True, dim=-1)
        weight_stack = weight_stack[:, :3] # [M, 3]
        selected_frames = selected_frames[:, :3] # [M, 3]
        slices = []
        for k in range(3):
            slices.append(
                motion_stack.transpose(1, 2)[(torch.arange(motion_stack.shape[0]), selected_frames[:, k])]
                ) # [M, 3]
        motion_stack = torch.stack(slices, dim=-1) # [M, 3, 3]
        
        mean_motion = (motion_stack * weight_stack.unsqueeze(-2)).sum(-1) / weight_stack.sum(-1).unsqueeze(-1) # [M, 3]
        motion_std = ((motion_stack - mean_motion.unsqueeze(-1)).square()).sum(-1)
        motion_std = (motion_std.sum(-1)/3).sqrt()
        mean_velocity = mean_motion.norm(p=2, dim=-1)
        motion_mask = (mean_velocity > 0.15) & (motion_std < 0.05) & (num_points_per_cluster > 10)
        motion_mask = motion_mask & ((max_z - min_z) > 0.2)
        motion_mask_p = motion_mask[point2cluster]

        if self.visualize:
            mean_motion_p = mean_motion[point2cluster]
            motion_std_p = motion_std[point2cluster]
            mean_velocity_p = mean_velocity[point2cluster]
            #for ths in [0.05, 0.1, 0.15, 0.2]:
            #    ps_points.add_scalar_quantity(f'std(motion) per cluster > {ths}',
            #                                  (motion_std_p > ths).float())
            ps_points.add_scalar_quantity(f'std(motion) per cluster', motion_std_p)
            ps_points.add_vector_quantity(f'mean motion per cluster', mean_motion_p)
            ps_points.add_scalar_quantity(f'velocity per cluster', mean_velocity_p)
            #ps_points.add_scalar_quantity(f'moving objects', motion_mask_p)
            ps.register_point_cloud(f'moving objects', points[motion_mask_p], radius=4.5e-4, color=[1,1,0], enabled=True)
            ps.show()

        out_dict = {}
        out_dict['velocity'] = mean_velocity
        #out_dict['motion'] = mean_motion
        out_dict['std'] = motion_std
        out_dict['point2cluster'] = point2cluster
        out_dict['valid_idx'] = valid_idx
        out_dict['obj_idx'] = torch.where(obj_mask)[0]
        
        if self.eval_stats:
            if self.visualize:
                ps.register_point_cloud(f'gt objects', points[obj_mask], radius=4e-4, color=[1,0,0], enabled=True)

            top = (motion_mask_p & obj_mask).float().sum()
            bottom = motion_mask_p.float().sum()
            self.stats[0] += top.item()
            self.stats[1] += bottom.item()
            if self.stats[1] > 0:
                print(f'TP/(TP+FP)={self.stats[0]/self.stats[1]:.6f}, TP={self.stats[0]}, FP={self.stats[1]-self.stats[0]}')
            out_dict['moving'] = obj_mask

        return out_dict

    def __call__(self, results):
        T = results['pose']
        Tinv = np.linalg.inv(T)
        pc = self._range_filter(results['points'])
        ts = [results['timestamp']]
        sweep_points = [pc]
        if len(results['sweeps']) > self.sweeps_num:
            results['sweeps'] = results['sweeps'][:self.sweeps_num]
        gs = GridSampling(0.1)
        for s in results['sweeps']:
            points = self._load_points(s['velodyne_path'])
            points = self._range_filter(points)
            Ti = Tinv @ s['pose']
            points.rotate(Ti[:3, :3].T)
            points.translate(Ti[:3, 3])
            points.tensor = gs(Data(pos=points.tensor)).pos
            sweep_points.append(points)
            ts.append(s['timestamp'])
         
        if len(sweep_points) > 4:
            N = len(sweep_points)
            vel = [[] for i in range(N)]
            normals = []
            gt_bbox = results['gt_bboxes_3d']
            mask = gt_bbox.points_in_boxes(sweep_points[0].tensor[:, :3].cuda()).cpu()
            seg_mask = torch.zeros_like(mask)
            segments = {}
            for i in range(mask.max()-1):
                segment = segments.get(results['gt_names'][i], [])
                segment.append(torch.where(mask == i)[0])
                segments[results['gt_names'][i]] = segment
            for name in segments.keys():
                if len(segments[name]) == 1:
                    segments[name] = segments[name][0]
                else:
                    segments[name] = torch.cat(segments[name], dim=0)
            filename = results['pts_filename']
            filename_p = filename.replace('velodyne', 'motion_mask').replace('.bin', '.pth')
            if not os.path.exists(filename_p):
                motion_dict = \
                    self.recursive_segment(
                        sweep_points[0].tensor[:, :3], 
                        [sweep_points[i].tensor[:, :3] 
                            for i in range(1, len(sweep_points))], 
                        segments)
                #motion_dict['points'] = sweep_points[0].tensor[:, :3]
                #for i in range(1, len(sweep_points)):
                #    motion_dict[f'ref-{i-1}'] = sweep_points[i].tensor[:, :3]
                print(f'saving to {filename_p}')
                torch.save(motion_dict, filename_p)
        self.counter += 1
            
        return results

@PIPELINES.register_module()
class RandomFlip3D(RandomFlip):
    """Flip the points & bbox.

    If the input dict contains the key "flip", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.

    Args:
        sync_2d (bool, optional): Whether to apply flip according to the 2D
            images. If True, it will apply the same flip as that to 2D images.
            If False, it will decide whether to flip randomly and independently
            to that of 2D images. Defaults to True.
        flip_ratio_bev_horizontal (float, optional): The flipping probability
            in horizontal direction. Defaults to 0.0.
        flip_ratio_bev_vertical (float, optional): The flipping probability
            in vertical direction. Defaults to 0.0.
    """

    def __init__(self,
                 sync_2d=True,
                 flip_ratio_bev_horizontal=0.0,
                 flip_ratio_bev_vertical=0.0,
                 **kwargs):
        super(RandomFlip3D, self).__init__(
            flip_ratio=flip_ratio_bev_horizontal, **kwargs)
        self.sync_2d = sync_2d
        self.flip_ratio_bev_vertical = flip_ratio_bev_vertical
        if flip_ratio_bev_horizontal is not None:
            assert isinstance(
                flip_ratio_bev_horizontal,
                (int, float)) and 0 <= flip_ratio_bev_horizontal <= 1
        if flip_ratio_bev_vertical is not None:
            assert isinstance(
                flip_ratio_bev_vertical,
                (int, float)) and 0 <= flip_ratio_bev_vertical <= 1

    def random_flip_data_3d(self, input_dict, direction='horizontal'):
        """Flip 3D data randomly.

        Args:
            input_dict (dict): Result dict from loading pipeline.
            direction (str): Flip direction. Default: horizontal.

        Returns:
            dict: Flipped results, 'points', 'bbox3d_fields' keys are \
                updated in the result dict.
        """
        assert direction in ['horizontal', 'vertical']
        if len(input_dict['bbox3d_fields']) == 0:  # test mode
            input_dict['bbox3d_fields'].append('empty_box3d')
            input_dict['empty_box3d'] = input_dict['box_type_3d'](
                np.array([], dtype=np.float32))
        assert len(input_dict['bbox3d_fields']) == 1
        for key in input_dict['bbox3d_fields']:
            if 'points' in input_dict:
                input_dict['points'] = input_dict[key].flip(
                    direction, points=input_dict['points'])
            else:
                input_dict[key].flip(direction)
        if 'centers2d' in input_dict:
            assert self.sync_2d is True and direction == 'horizontal', \
                'Only support sync_2d=True and horizontal flip with images'
            w = input_dict['img_shape'][1]
            input_dict['centers2d'][..., 0] = \
                w - input_dict['centers2d'][..., 0]

    def __call__(self, input_dict):
        """Call function to flip points, values in the ``bbox3d_fields`` and \
        also flip 2D image and its annotations.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Flipped results, 'flip', 'flip_direction', \
                'pcd_horizontal_flip' and 'pcd_vertical_flip' keys are added \
                into result dict.
        """
        # filp 2D image and its annotations
        super(RandomFlip3D, self).__call__(input_dict)

        if self.sync_2d:
            input_dict['pcd_horizontal_flip'] = input_dict['flip']
            input_dict['pcd_vertical_flip'] = False
        else:
            if 'pcd_horizontal_flip' not in input_dict:
                flip_horizontal = True if np.random.rand(
                ) < self.flip_ratio else False
                input_dict['pcd_horizontal_flip'] = flip_horizontal
            if 'pcd_vertical_flip' not in input_dict:
                flip_vertical = True if np.random.rand(
                ) < self.flip_ratio_bev_vertical else False
                input_dict['pcd_vertical_flip'] = flip_vertical

        if 'transformation_3d_flow' not in input_dict:
            input_dict['transformation_3d_flow'] = []

        if input_dict['pcd_horizontal_flip']:
            self.random_flip_data_3d(input_dict, 'horizontal')
            input_dict['transformation_3d_flow'].extend(['HF'])
        if input_dict['pcd_vertical_flip']:
            self.random_flip_data_3d(input_dict, 'vertical')
            input_dict['transformation_3d_flow'].extend(['VF'])
        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(sync_2d={self.sync_2d},'
        repr_str += f' flip_ratio_bev_vertical={self.flip_ratio_bev_vertical})'
        return repr_str

@PIPELINES.register_module()
class RemoveBackground(object):
    def __call__(self, input_dict):
        gt_bboxes_3d = input_dict['gt_bboxes_3d']
        points = input_dict['points']
        indices = points_in_boxes_cpu(points[:, :3].tensor, gt_bboxes_3d.tensor[:, :7])
        input_dict['points'] = points[indices.sum(0) > 0]
        return input_dict

@PIPELINES.register_module()
class RemoveUnusedClasses(object):
    def __init__(self, classes):
        self.classes = classes

    def __call__(self, input_dict):
        gt_bboxes_3d = input_dict['gt_bboxes_3d']
        gt_names = input_dict['gt_names']
        points = input_dict['points']
        removed_box_indices = []
        remain_box_indices = []
        for i, name in enumerate(gt_names):
            if name not in self.classes:
                removed_box_indices.append(i)
            else:
                remain_box_indices.append(i)
        input_dict['gt_names'] = input_dict['gt_names'][remain_box_indices]
        removed_box_indices = np.array(removed_box_indices).astype(np.int32)
        remain_box_indices = np.array(remain_box_indices).astype(np.int32)
        indices = points_in_boxes_cpu(points[:, :3].tensor, gt_bboxes_3d.tensor[removed_box_indices, :7])
        input_dict['points'] = points[indices.sum(0) == 0]
        gt_bboxes_3d = gt_bboxes_3d[remain_box_indices]
        input_dict['gt_bboxes_3d'] = gt_bboxes_3d
        input_dict['gt_labels_3d'] = input_dict['gt_labels_3d'][remain_box_indices]
        return input_dict

@PIPELINES.register_module()
class ObjectSample(object):
    """Sample GT objects to the data.

    Args:
        db_sampler (dict): Config dict of the database sampler.
        sample_2d (bool): Whether to also paste 2D image patch to the images
            This should be true when applying multi-modality cut-and-paste.
            Defaults to False.
    """

    def __init__(self, db_sampler, sample_2d=False):
        self.sampler_cfg = db_sampler
        self.sample_2d = sample_2d
        if 'type' not in db_sampler.keys():
            db_sampler['type'] = 'DataBaseSampler'
        self.db_sampler = build_from_cfg(db_sampler, OBJECTSAMPLERS)

    @staticmethod
    def remove_points_in_boxes(points, boxes):
        """Remove the points in the sampled bounding boxes.

        Args:
            points (:obj:`BasePoints`): Input point cloud array.
            boxes (np.ndarray): Sampled ground truth boxes.

        Returns:
            np.ndarray: Points with those in the boxes removed.
        """
        masks = box_np_ops.points_in_rbbox(points.coord.numpy(), boxes)
        points = points[np.logical_not(masks.any(-1))]
        return points

    def __call__(self, input_dict):
        """Call function to sample ground truth objects to the data.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after object sampling augmentation, \
                'points', 'gt_bboxes_3d', 'gt_labels_3d' keys are updated \
                in the result dict.
        """
        gt_bboxes_3d = input_dict['gt_bboxes_3d']
        gt_labels_3d = input_dict['gt_labels_3d']
        gt_names = input_dict['gt_names']
        # change to float for blending operation
        points = input_dict['points']
        if self.sample_2d:
            img = input_dict['img']
            gt_bboxes_2d = input_dict['gt_bboxes']
            # Assume for now 3D & 2D bboxes are the same
            sampled_dict = self.db_sampler.sample_all(
                gt_bboxes_3d.tensor.numpy(),
                gt_labels_3d,
                gt_bboxes_2d=gt_bboxes_2d,
                img=img)
        else:
            sampled_dict = self.db_sampler.sample_all(
                gt_bboxes_3d.tensor.numpy(), gt_labels_3d, img=None)

        if sampled_dict is not None:
            sampled_gt_bboxes_3d = sampled_dict['gt_bboxes_3d']
            sampled_points = sampled_dict['points']
            sampled_gt_labels = sampled_dict['gt_labels_3d']
            sampled_gt_names = sampled_dict['gt_names']

            gt_labels_3d = np.concatenate([gt_labels_3d, sampled_gt_labels],
                                          axis=0)
            gt_names = np.concatenate([gt_names, sampled_gt_names])
            gt_bboxes_3d = gt_bboxes_3d.new_box(
                np.concatenate(
                    [gt_bboxes_3d.tensor.numpy(), sampled_gt_bboxes_3d]))
            
            #centers = gt_bboxes_3d.center.numpy()
            #dist_matrix = np.linalg.norm(centers[:, np.newaxis, :] - centers[np.newaxis, ...], ord=2, axis=-1) + np.eye(centers.shape[0])*1e10
            #mean_min_dist = np.mean(dist_matrix.min(-1))
            #print('mean(min_dist)={}'.format(mean_min_dist))
            #assert False

            points = self.remove_points_in_boxes(points, sampled_gt_bboxes_3d)
            # check the points dimension
            points = points.cat([sampled_points, points])

            if self.sample_2d:
                sampled_gt_bboxes_2d = sampled_dict['gt_bboxes_2d']
                gt_bboxes_2d = np.concatenate(
                    [gt_bboxes_2d, sampled_gt_bboxes_2d]).astype(np.float32)

                input_dict['gt_bboxes'] = gt_bboxes_2d
                input_dict['img'] = sampled_dict['img']

        input_dict['gt_bboxes_3d'] = gt_bboxes_3d
        input_dict['gt_labels_3d'] = gt_labels_3d.astype(np.long)
        input_dict['gt_names'] = gt_names
        input_dict['points'] = points

        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f' sample_2d={self.sample_2d},'
        repr_str += f' data_root={self.sampler_cfg.data_root},'
        repr_str += f' info_path={self.sampler_cfg.info_path},'
        repr_str += f' rate={self.sampler_cfg.rate},'
        repr_str += f' prepare={self.sampler_cfg.prepare},'
        repr_str += f' classes={self.sampler_cfg.classes},'
        repr_str += f' sample_groups={self.sampler_cfg.sample_groups}'
        return repr_str


@PIPELINES.register_module()
class ObjectNoise(object):
    """Apply noise to each GT objects in the scene.

    Args:
        translation_std (list[float], optional): Standard deviation of the
            distribution where translation noise are sampled from.
            Defaults to [0.25, 0.25, 0.25].
        global_rot_range (list[float], optional): Global rotation to the scene.
            Defaults to [0.0, 0.0].
        rot_range (list[float], optional): Object rotation range.
            Defaults to [-0.15707963267, 0.15707963267].
        num_try (int, optional): Number of times to try if the noise applied is
            invalid. Defaults to 100.
    """

    def __init__(self,
                 translation_std=[0.25, 0.25, 0.25],
                 global_rot_range=[0.0, 0.0],
                 rot_range=[-0.15707963267, 0.15707963267],
                 num_try=100):
        self.translation_std = translation_std
        self.global_rot_range = global_rot_range
        self.rot_range = rot_range
        self.num_try = num_try

    def __call__(self, input_dict):
        """Call function to apply noise to each ground truth in the scene.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after adding noise to each object, \
                'points', 'gt_bboxes_3d' keys are updated in the result dict.
        """
        gt_bboxes_3d = input_dict['gt_bboxes_3d']
        points = input_dict['points']

        # TODO: check this inplace function
        numpy_box = gt_bboxes_3d.tensor.numpy()
        numpy_points = points.tensor.numpy()

        noise_per_object_v3_(
            numpy_box,
            numpy_points,
            rotation_perturb=self.rot_range,
            center_noise_std=self.translation_std,
            global_random_rot_range=self.global_rot_range,
            num_try=self.num_try)

        input_dict['gt_bboxes_3d'] = gt_bboxes_3d.new_box(numpy_box)
        input_dict['points'] = points.new_point(numpy_points)
        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(num_try={self.num_try},'
        repr_str += f' translation_std={self.translation_std},'
        repr_str += f' global_rot_range={self.global_rot_range},'
        repr_str += f' rot_range={self.rot_range})'
        return repr_str


@PIPELINES.register_module()
class GlobalAlignment(object):
    """Apply global alignment to 3D scene points by rotation and translation.

    Args:
        rotation_axis (int): Rotation axis for points and bboxes rotation.

    Note:
        We do not record the applied rotation and translation as in \
            GlobalRotScaleTrans. Because usually, we do not need to reverse \
            the alignment step.
        For example, ScanNet 3D detection task uses aligned ground-truth \
            bounding boxes for evaluation.
    """

    def __init__(self, rotation_axis):
        self.rotation_axis = rotation_axis

    def _trans_points(self, input_dict, trans_factor):
        """Private function to translate points.

        Args:
            input_dict (dict): Result dict from loading pipeline.
            trans_factor (np.ndarray): Translation vector to be applied.

        Returns:
            dict: Results after translation, 'points' is updated in the dict.
        """
        input_dict['points'].translate(trans_factor)

    def _rot_points(self, input_dict, rot_mat):
        """Private function to rotate bounding boxes and points.

        Args:
            input_dict (dict): Result dict from loading pipeline.
            rot_mat (np.ndarray): Rotation matrix to be applied.

        Returns:
            dict: Results after rotation, 'points' is updated in the dict.
        """
        # input should be rot_mat_T so I transpose it here
        input_dict['points'].rotate(rot_mat.T)

    def _check_rot_mat(self, rot_mat):
        """Check if rotation matrix is valid for self.rotation_axis.

        Args:
            rot_mat (np.ndarray): Rotation matrix to be applied.
        """
        is_valid = np.allclose(np.linalg.det(rot_mat), 1.0)
        valid_array = np.zeros(3)
        valid_array[self.rotation_axis] = 1.0
        is_valid &= (rot_mat[self.rotation_axis, :] == valid_array).all()
        is_valid &= (rot_mat[:, self.rotation_axis] == valid_array).all()
        assert is_valid, f'invalid rotation matrix {rot_mat}'

    def __call__(self, input_dict):
        """Call function to shuffle points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after global alignment, 'points' and keys in \
                input_dict['bbox3d_fields'] are updated in the result dict.
        """
        assert 'axis_align_matrix' in input_dict['ann_info'].keys(), \
            'axis_align_matrix is not provided in GlobalAlignment'

        axis_align_matrix = input_dict['ann_info']['axis_align_matrix']
        assert axis_align_matrix.shape == (4, 4), \
            f'invalid shape {axis_align_matrix.shape} for axis_align_matrix'
        rot_mat = axis_align_matrix[:3, :3]
        trans_vec = axis_align_matrix[:3, -1]

        self._check_rot_mat(rot_mat)
        self._rot_points(input_dict, rot_mat)
        self._trans_points(input_dict, trans_vec)

        return input_dict

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(rotation_axis={self.rotation_axis})'
        return repr_str


@PIPELINES.register_module()
class GlobalRotScaleTrans(object):
    """Apply global rotation, scaling and translation to a 3D scene.

    Args:
        rot_range (list[float]): Range of rotation angle.
            Defaults to [-0.78539816, 0.78539816] (close to [-pi/4, pi/4]).
        scale_ratio_range (list[float]): Range of scale ratio.
            Defaults to [0.95, 1.05].
        translation_std (list[float]): The standard deviation of ranslation
            noise. This apply random translation to a scene by a noise, which
            is sampled from a gaussian distribution whose standard deviation
            is set by ``translation_std``. Defaults to [0, 0, 0]
        shift_height (bool): Whether to shift height.
            (the fourth dimension of indoor points) when scaling.
            Defaults to False.
    """

    def __init__(self,
                 rot_range=[-0.78539816, 0.78539816],
                 scale_ratio_range=[0.95, 1.05],
                 translation_std=[0, 0, 0],
                 shift_height=False):
        seq_types = (list, tuple, np.ndarray)
        if not isinstance(rot_range, seq_types):
            assert isinstance(rot_range, (int, float)), \
                f'unsupported rot_range type {type(rot_range)}'
            rot_range = [-rot_range, rot_range]
        self.rot_range = rot_range

        assert isinstance(scale_ratio_range, seq_types), \
            f'unsupported scale_ratio_range type {type(scale_ratio_range)}'
        self.scale_ratio_range = scale_ratio_range

        if not isinstance(translation_std, seq_types):
            assert isinstance(translation_std, (int, float)), \
                f'unsupported translation_std type {type(translation_std)}'
            translation_std = [
                translation_std, translation_std, translation_std
            ]
        self.translation_std = translation_std
        self.shift_height = shift_height

    def _trans_bbox_points(self, input_dict):
        """Private function to translate bounding boxes and points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after translation, 'points', 'pcd_trans' \
                and keys in input_dict['bbox3d_fields'] are updated \
                in the result dict.
        """
        translation_std = np.array(self.translation_std, dtype=np.float32)
        trans_factor = np.random.normal(scale=translation_std, size=3).T

        input_dict['points'].translate(trans_factor)
        input_dict['pcd_trans'] = trans_factor
        for key in input_dict['bbox3d_fields']:
            input_dict[key].translate(trans_factor)

    def _rot_bbox_points(self, input_dict):
        """Private function to rotate bounding boxes and points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after rotation, 'points', 'pcd_rotation' \
                and keys in input_dict['bbox3d_fields'] are updated \
                in the result dict.
        """
        rotation = self.rot_range
        noise_rotation = np.random.uniform(rotation[0], rotation[1])

        # if no bbox in input_dict, only rotate points
        if len(input_dict['bbox3d_fields']) == 0:
            rot_mat_T = input_dict['points'].rotate(noise_rotation)
            input_dict['pcd_rotation'] = rot_mat_T
            return

        # rotate points with bboxes
        for key in input_dict['bbox3d_fields']:
            if len(input_dict[key].tensor) != 0:
                points, rot_mat_T = input_dict[key].rotate(
                    noise_rotation, input_dict['points'])
                input_dict['points'] = points
                input_dict['pcd_rotation'] = rot_mat_T

    def _scale_bbox_points(self, input_dict):
        """Private function to scale bounding boxes and points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after scaling, 'points'and keys in \
                input_dict['bbox3d_fields'] are updated in the result dict.
        """
        scale = input_dict['pcd_scale_factor']
        points = input_dict['points']
        points.scale(scale)
        if self.shift_height:
            assert 'height' in points.attribute_dims.keys(), \
                'setting shift_height=True but points have no height attribute'
            points.tensor[:, points.attribute_dims['height']] *= scale
        input_dict['points'] = points

        for key in input_dict['bbox3d_fields']:
            input_dict[key].scale(scale)

    def _random_scale(self, input_dict):
        """Private function to randomly set the scale factor.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after scaling, 'pcd_scale_factor' are updated \
                in the result dict.
        """
        scale_factor = np.random.uniform(self.scale_ratio_range[0],
                                         self.scale_ratio_range[1])
        input_dict['pcd_scale_factor'] = scale_factor

    def __call__(self, input_dict):
        """Private function to rotate, scale and translate bounding boxes and \
        points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after scaling, 'points', 'pcd_rotation',
                'pcd_scale_factor', 'pcd_trans' and keys in \
                input_dict['bbox3d_fields'] are updated in the result dict.
        """
        if 'transformation_3d_flow' not in input_dict:
            input_dict['transformation_3d_flow'] = []

        self._rot_bbox_points(input_dict)

        if 'pcd_scale_factor' not in input_dict:
            self._random_scale(input_dict)
        self._scale_bbox_points(input_dict)

        self._trans_bbox_points(input_dict)

        input_dict['transformation_3d_flow'].extend(['R', 'S', 'T'])
        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(rot_range={self.rot_range},'
        repr_str += f' scale_ratio_range={self.scale_ratio_range},'
        repr_str += f' translation_std={self.translation_std},'
        repr_str += f' shift_height={self.shift_height})'
        return repr_str


@PIPELINES.register_module()
class PointShuffle(object):
    """Shuffle input points."""

    def __call__(self, input_dict):
        """Call function to shuffle points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'points', 'pts_instance_mask' \
                and 'pts_semantic_mask' keys are updated in the result dict.
        """
        idx = input_dict['points'].shuffle()
        idx = idx.numpy()

        pts_instance_mask = input_dict.get('pts_instance_mask', None)
        pts_semantic_mask = input_dict.get('pts_semantic_mask', None)

        if pts_instance_mask is not None:
            input_dict['pts_instance_mask'] = pts_instance_mask[idx]

        if pts_semantic_mask is not None:
            input_dict['pts_semantic_mask'] = pts_semantic_mask[idx]

        return input_dict

    def __repr__(self):
        return self.__class__.__name__


@PIPELINES.register_module()
class ObjectRangeFilter(object):
    """Filter objects by the range.

    Args:
        point_cloud_range (list[float]): Point cloud range.
    """

    def __init__(self, point_cloud_range):
        self.pcd_range = np.array(point_cloud_range, dtype=np.float32)
        self.bev_range = self.pcd_range[[0, 1, 3, 4]]

    def __call__(self, input_dict):
        """Call function to filter objects by the range.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'gt_bboxes_3d', 'gt_labels_3d' \
                keys are updated in the result dict.
        """
        gt_bboxes_3d = input_dict['gt_bboxes_3d']
        gt_labels_3d = input_dict['gt_labels_3d']
        mask = gt_bboxes_3d.in_range_bev(self.bev_range)
        gt_bboxes_3d = gt_bboxes_3d[mask]
        # mask is a torch tensor but gt_labels_3d is still numpy array
        # using mask to index gt_labels_3d will cause bug when
        # len(gt_labels_3d) == 1, where mask=1 will be interpreted
        # as gt_labels_3d[1] and cause out of index error
        try:
            gt_labels_3d = gt_labels_3d[mask.numpy().astype(np.bool)]
        except Exception as e:
            import ipdb; ipdb.set_trace()
            print(e)

        # limit rad to [-pi, pi]
        gt_bboxes_3d.limit_yaw(offset=0.5, period=2 * np.pi)
        input_dict['gt_bboxes_3d'] = gt_bboxes_3d
        input_dict['gt_labels_3d'] = gt_labels_3d

        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(point_cloud_range={self.pcd_range.tolist()})'
        return repr_str


@PIPELINES.register_module()
class PointsRangeFilter(object):
    """Filter points by the range.

    Args:
        point_cloud_range (list[float]): Point cloud range.
    """

    def __init__(self, point_cloud_range):
        self.pcd_range = np.array(point_cloud_range, dtype=np.float32)

    def __call__(self, input_dict):
        """Call function to filter points by the range.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'points', 'pts_instance_mask' \
                and 'pts_semantic_mask' keys are updated in the result dict.
        """
        points = input_dict['points']
        points_mask = points.in_range_3d(self.pcd_range)
        clean_points = points[points_mask]
        input_dict['points'] = clean_points
        points_mask = points_mask.numpy()

        pts_instance_mask = input_dict.get('pts_instance_mask', None)
        pts_semantic_mask = input_dict.get('pts_semantic_mask', None)

        if pts_instance_mask is not None:
            input_dict['pts_instance_mask'] = pts_instance_mask[points_mask]

        if pts_semantic_mask is not None:
            input_dict['pts_semantic_mask'] = pts_semantic_mask[points_mask]

        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(point_cloud_range={self.pcd_range.tolist()})'
        return repr_str


@PIPELINES.register_module()
class ObjectNameFilter(object):
    """Filter GT objects by their names.

    Args:
        classes (list[str]): List of class names to be kept for training.
    """

    def __init__(self, classes):
        self.classes = classes
        self.labels = list(range(len(self.classes)))

    def __call__(self, input_dict):
        """Call function to filter objects by their names.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'gt_bboxes_3d', 'gt_labels_3d' \
                keys are updated in the result dict.
        """
        gt_labels_3d = input_dict['gt_labels_3d']
        gt_bboxes_mask = np.array([n in self.labels for n in gt_labels_3d],
                                  dtype=np.bool_)
        input_dict['gt_bboxes_3d'] = input_dict['gt_bboxes_3d'][gt_bboxes_mask]
        input_dict['gt_labels_3d'] = input_dict['gt_labels_3d'][gt_bboxes_mask]

        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(classes={self.classes})'
        return repr_str

@PIPELINES.register_module()
class FPSPointSample(object):
    """Furthest Point Sampling (FPS).

    Sampling data to a certain number, use FPS instead of random sampling.

    Args:
        name (str): Name of the dataset.
        num_points (int): Number of points to be sampled.
    """

    def __init__(self, num_points):
        self.num_points = num_points
        #self.fps = Points_Sampler([num_points])

    def points_fps_sampling(self,
                            points,
                            num_samples,
                            replace=None,
                            return_choices=False):
        """FPS point sampling.

        Sample points to a certain number, use FPS instead of random sampling.

        Args:
            points (np.ndarray | :obj:`BasePoints`): 3D Points.
            num_samples (int): Number of samples to be sampled.
            replace (bool): Whether the sample is with or without replacement.
            Defaults to None.
            return_choices (bool): Whether return choice. Defaults to False.

        Returns:
            tuple[np.ndarray] | np.ndarray:

                - points (np.ndarray | :obj:`BasePoints`): 3D Points.
                - choices (np.ndarray, optional): The generated random samples.
        """
        if replace is None:
            replace = (points.shape[0] < num_samples)
        if replace:
            choices = np.random.choice(
                points.shape[0], num_samples, replace=replace)
        else:
            import ipdb; ipdb.set_trace()
            if isinstance(points, np.ndarray):
                points_torch = torch.as_tensor(points)
                ratio = (self.num_points + 1) / points_torch.shape[0]
                choices = fps(points_torch, ratio=ratio)[:self.num_points]
                #choices = self.fps(points_torch[:, :, :3], points_torch[:, :, 3:].transpose(1, 2))[0]
            else:
                points_torch = points.tensor
                ratio = (self.num_points + 1) / points_torch.shape[0]
                choices = fps(points_torch, ratio=ratio)[:self.num_points]
                #choices = self.fps(points_torch[:, :, :3], points_torch[:, :, 3:].transpose(1, 2))[0]
        if return_choices:
            return points[choices], choices
        else:
            return points[choices]

    def __call__(self, results):
        """Call function to sample points to in indoor scenes.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after sampling, 'points', 'pts_instance_mask' \
                and 'pts_semantic_mask' keys are updated in the result dict.
        """
        points = results['points']
        points, choices = self.points_fps_sampling(
            points, self.num_points, return_choices=True)
        results['points'] = points

        pts_instance_mask = results.get('pts_instance_mask', None)
        pts_semantic_mask = results.get('pts_semantic_mask', None)

        if pts_instance_mask is not None:
            pts_instance_mask = pts_instance_mask[choices]
            results['pts_instance_mask'] = pts_instance_mask

        if pts_semantic_mask is not None:
            pts_semantic_mask = pts_semantic_mask[choices]
            results['pts_semantic_mask'] = pts_semantic_mask

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(num_points={self.num_points})'
        return repr_str


@PIPELINES.register_module()
class IndoorPointSample(object):
    """Indoor point sample.

    Sampling data to a certain number.

    Args:
        name (str): Name of the dataset.
        num_points (int): Number of points to be sampled.
    """

    def __init__(self, num_points):
        self.num_points = num_points

    def points_random_sampling(self,
                               points,
                               num_samples,
                               replace=None,
                               return_choices=False):
        """Points random sampling.

        Sample points to a certain number.

        Args:
            points (np.ndarray | :obj:`BasePoints`): 3D Points.
            num_samples (int): Number of samples to be sampled.
            replace (bool): Whether the sample is with or without replacement.
            Defaults to None.
            return_choices (bool): Whether return choice. Defaults to False.

        Returns:
            tuple[np.ndarray] | np.ndarray:

                - points (np.ndarray | :obj:`BasePoints`): 3D Points.
                - choices (np.ndarray, optional): The generated random samples.
        """
        if replace is None:
            replace = (points.shape[0] < num_samples)
        choices = np.random.choice(
            points.shape[0], num_samples, replace=replace)
        if return_choices:
            return points[choices], choices
        else:
            return points[choices]

    def __call__(self, results):
        """Call function to sample points to in indoor scenes.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after sampling, 'points', 'pts_instance_mask' \
                and 'pts_semantic_mask' keys are updated in the result dict.
        """
        points = results['points']
        points, choices = self.points_random_sampling(
            points, self.num_points, return_choices=True)
        results['points'] = points

        pts_instance_mask = results.get('pts_instance_mask', None)
        pts_semantic_mask = results.get('pts_semantic_mask', None)

        if pts_instance_mask is not None:
            pts_instance_mask = pts_instance_mask[choices]
            results['pts_instance_mask'] = pts_instance_mask

        if pts_semantic_mask is not None:
            pts_semantic_mask = pts_semantic_mask[choices]
            results['pts_semantic_mask'] = pts_semantic_mask

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(num_points={self.num_points})'
        return repr_str


@PIPELINES.register_module()
class IndoorPatchPointSample(object):
    r"""Indoor point sample within a patch. Modified from `PointNet++ <https://
    github.com/charlesq34/pointnet2/blob/master/scannet/scannet_dataset.py>`_.

    Sampling data to a certain number for semantic segmentation.

    Args:
        num_points (int): Number of points to be sampled.
        block_size (float, optional): Size of a block to sample points from.
            Defaults to 1.5.
        sample_rate (float, optional): Stride used in sliding patch generation.
            Defaults to 1.0.
        ignore_index (int, optional): Label index that won't be used for the
            segmentation task. This is set in PointSegClassMapping as neg_cls.
            Defaults to None.
        use_normalized_coord (bool, optional): Whether to use normalized xyz as
            additional features. Defaults to False.
        num_try (int, optional): Number of times to try if the patch selected
            is invalid. Defaults to 10.
    """

    def __init__(self,
                 num_points,
                 block_size=1.5,
                 sample_rate=1.0,
                 ignore_index=None,
                 use_normalized_coord=False,
                 num_try=10):
        self.num_points = num_points
        self.block_size = block_size
        self.sample_rate = sample_rate
        self.ignore_index = ignore_index
        self.use_normalized_coord = use_normalized_coord
        self.num_try = num_try

    def _input_generation(self, coords, patch_center, coord_max, attributes,
                          attribute_dims, point_type):
        """Generating model input.

        Generate input by subtracting patch center and adding additional \
            features. Currently support colors and normalized xyz as features.

        Args:
            coords (np.ndarray): Sampled 3D Points.
            patch_center (np.ndarray): Center coordinate of the selected patch.
            coord_max (np.ndarray): Max coordinate of all 3D Points.
            attributes (np.ndarray): features of input points.
            attribute_dims (dict): Dictionary to indicate the meaning of extra
                dimension.
            point_type (type): class of input points inherited from BasePoints.

        Returns:
            :obj:`BasePoints`: The generated input data.
        """
        # subtract patch center, the z dimension is not centered
        centered_coords = coords.copy()
        centered_coords[:, 0] -= patch_center[0]
        centered_coords[:, 1] -= patch_center[1]

        if self.use_normalized_coord:
            normalized_coord = coords / coord_max
            attributes = np.concatenate([attributes, normalized_coord], axis=1)
            if attribute_dims is None:
                attribute_dims = dict()
            attribute_dims.update(
                dict(normalized_coord=[
                    attributes.shape[1], attributes.shape[1] +
                    1, attributes.shape[1] + 2
                ]))

        points = np.concatenate([centered_coords, attributes], axis=1)
        points = point_type(
            points, points_dim=points.shape[1], attribute_dims=attribute_dims)

        return points

    def _patch_points_sampling(self, points, sem_mask, replace=None):
        """Patch points sampling.

        First sample a valid patch.
        Then sample points within that patch to a certain number.

        Args:
            points (:obj:`BasePoints`): 3D Points.
            sem_mask (np.ndarray): semantic segmentation mask for input points.
            replace (bool): Whether the sample is with or without replacement.
                Defaults to None.

        Returns:
            tuple[:obj:`BasePoints`, np.ndarray] | :obj:`BasePoints`:

                - points (:obj:`BasePoints`): 3D Points.
                - choices (np.ndarray): The generated random samples.
        """
        coords = points.coord.numpy()
        attributes = points.tensor[:, 3:].numpy()
        attribute_dims = points.attribute_dims
        point_type = type(points)

        coord_max = np.amax(coords, axis=0)
        coord_min = np.amin(coords, axis=0)

        for i in range(self.num_try):
            # random sample a point as patch center
            cur_center = coords[np.random.choice(coords.shape[0])]

            # boundary of a patch
            cur_max = cur_center + np.array(
                [self.block_size / 2.0, self.block_size / 2.0, 0.0])
            cur_min = cur_center - np.array(
                [self.block_size / 2.0, self.block_size / 2.0, 0.0])
            cur_max[2] = coord_max[2]
            cur_min[2] = coord_min[2]
            cur_choice = np.sum(
                (coords >= (cur_min - 0.2)) * (coords <= (cur_max + 0.2)),
                axis=1) == 3

            if not cur_choice.any():  # no points in this patch
                continue

            cur_coords = coords[cur_choice, :]
            cur_sem_mask = sem_mask[cur_choice]

            # two criterion for patch sampling, adopted from PointNet++
            # points within selected patch shoule be scattered separately
            mask = np.sum(
                (cur_coords >= (cur_min - 0.01)) * (cur_coords <=
                                                    (cur_max + 0.01)),
                axis=1) == 3
            # not sure if 31, 31, 62 are just some big values used to transform
            # coords from 3d array to 1d and then check their uniqueness
            # this is used in all the ScanNet code following PointNet++
            vidx = np.ceil((cur_coords[mask, :] - cur_min) /
                           (cur_max - cur_min) * np.array([31.0, 31.0, 62.0]))
            vidx = np.unique(vidx[:, 0] * 31.0 * 62.0 + vidx[:, 1] * 62.0 +
                             vidx[:, 2])
            flag1 = len(vidx) / 31.0 / 31.0 / 62.0 >= 0.02

            # selected patch should contain enough annotated points
            if self.ignore_index is None:
                flag2 = True
            else:
                flag2 = np.sum(cur_sem_mask != self.ignore_index) / \
                               len(cur_sem_mask) >= 0.7

            if flag1 and flag2:
                break

        # random sample idx
        if replace is None:
            replace = (cur_sem_mask.shape[0] < self.num_points)
        choices = np.random.choice(
            np.where(cur_choice)[0], self.num_points, replace=replace)

        # construct model input
        points = self._input_generation(coords[choices], cur_center, coord_max,
                                        attributes[choices], attribute_dims,
                                        point_type)

        return points, choices

    def __call__(self, results):
        """Call function to sample points to in indoor scenes.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after sampling, 'points', 'pts_instance_mask' \
                and 'pts_semantic_mask' keys are updated in the result dict.
        """
        points = results['points']

        assert 'pts_semantic_mask' in results.keys(), \
            'semantic mask should be provided in training and evaluation'
        pts_semantic_mask = results['pts_semantic_mask']

        points, choices = self._patch_points_sampling(points,
                                                      pts_semantic_mask)

        results['points'] = points
        results['pts_semantic_mask'] = pts_semantic_mask[choices]
        pts_instance_mask = results.get('pts_instance_mask', None)
        if pts_instance_mask is not None:
            results['pts_instance_mask'] = pts_instance_mask[choices]

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(num_points={self.num_points},'
        repr_str += f' block_size={self.block_size},'
        repr_str += f' sample_rate={self.sample_rate},'
        repr_str += f' ignore_index={self.ignore_index},'
        repr_str += f' use_normalized_coord={self.use_normalized_coord},'
        repr_str += f' num_try={self.num_try})'
        return repr_str


@PIPELINES.register_module()
class BackgroundPointsFilter(object):
    """Filter background points near the bounding box.

    Args:
        bbox_enlarge_range (tuple[float], float): Bbox enlarge range.
    """

    def __init__(self, bbox_enlarge_range):
        assert (is_tuple_of(bbox_enlarge_range, float)
                and len(bbox_enlarge_range) == 3) \
            or isinstance(bbox_enlarge_range, float), \
            f'Invalid arguments bbox_enlarge_range {bbox_enlarge_range}'

        if isinstance(bbox_enlarge_range, float):
            bbox_enlarge_range = [bbox_enlarge_range] * 3
        self.bbox_enlarge_range = np.array(
            bbox_enlarge_range, dtype=np.float32)[np.newaxis, :]

    def __call__(self, input_dict):
        """Call function to filter points by the range.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'points', 'pts_instance_mask' \
                and 'pts_semantic_mask' keys are updated in the result dict.
        """
        points = input_dict['points']
        gt_bboxes_3d = input_dict['gt_bboxes_3d']

        gt_bboxes_3d_np = gt_bboxes_3d.tensor.numpy()
        gt_bboxes_3d_np[:, :3] = gt_bboxes_3d.gravity_center.numpy()
        enlarged_gt_bboxes_3d = gt_bboxes_3d_np.copy()
        enlarged_gt_bboxes_3d[:, 3:6] += self.bbox_enlarge_range
        points_numpy = points.tensor.numpy()
        foreground_masks = box_np_ops.points_in_rbbox(points_numpy,
                                                      gt_bboxes_3d_np)
        enlarge_foreground_masks = box_np_ops.points_in_rbbox(
            points_numpy, enlarged_gt_bboxes_3d)
        foreground_masks = foreground_masks.max(1)
        enlarge_foreground_masks = enlarge_foreground_masks.max(1)
        valid_masks = ~np.logical_and(~foreground_masks,
                                      enlarge_foreground_masks)

        input_dict['points'] = points[valid_masks]
        pts_instance_mask = input_dict.get('pts_instance_mask', None)
        if pts_instance_mask is not None:
            input_dict['pts_instance_mask'] = pts_instance_mask[valid_masks]

        pts_semantic_mask = input_dict.get('pts_semantic_mask', None)
        if pts_semantic_mask is not None:
            input_dict['pts_semantic_mask'] = pts_semantic_mask[valid_masks]
        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(bbox_enlarge_range={self.bbox_enlarge_range.tolist()})'
        return repr_str


@PIPELINES.register_module()
class VoxelBasedPointSampler(object):
    """Voxel based point sampler.

    Apply voxel sampling to multiple sweep points.

    Args:
        cur_sweep_cfg (dict): Config for sampling current points.
        prev_sweep_cfg (dict): Config for sampling previous points.
        time_dim (int): Index that indicate the time dimention
            for input points.
    """

    def __init__(self, cur_sweep_cfg, prev_sweep_cfg=None, time_dim=3):
        self.cur_voxel_generator = VoxelGenerator(**cur_sweep_cfg)
        self.cur_voxel_num = self.cur_voxel_generator._max_voxels
        self.time_dim = time_dim
        if prev_sweep_cfg is not None:
            assert prev_sweep_cfg['max_num_points'] == \
                cur_sweep_cfg['max_num_points']
            self.prev_voxel_generator = VoxelGenerator(**prev_sweep_cfg)
            self.prev_voxel_num = self.prev_voxel_generator._max_voxels
        else:
            self.prev_voxel_generator = None
            self.prev_voxel_num = 0

    def _sample_points(self, points, sampler, point_dim):
        """Sample points for each points subset.

        Args:
            points (np.ndarray): Points subset to be sampled.
            sampler (VoxelGenerator): Voxel based sampler for
                each points subset.
            point_dim (int): The dimention of each points

        Returns:
            np.ndarray: Sampled points.
        """
        voxels, coors, num_points_per_voxel = sampler.generate(points)
        if voxels.shape[0] < sampler._max_voxels:
            padding_points = np.zeros([
                sampler._max_voxels - voxels.shape[0], sampler._max_num_points,
                point_dim
            ],
                                      dtype=points.dtype)
            padding_points[:] = voxels[0]
            sample_points = np.concatenate([voxels, padding_points], axis=0)
        else:
            sample_points = voxels

        return sample_points

    def __call__(self, results):
        """Call function to sample points from multiple sweeps.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after sampling, 'points', 'pts_instance_mask' \
                and 'pts_semantic_mask' keys are updated in the result dict.
        """
        points = results['points']
        original_dim = points.shape[1]

        # TODO: process instance and semantic mask while _max_num_points
        # is larger than 1
        # Extend points with seg and mask fields
        map_fields2dim = []
        start_dim = original_dim
        points_numpy = points.tensor.numpy()
        extra_channel = [points_numpy]
        for idx, key in enumerate(results['pts_mask_fields']):
            map_fields2dim.append((key, idx + start_dim))
            extra_channel.append(results[key][..., None])

        start_dim += len(results['pts_mask_fields'])
        for idx, key in enumerate(results['pts_seg_fields']):
            map_fields2dim.append((key, idx + start_dim))
            extra_channel.append(results[key][..., None])

        points_numpy = np.concatenate(extra_channel, axis=-1)

        # Split points into two part, current sweep points and
        # previous sweeps points.
        # TODO: support different sampling methods for next sweeps points
        # and previous sweeps points.
        cur_points_flag = (points_numpy[:, self.time_dim] == 0)
        cur_sweep_points = points_numpy[cur_points_flag]
        prev_sweeps_points = points_numpy[~cur_points_flag]
        if prev_sweeps_points.shape[0] == 0:
            prev_sweeps_points = cur_sweep_points

        # Shuffle points before sampling
        np.random.shuffle(cur_sweep_points)
        np.random.shuffle(prev_sweeps_points)

        cur_sweep_points = self._sample_points(cur_sweep_points,
                                               self.cur_voxel_generator,
                                               points_numpy.shape[1])
        if self.prev_voxel_generator is not None:
            prev_sweeps_points = self._sample_points(prev_sweeps_points,
                                                     self.prev_voxel_generator,
                                                     points_numpy.shape[1])

            points_numpy = np.concatenate(
                [cur_sweep_points, prev_sweeps_points], 0)
        else:
            points_numpy = cur_sweep_points

        if self.cur_voxel_generator._max_num_points == 1:
            points_numpy = points_numpy.squeeze(1)
        results['points'] = points.new_point(points_numpy[..., :original_dim])

        # Restore the correspoinding seg and mask fields
        for key, dim_index in map_fields2dim:
            results[key] = points_numpy[..., dim_index]

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""

        def _auto_indent(repr_str, indent):
            repr_str = repr_str.split('\n')
            repr_str = [' ' * indent + t + '\n' for t in repr_str]
            repr_str = ''.join(repr_str)[:-1]
            return repr_str

        repr_str = self.__class__.__name__
        indent = 4
        repr_str += '(\n'
        repr_str += ' ' * indent + f'num_cur_sweep={self.cur_voxel_num},\n'
        repr_str += ' ' * indent + f'num_prev_sweep={self.prev_voxel_num},\n'
        repr_str += ' ' * indent + f'time_dim={self.time_dim},\n'
        repr_str += ' ' * indent + 'cur_voxel_generator=\n'
        repr_str += f'{_auto_indent(repr(self.cur_voxel_generator), 8)},\n'
        repr_str += ' ' * indent + 'prev_voxel_generator=\n'
        repr_str += f'{_auto_indent(repr(self.prev_voxel_generator), 8)})'
        return repr_str

