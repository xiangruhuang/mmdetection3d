import mmcv
import numpy as np
import os
import tempfile
import torch
from mmcv.utils import print_log, build_from_cfg
from os import path as osp

from torch.utils.data import Dataset
from mmdet.datasets import DATASETS, PIPELINES
from ..core.bbox import Box3DMode, points_cam2img, get_box_type
from .kitti_dataset import KittiDataset
from .builder import OBJECTSAMPLERS
from .pipelines import Compose
from mmdet3d.ops import knn
from torch_geometric.nn import knn as knn_cpu

@DATASETS.register_module()
class WaymoGTDataset(Dataset):
    def __init__(self,
                 db_sampler,
                 box_type_3d='LiDAR',
                 pipeline=None,
                 test_mode=False,
                 classes=None,
                 split=None,
                 gap=0.1,
                 load_interval=1,
                 visualize=False,
                 train_interval=8,
                 filter_by_points=dict(Car=30, Pedestrian=30),
                 maximum_samples=dict(Car=200, Pedestrian=200),
                 use_single=True,
                 use_couple=True,
                 points_loader=dict(
                     type='LoadPointsFromFile',
                     coord_type='LIDAR',
                     load_dim=5,
                     use_dim=[0,1,2])):
        super().__init__()
        if 'type' not in db_sampler.keys():
            db_sampler['type'] = 'DataBaseSampler'
        # self.db_sampler.sampler_dict['Car']._sampled_list
        # self.db_sampler.data_root, self.sampler
        self.box_type_3d, self.box_mode_3d = get_box_type(box_type_3d)
        self.CLASSES = classes
        self.cat2id = {name: i for i, name in enumerate(self.CLASSES)}
        if pipeline is not None:
            self.pipeline = Compose(pipeline)
        self.test_mode = test_mode
        self.gap = gap

        self.samples = {}
        self.indices = {}
        for cls in classes:
            self.samples[cls] = []
            self.indices[cls] = []
            for i, sample in enumerate(
                    self.db_sampler.sampler_dict[cls]._sampled_list):
                if sample['num_points_in_gt'] < filter_by_points[cls]:
                    continue
                self.samples[cls].append(sample)
                self.indices[cls].append(i)
                if len(self.samples[cls]) >= maximum_samples[cls]:
                    break

        import itertools
        self.scenes = []
        # Single Object Scenes
        if use_single:
            for cls in classes:
                for i in range(len(self.samples[cls])):
                    self.scenes.append({cls: i})

        if use_couple:
            if split == 'training':
                for cls1, cls2 in itertools.combinations(classes, 2):
                    for i in range(len(self.samples[cls1])):
                        for j in range(len(self.samples[cls2])):
                            if (i + j) % train_interval == 0:
                                self.scenes.append({cls1: i, cls2: j})
            else:
                for cls1, cls2 in itertools.combinations(classes, 2):
                    for i in range(len(self.samples[cls1])):
                        for j in range(len(self.samples[cls2])):
                            if (i + j) % train_interval != 0:
                                self.scenes.append({cls1: i, cls2: j})
       
        self.scenes = self.scenes[::load_interval]
        self.trans = {}

        self.points_loader = mmcv.build_from_cfg(points_loader, PIPELINES)
        
        if not self.test_mode:
            self._set_group_flag()
        if visualize:
            import polyscope as ps
            ps.init()
            for cls in classes:
                print(f'visualizing class={cls}')
                ps.remove_all_structures()
                for idx, sample in enumerate(self.samples[cls]):
                    pts_filename = os.path.join(
                        self.db_sampler.data_root, sample['path']
                    )
                    print(pts_filename)
                    results = dict(pts_filename=pts_filename)
                    cls_points = self.points_loader(results)['points'].tensor.cpu().numpy()
                    cls_points += np.array([(idx // 15) * 5, 0, (idx % 15) * 5])
                    ps.register_point_cloud(f'sample-{idx}', cls_points)
                ps.show()
   
    def __len__(self):
        return len(self.scenes)

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Standard input_dict consists of the
                data information.

                - sample_idx (str): sample index
                - pts_filename (str): filename of point clouds
                - img_prefix (str | None): prefix of image files
                - img_info (dict): image info
                - lidar2img (list[np.ndarray], optional): transformations from
                    lidar to different cameras
                - ann_info (dict): annotation info
        """
        scene = self.scenes[index]
        points, gt_labels = None, None
        for cls in self.CLASSES:
            idx = scene.get(cls, None)
            if idx is not None:
                sample = self.samples[cls][idx]
                pts_filename = os.path.join(
                    self.db_sampler.data_root, sample['path']
                )
                results = dict(pts_filename=pts_filename)
                cls_points = self.points_loader(results)['points']
                cls_gt_labels = torch.zeros(cls_points.shape[0]).long()+self.cat2id[cls]
                if points is None:
                    points = cls_points
                    gt_labels = cls_gt_labels
                else:
                    max0, min0 = points.tensor.max(0)[0], points.tensor.min(0)[0]
                    #print(max0.shape, min0.shape, max0, min0)
                    max1, min1 = cls_points.tensor.max(0)[0], cls_points.tensor.min(0)[0]
                    dim, sign = np.random.randint(3), np.random.randint(2)
                    trans = self.trans.get((index, cls, dim, sign), None)
                    if trans is None:
                        trans = torch.zeros(3)
                        if sign == 0:
                            trans[dim] = -(max1[dim] + self.gap - min0[dim])
                        else:
                            trans[dim] = -(min1[dim] - self.gap - max0[dim])
                        cls_points.tensor += trans
                        _, indices = knn_cpu(points.tensor, cls_points.tensor, k=1)
                        min_dist = (cls_points.tensor - points.tensor[indices]).norm(p=2, dim=-1)
                        idx = min_dist.argmin()
                        dr = points.tensor[indices[idx]] - cls_points.tensor[idx]
                        unit_dr = dr / dr.norm(p=2, dim=-1)
                        trans_i = unit_dr * (min_dist.min() - self.gap)
                        trans += trans_i
                        cls_points.translate(trans_i)
                        self.trans[(index, cls, dim, sign)] = trans
                    else:
                        cls_points.translate(trans)
                    points = points.cat([points, cls_points])
                    gt_labels = torch.cat([gt_labels, cls_gt_labels], dim=0)

                points.translate(-points.tensor.mean(0)[:3])
        input_dict = dict(
            points = points,
            gt_labels = gt_labels)

        return input_dict
    
    def prepare_train_data(self, index):
        """Training data preparation.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Training data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        #if input_dict is None:
        #    return None
        #self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        #if self.filter_empty_gt and \
        #        (example is None or
        #            ~(example['gt_labels_3d']._data != -1).any()):
        #    return None
        return example
    
    def prepare_test_data(self, index):
        """Prepare data for testing.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Testing data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        example = self.pipeline(input_dict)

        return example

    def __getitem__(self, idx):
        """Get item from infos according to the given index.

        Returns:
            dict: Data dictionary of the corresponding index.
        """
        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data
    
    def pre_pipeline(self, results):
        """Initialization before data preparation.

        Args:
            results (dict): Dict before data preprocessing.

                - img_fields (list): Image fields.
                - bbox3d_fields (list): 3D bounding boxes fields.
                - pts_mask_fields (list): Mask fields of points.
                - pts_seg_fields (list): Mask fields of point segments.
                - bbox_fields (list): Fields of bounding boxes.
                - mask_fields (list): Fields of masks.
                - seg_fields (list): Segment fields.
                - box_type_3d (str): 3D box type.
                - box_mode_3d (str): 3D box mode.
        """
        results['img_fields'] = []
        results['bbox3d_fields'] = []
        results['pts_mask_fields'] = []
        results['pts_seg_fields'] = []
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []
        results['box_type_3d'] = self.box_type_3d
        results['box_mode_3d'] = self.box_mode_3d
    
    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0. In 3D datasets, they are all the same, thus are all
        zeros.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)

    def evaluate(self, results, logger=None, **kwargs):
        assert len(results) == len(self.scenes)
        acc, intersect_acc, dist2set = [], [], []
        for i, res_dict in enumerate(results):
            data = self.__getitem__(i)
            points = data['points'].data
            y = data['gt_labels'].data
            pred = res_dict['pred']
            prob = res_dict['prob']
            acc.append((pred == y).float())

            intersect = (y == -1)
            dist = torch.zeros_like(intersect, dtype=torch.float) + 1e5
            if y.min() < y.max():
                for l1 in range(y.min(), y.max()+1):
                    p0 = points[y == l1].cuda()
                    p1 = points[y != l1].cuda()
                    indices = knn(1, p1.unsqueeze(0), p0.unsqueeze(0))
                    dists = (p1[indices[0, 0]] - p0).norm(p=2, dim=-1).cpu()
                    dist[y == l1] = dists
                    #mask = (dists < 0.2)
                    #intersect[y == l1] = mask
                #intersect_acc.append((pred == y)[intersect].float())
            dist2set.append(dist)
            
        acc = torch.cat(acc, dim=0)
        dist = torch.cat(dist2set, dim=0)
        ranges = torch.linspace(dist.min(), 0.5, 10)
        ranges = torch.cat([ranges, torch.zeros_like(ranges[0:1]) + 1e10])
        msg = f' acc={acc.mean().item():.4f}'
        for i, r in enumerate(ranges[1:]):
            l = ranges[i]
            acc_lr = acc[(dist <= r) & (dist >= l)].mean()
            ratio = ((dist <= r) & (dist >= l)).float().mean()
            msg += f', [{l:.3f}, {r:.3f}] ({ratio:.4f})={acc_lr:.4f}'
        #intersect_acc = torch.cat(intersect_acc, dim=0)
        acc = acc.mean().item()
        #intersect_acc = intersect_acc.mean().item()
        if logger is not None:
            logger.info(msg)
        else:
            print(msg)

        return dict(acc=acc)

    def __repr__(self):
        return f'{self.__class__.__name__}: scenes={len(self.scenes)}'
