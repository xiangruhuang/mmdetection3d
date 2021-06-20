import pickle
from mmdet3d.datasets import build_dataset
from mmdet.datasets import build_dataloader
import numpy as np
import sys
    
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

dataset_type = 'NuScenesDataset'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')


db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'nuscenes_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(
            car=5,
            truck=5,
            bus=5,
            trailer=5,
            construction_vehicle=5,
            traffic_cone=5,
            barrier=5,
            motorcycle=5,
            bicycle=5,
            pedestrian=5)),
    classes=class_names,
    sample_groups=dict(
        car=2,
        truck=3,
        construction_vehicle=7,
        bus=4,
        trailer=6,
        barrier=2,
        motorcycle=6,
        bicycle=6,
        pedestrian=2,
        traffic_cone=2),
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args))

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args,
        pad_empty_sweeps=True,
        remove_close=True),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='ObjectSample', db_sampler=db_sampler),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0]),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    #dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]

#data = {'type': 'CBGSDataset', 'data_root': 'data/nuscenes/', 
#        'ann_file': 'data/nuscenes/nuscenes_infos_train.pkl', 
#        'pipeline': [{'type': 'LoadPointsFromFile', 'coord_type': 'LIDAR', 'load_dim': 5, 'use_dim': 5, 
#        'file_client_args': {'backend': 'disk'}}, 
#        {'type': 'LoadPointsFromMultiSweeps', 'sweeps_num': 10, 'file_client_args': {'backend': 'disk'}}, {'type': 'LoadAnnotations3D', 'with_bbox_3d': True, 'with_label_3d': True}, 
#        {'type': 'GlobalRotScaleTrans', 'rot_range': [-0.3925, 0.3925], 'scale_ratio_range': [0.95, 1.05], 'translation_std': [0, 0, 0]}, 
#        {'type': 'RandomFlip3D', 'flip_ratio_bev_horizontal': 0.5}, 
#        {'type': 'PointsRangeFilter', 'point_cloud_range': [-50, -50, -5, 50, 50, 3]}, 
#        {'type': 'ObjectRangeFilter', 'point_cloud_range': [-50, -50, -5, 50, 50, 3]},
#        {'type': 'ObjectNameFilter', 'classes': ['car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier']}, 
#        {'type': 'PointShuffle'}, 
#        {'type': 'DefaultFormatBundle3D', 'class_names': ['car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier']}, {'type': 'Collect3D', 'keys': ['points', 'gt_bboxes_3d', 'gt_labels_3d']}], 
#        'classes': ['car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'], 'modality': {'use_lidar': True, 'use_camera': False, 'use_radar': False, 'use_map': False, 'use_external': False}, 'test_mode': False, 'box_type_3d': 'LiDAR', 'dataset': {'type': 'NuScenesDataset', 'data_root': 'data/nuscenes/', 'ann_file': 'data/nuscenes/nuscenes_infos_train.pkl', 'pipeline': [{'type': 'LoadPointsFromFile', 'coord_type': 'LIDAR', 'load_dim': 5, 'use_dim': 5, 'file_client_args': {'backend': 'disk'}}, {'type': 'LoadPointsFromMultiSweeps', 'sweeps_num': 9, 'use_dim': [0, 1, 2, 3, 4], 'file_client_args': {'backend': 'disk'}, 'pad_empty_sweeps': True, 'remove_close': True}, {'type': 'LoadAnnotations3D', 'with_bbox_3d': True, 'with_label_3d': True}, {'type': 'ObjectSample', 'db_sampler': {'data_root': 'data/nuscenes/', 'info_path': 'data/nuscenes/nuscenes_dbinfos_train.pkl', 'rate': 1.0, 'prepare': {'filter_by_difficulty': [-1], 'filter_by_min_points': {'car': 5, 'truck': 5, 'bus': 5, 'trailer': 5, 'construction_vehicle': 5, 'traffic_cone': 5, 'barrier': 5, 'motorcycle': 5, 'bicycle': 5, 'pedestrian': 5}}, 'classes': ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'], 'sample_groups': {'car': 2, 'truck': 3, 'construction_vehicle': 7, 'bus': 4, 'trailer': 6, 'barrier': 2, 'motorcycle': 6, 'bicycle': 6, 'pedestrian': 2, 'traffic_cone': 2}, 'points_loader': {'type': 'LoadPointsFromFile', 'coord_type': 'LIDAR', 'load_dim': 5, 'use_dim': [0, 1, 2, 3, 4], 'file_client_args': {'backend': 'disk'}}, 'type': 'DataBaseSampler'}}, {'type': 'GlobalRotScaleTrans', 'rot_range': [-0.3925, 0.3925], 'scale_ratio_range': [0.95, 1.05], 'translation_std': [0, 0, 0]}, {'type': 'RandomFlip3D', 'sync_2d': False, 'flip_ratio_bev_horizontal': 0.5, 'flip_ratio_bev_vertical': 0.5}, {'type': 'PointsRangeFilter', 'point_cloud_range': [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]}, {'type': 'ObjectRangeFilter', 'point_cloud_range': [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]}, {'type': 'ObjectNameFilter', 'classes': ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']}, {'type': 'PointShuffle'}, {'type': 'DefaultFormatBundle3D', 'class_names': ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']}, {'type': 'Collect3D', 'keys': ['points', 'gt_bboxes_3d', 'gt_labels_3d']}], 'classes': ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'], 'load_interval': 1, 'test_mode': False, 'use_valid_flag': True, 'box_type_3d': 'LiDAR'}}

train=dict(
    type=dataset_type,
    data_root=data_root,
    ann_file=data_root + 'nuscenes_infos_train.pkl',
    pipeline=train_pipeline,
    classes=class_names,
    load_interval=1,
    test_mode=False,
    use_valid_flag=True,
    # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
    # and box_type_3d='Depth' in sunrgbd and scannet dataset.
    box_type_3d='LiDAR')

dataset = build_dataset(train)

#data_loader = build_dataloader(dataset, 1, 3, 1, dist=False, seed=816)
#
#import ipdb; ipdb.set_trace()
#for data in data_loader:
#    print(data)

print(len(dataset))
i = 0
while i < len(dataset):
    i = np.random.randint(len(dataset))
    print(f'showing data sample {i}')
    if dataset.show_data(i, 'visualization/nuscenes/'):
        i += 10
    else:
        i += 1
