_base_ = [
    '../_base_/datasets/waymoD5-3d-3class.py',
    '../_base_/models/centerpoint_01voxel_second_secfpn_waymo.py',
    '../_base_/schedules/cyclic_20e_waymo_centerpoint.py', '../_base_/default_runtime.py'
]

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-75.2, -75.2, -2.0, 75.2, 75.2, 4.0]
# For nuScenes we usually do 10-class detection
class_names = [
    'Car', 'Pedestrian', 'Cyclist'
]

model = dict(
    pts_voxel_layer=dict(point_cloud_range=point_cloud_range),
    pts_bbox_head=dict(bbox_coder=dict(pc_range=point_cloud_range[:2])),
    # model training and testing settings
    train_cfg=dict(pts=dict(point_cloud_range=point_cloud_range)),
    test_cfg=dict(pts=dict(pc_range=point_cloud_range[:2])))

dataset_type = 'WaymoDataset'
data_root = 'data/waymo/kitti_format/'
file_client_args = dict(backend='disk')

db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'waymo_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(
            Car=5,
            Pedestrian=10,
            Cyclist=10)),
    classes=class_names,
    sample_groups=dict(
        Car=15,
        Pedestrian=10,
        Cyclist=10,
        ),
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
        load_dim=6,
        use_dim=5,
        file_client_args=file_client_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='ObjectSample', db_sampler=db_sampler),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
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
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=6,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1024, 1024),
        pts_scale_ratio=1,
        flip=True,
        pcd_horizontal_flip=True,
        pcd_vertical_flip=True,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D', sync_2d=False),
            dict(
                type='PointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ])
]
# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=6,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='DefaultFormatBundle3D',
        class_names=class_names,
        with_label=False),
    dict(type='Collect3D', keys=['points'])
]

data = dict(
    samples_per_gpu=3,
    workers_per_gpu=3,
    train=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + 'waymo_infos_train.pkl',
            pipeline=train_pipeline,
            classes=class_names,
            split='training',
            load_interval=1,
            test_mode=False,
            # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
            # and box_type_3d='Depth' in sunrgbd and scannet dataset.
            box_type_3d='LiDAR')),
    val=dict(
        pipeline=test_pipeline,
        classes=class_names,
        data_root=data_root,
        ann_file=data_root + 'waymo_infos_val.pkl',
        split='training',
        test_mode=True),
    test=dict(
        pipeline=test_pipeline,
        classes=class_names,
        data_root=data_root,
        ann_file=data_root + 'waymo_infos_val.pkl',
        test_mode=True,
        split='training'))

evaluation = dict(interval=30, pipeline=eval_pipeline)
