_base_ = './centerpoint_0075voxel_second_secfpn_dcn_' \
         'circlenms_4x8_cyclic_20e_waymo.py'

point_cloud_range = [-75.2, -75.2, -2.0, 75.2, 75.2, 4.0]
file_client_args = dict(backend='disk')
class_names = [
    'Car', 'Pedestrian', 'Cyclist'
]

dataset_type = 'WaymoDataset'
data_root = 'data/waymo/kitti_format/'
file_client_args = dict(backend='disk')
db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'waymo_dbinfos_subtrain.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(Car=5, Pedestrian=10, Cyclist=10)),
    classes=class_names,
    sample_groups=dict(Car=15, Pedestrian=10, Cyclist=10),
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
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        file_client_args=file_client_args),
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
    #dict(
    #    type='LoadPointsFromMultiSweeps',
    #    sweeps_num=9,
    #    use_dim=[0, 1, 2, 3, 4],
    #    file_client_args=file_client_args,
    #    pad_empty_sweeps=True,
    #    remove_close=True),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1536, 1536),
        pts_scale_ratio=1,
        # Add double-flip augmentation
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

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + 'waymo_infos_subtrain.pkl',
            pipeline=train_pipeline,
            load_interval=1)),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline)
)

evaluation = dict(interval=25)

workflow=[('train', 1)]
