# dataset settings
# D5 in the config name means the whole dataset is divided into 5 folds
# We only use one fold for efficient experiments
dataset_type = 'WaymoGTDataset'
data_root = 'data/waymo/kitti_format/'
file_client_args = dict(backend='disk')
# Uncomment the following if use ceph or other file clients.
# See https://mmcv.readthedocs.io/en/latest/api.html#mmcv.fileio.FileClient
# for more details.
# file_client_args = dict(
#     backend='petrel', path_mapping=dict(data='s3://waymo_data/'))

class_names = ['Car', 'Pedestrian'] #, 'Cyclist']
point_cloud_range = [-74.88, -74.88, -2, 74.88, 74.88, 4]
input_modality = dict(use_lidar=True, use_camera=False)
db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'waymo_dbinfos_train.pkl',
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
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_labels'])
]
test_pipeline = [
    dict(
        type='DefaultFormatBundle3D',
        class_names=class_names,
        with_label=False),
    dict(type='Collect3D', keys=['points', 'gt_labels']),
]
# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
eval_pipeline = [
    dict(
        type='DefaultFormatBundle3D',
        class_names=class_names,
        with_label=False),
    dict(type='Collect3D', keys=['points', 'gt_labels'])
]

shared_args=dict(
    classes=class_names,
    db_sampler=db_sampler,
    gap=0.1,
    visualize=False,
    train_interval=8,
    filter_by_points=dict(Car=30, Pedestrian=30),
    maximum_samples=dict(Car=200, Pedestrian=200),
)
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=5,
    train=dict(
        type=dataset_type,
        split='training',
        **shared_args),
    val=dict(
        type=dataset_type,
        split='testing',
        **shared_args),
    test=dict(
        type=dataset_type,
        split='testing',
        **shared_args),

#evaluation = dict(interval=24, pipeline=eval_pipeline)
