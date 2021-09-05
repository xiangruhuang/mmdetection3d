_base_ = [
    '../_base_/datasets/waymoD5-3d-3class-10percent.py',
    '../_base_/models/centerpoint_ssl_01voxel_waymo.py',
    '../_base_/schedules/cyclic_20e_waymo_centerpoint.py',
    #'../_base_/schedules/schedule_2x.py',
    '../_base_/default_runtime.py',
]

class_names = [
    'Car',
    'Pedestrian',
    'Cyclist',
]

point_cloud_range = [-75.2, -75.2, -2, 75.2, 75.2, 4]

model = dict(
    pts_voxel_layer=dict(point_cloud_range=point_cloud_range),
    pts_bbox_head=dict(bbox_coder=dict(pc_range=point_cloud_range[:2])),
    # model training and testing settings
    train_cfg=dict(pts=dict(point_cloud_range=point_cloud_range)),
    test_cfg=dict(pts=dict(pc_range=point_cloud_range[:2], nms_type='circle')))

workflow = [('train', 1), ('val', 1)]
runner = dict(type='EpochBasedRunner', max_epochs=24)
#resume_from = './work_dirs/centerpoint_ssl_01voxel_waymo_10percent/epoch_45.pth'
eval_options=dict(
    pklfile_prefix='./work_dirs/centerpoint_ssl_01voxel_waymo_10percent')

data_root = 'data/waymo/kitti_format/'
data=dict(
    train=dict(dataset=dict(
        load_interval=150,
        ann_file=data_root + 'waymo_infos_subtrain.pkl')),
    val=dict(
        load_interval=150,
        ann_file=data_root + 'waymo_infos_subtrain.pkl'),
    test=dict(
        load_interval=150,
        ann_file=data_root + 'waymo_infos_subtrain.pkl'))
