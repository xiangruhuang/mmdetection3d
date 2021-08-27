_base_ = [
    '../_base_/datasets/waymoD5-3d-3class-ssl-withmotion.py',
    '../_base_/models/centerpoint_ssl_01voxel_waymo.py',
    '../_base_/schedules/cyclic_20e.py', '../_base_/default_runtime.py'
]

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
#point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
# For nuScenes we usually do 10-class detection
class_names = [
    'Car',
    'Pedestrian',
    'Cyclist',
]

point_cloud_range = [-74.88, -74.88, 0.3, 74.88, 74.88, 4]

model = dict(
    pts_voxel_layer=dict(point_cloud_range=point_cloud_range),
    pts_bbox_head=dict(bbox_coder=dict(pc_range=point_cloud_range[:2])),
    # model training and testing settings
    train_cfg=dict(pts=dict(point_cloud_range=point_cloud_range)),
    test_cfg=dict(pts=dict(pc_range=point_cloud_range[:2], nms_type='circle')),
    ssl_mlps=dict(ssl_channels=[
        ([5, 128, 2], 0.0),
        ([16, 128, 2], 0.5),
        ([32, 128, 2], 0.5),
        ([64, 128, 2], 0.5),
        ([128, 128, 2], 0.5),
        ([128, 128, 2], 0.5),
        ])
    )

workflow = [('train', 1)]
runner = dict(type='EpochBasedRunner', max_epochs=40)
#resume_from = './work_dirs/centerpoint_ssl_01voxel_waymo_10percent/latest.pth'
eval_options=dict(prklfile_prefix='./work_dirs/centerpoint_ssl_01voxel_waymo_10percent')

