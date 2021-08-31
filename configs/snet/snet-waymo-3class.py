_base_ = [
    '../_base_/models/snet.py',
    '../_base_/datasets/waymoD5-3d-3class-seg.py',
    '../_base_/schedules/cyclic_20e.py', '../_base_/default_runtime.py'
]

point_cloud_range = [-75.2, -75.2, 0.3, 75.2, 75.2, 4]

model = dict(
    # model training and testing settings
    train_cfg=dict(pts=dict(point_cloud_range=point_cloud_range)),
    test_cfg=dict(pts=dict(pc_range=point_cloud_range[:2], nms_type='circle'))
)

workflow = [('train', 1)]
runner = dict(type='EpochBasedRunner', max_epochs=80)
#resume_from = './work_dirs/centerpoint_ssl_01voxel_waymo_10percent/epoch_40.pth'
eval_options=dict(
    pklfile_prefix='./work_dirs/snet-waymo/')

