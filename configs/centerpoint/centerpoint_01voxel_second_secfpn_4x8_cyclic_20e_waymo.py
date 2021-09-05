_base_ = [
    '../_base_/datasets/waymoD5-3d-3class.py',
    '../_base_/models/centerpoint_01voxel_second_secfpn_waymo.py',
    '../_base_/schedules/cyclic_20e.py', '../_base_/default_runtime.py'
]

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-75.2, -75.2, -2.0, 75.2, 75.2, 4.0]
# For nuScenes we usually do 10-class detection
class_names = [
    'Car', 'Pedestrian', 'Cyclist',
]

model = dict(
    pts_voxel_layer=dict(point_cloud_range=point_cloud_range),
    pts_bbox_head=dict(bbox_coder=dict(pc_range=point_cloud_range[:2])),
    # model training and testing settings
    train_cfg=dict(pts=dict(point_cloud_range=point_cloud_range)),
    test_cfg=dict(pts=dict(pc_range=point_cloud_range[:2])))

dataset_type = 'WaymoDataset'
data_root = 'data/waymo/kitti_format'
file_client_args = dict(backend='disk')
workflow = [('train', 1), ('test', 1)]
