_base_ = ['./centerpoint_01voxel_second_secfpn_4x8_cyclic_20e_waymo.py']

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
voxel_size = [0.1, 0.1, 0.15]
point_cloud_range = [-75.2, -75.2, -2.0, 75.2, 75.2, 4.0]
# For nuScenes we usually do 10-class detection
class_names = [
    'Car', 'Pedestrian', 'Cyclist',
]

model = dict(
    pts_voxel_layer=dict(
        voxel_size=voxel_size, point_cloud_range=point_cloud_range),
    pts_middle_encoder=dict(sparse_shape=[41, 1536, 1536]),
    pts_bbox_head=dict(
        bbox_coder=dict(
            voxel_size=voxel_size[:2], pc_range=point_cloud_range[:2])),
    train_cfg=dict(
        pts=dict(
            grid_size=[1536, 1536, 40],
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range)),
    test_cfg=dict(
        pts=dict(voxel_size=voxel_size[:2], pc_range=point_cloud_range[:2])))

