_base_ = [
    '../_base_/datasets/waymo_cars_and_peds.py',
    '../_base_/models/dgcnn.py',
    '../_base_/schedules/cyclic_20e.py', '../_base_/default_runtime.py'
]

# data settings
data = dict(samples_per_gpu=16)
evaluation = dict(interval=20)

# runtime settings
checkpoint_config = dict(interval=10)
# PointNet2-MSG needs longer training time than PointNet2-SSG
runner = dict(type='EpochBasedRunner', max_epochs=250)

#resume_from = 'work_dirs/pointnet2_cars_and_peds/latest.pth'
