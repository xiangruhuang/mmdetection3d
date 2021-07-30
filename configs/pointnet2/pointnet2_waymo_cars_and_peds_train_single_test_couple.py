_base_ = [
    '../_base_/datasets/waymo_cars_and_peds_train_single_test_couple.py',
    '../_base_/models/pointnet2_seg.py',
    '../_base_/schedules/cyclic_20e.py', '../_base_/default_runtime.py'
]

# data settings
data = dict(samples_per_gpu=16)
evaluation = dict(interval=10)

# runtime settings
checkpoint_config = dict(interval=50)
log_config = dict(interval=1)
# PointNet2-MSG needs longer training time than PointNet2-SSG
runner = dict(type='EpochBasedRunner', max_epochs=1000)

#resume_from = 'work_dirs/pointnet2_cars_and_peds/latest.pth'
