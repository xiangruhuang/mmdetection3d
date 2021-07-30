_base_ = [
    '../_base_/datasets/waymo_cars_and_peds.py',
    '../_base_/models/pointnet2_seg.py',
    '../_base_/schedules/cyclic_20e.py', '../_base_/default_runtime.py'
]

# data settings
data = dict(samples_per_gpu=16)
evaluation = dict(interval=50)

# runtime settings
checkpoint_config = dict(interval=50)
# PointNet2-MSG needs longer training time than PointNet2-SSG
runner = dict(type='EpochBasedRunner', max_epochs=500)

#resume_from = 'work_dirs/pointnet2_waymo_cars_and_peds_gap01/latest.pth'

shared_args=dict(
    gap=0.1,
    train_interval=97,
    maximum_samples=dict(Car=1000, Pedestrian=1000),
)
data = dict(
    train=dict(
        load_interval=1,
        **shared_args),
    val=dict(
        load_interval=97,
        **shared_args),
    test=dict(
        load_interval=97,
        **shared_args)
    )
