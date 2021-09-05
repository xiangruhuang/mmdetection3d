_base_ = [
    '../_base_/models/hv_pointpillars_secfpn_waymo.py',
    '../_base_/datasets/waymoD5-3d-3class-10percent.py',
    '../_base_/schedules/schedule_2x.py',
    '../_base_/default_runtime.py',
]

#resume_from='work_dirs/hv_pointpillars_secfpn_sbn_2x16_2x_waymoD5-3d-3class/epoch_2.pth'

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

workflow = [('train', 1), ('val', 1)]
