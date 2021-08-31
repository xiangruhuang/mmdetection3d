_base_ = [
    '../_base_/models/hv_pointpillars_secfpn_waymo.py',
    '../_base_/datasets/waymoD5-3d-3class.py',
    '../_base_/schedules/schedule_2x.py',
    '../_base_/default_runtime.py',
]
resume_from='work_dirs/hv_pointpillars_secfpn_sbn_2x16_2x_waymoD5-3d-3class/epoch_2.pth'

