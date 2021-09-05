_base_ = [
    '../_base_/models/hv_pointpillars_secfpn_waymo.py',
    '../_base_/datasets/waymoD5-3d-3class-10percent.py',
    '../_base_/schedules/schedule_2x.py',
    '../_base_/default_runtime.py',
]

# data settings
data = dict(train=dict(load_interval=1),
	    val=dict(load_interval=1),
            test=dict(load_interval=1))

log_config = dict(interval=100)

resume_from = None
workflow = [('train', 1)]
