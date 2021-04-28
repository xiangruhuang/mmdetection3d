kitti:
	python -m tools.create_data kitti \
		--root-path ./data/kitti \
		--out-dir ./data/kitti \
		--extra-tag kitti

checkpoints/hv_second_secfpn_6x8_80e_kitti-3d-3class_20200620_230238-9208083a.pth:
	wget https://download.openmmlab.com/mmdetection3d/v0.1.0_models/second/hv_second_secfpn_6x8_80e_kitti-3d-3class/hv_second_secfpn_6x8_80e_kitti-3d-3class_20200620_230238-9208083a.pth \
		-P checkpoints

second.test: checkpoints/hv_second_secfpn_6x8_80e_kitti-3d-3class_20200620_230238-9208083a.pth
	python tools/test.py configs/second/hv_second_secfpn_6x8_80e_kitti-3d-3class.py $< --out results/second.pkl

