kitti:
	python -m tools.create_data kitti \
		--root-path ./data/kitti \
		--out-dir ./data/kitti \
		--extra-tag kitti

checkpoints/hv_second_secfpn_6x8_80e_kitti-3d-3class_20200620_230238-9208083a.pth:
	wget https://download.openmmlab.com/mmdetection3d/v0.1.0_models/second/hv_second_secfpn_6x8_80e_kitti-3d-3class/hv_second_secfpn_6x8_80e_kitti-3d-3class_20200620_230238-9208083a.pth -P checkpoints

checkpoints/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20200620_230421-aa0f3adb.pth:
	wget https://download.openmmlab.com/mmdetection3d/v0.1.0_models/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20200620_230421-aa0f3adb.pth -P checkpoints

checkpoints/hv_PartA2_secfpn_2x8_cyclic_80e_kitti-3d-3class_20200620_230724-a2672098.pth:
	wget https://download.openmmlab.com/mmdetection3d/v0.1.0_models/parta2/hv_PartA2_secfpn_2x8_cyclic_80e_kitti-3d-3class/hv_PartA2_secfpn_2x8_cyclic_80e_kitti-3d-3class_20200620_230724-a2672098.pth -P checkpoints

checkpoints/dv_second_secfpn_2x8_cosine_80e_kitti-3d-3class_20200620_231010-6aa607d3.pth:
	wget https://download.openmmlab.com/mmdetection3d/v0.1.0_models/dynamic_voxelization/dv_second_secfpn_2x8_cosine_80e_kitti-3d-3class/dv_second_secfpn_2x8_cosine_80e_kitti-3d-3class_20200620_231010-6aa607d3.pth -P checkpoints

checkpoints/dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class_20200621_003904-10140f2d.pth:
	wget https://download.openmmlab.com/mmdetection3d/v0.1.0_models/mvxnet/dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class/dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class_20200621_003904-10140f2d.pth -P checkpoints

second.test: checkpoints/hv_second_secfpn_6x8_80e_kitti-3d-3class_20200620_230238-9208083a.pth
	mkdir -p results
	CUDA_VISIBLE_DEVICES=7 python tools/test.py configs/second/hv_second_secfpn_6x8_80e_kitti-3d-3class.py $< --out results/second.pkl --eval mAP

pointpillars.test: checkpoints/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20200620_230421-aa0f3adb.pth
	mkdir -p results
	python tools/test.py configs/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class.py $< --out results/pointpillars.pkl

parta2.test: checkpoints/hv_PartA2_secfpn_2x8_cyclic_80e_kitti-3d-3class_20200620_230724-a2672098.pth
	mkdir -p results
	CUDA_VISIBLE_DEVICES=2 python tools/test.py configs/parta2/hv_PartA2_secfpn_2x8_cyclic_80e_kitti-3d-3class.py $< --out results/parta2.pkl

dynamic_voxelization.test: checkpoints/dv_second_secfpn_2x8_cosine_80e_kitti-3d-3class_20200620_231010-6aa607d3.pth
	mkdir -p results
	CUDA_VISIBLE_DEVICES=1 python tools/test.py configs/dynamic_voxelization/dv_second_secfpn_2x8_cosine_80e_kitti-3d-3class.py $< --out results/dynamic_voxelization.pkl

mvxnet.test: checkpoints/dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class_20200621_003904-10140f2d.pth
	mkdir -p results
	CUDA_VISIBLE_DEVICES=1 python tools/test.py configs/mvxnet/dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class.py $< --out results/mvxnet.pkl

kitti.extract_objects: checkpoints/hv_second_secfpn_6x8_80e_kitti-3d-3class_20200620_230238-9208083a.pth
	CUDA_VISIBLE_DEVICES=7 python tools/extract_human_scans.py configs/second/hv_second_secfpn_6x8_80e_kitti-3d-3class.py $<
