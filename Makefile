kitti:
	python -m tools.create_data kitti \
		--root-path ./data/kitti \
		--out-dir ./data/kitti \
		--extra-tag kitti

nuscenes:
	python -m tools.create_data nuscenes \
		--root-path ./data/nuscenes \
		--out-dir ./data/nuscenes \
		--extra-tag nuscenes

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

checkpoints/hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d_20200620_230725-0817d270.pth:
	wget https://download.openmmlab.com/mmdetection3d/v0.1.0_models/pointpillars/hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d/hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d_20200620_230725-0817d270.pth -P checkpoints

results/kitti/second.pkl: checkpoints/hv_second_secfpn_6x8_80e_kitti-3d-3class_20200620_230238-9208083a.pth results/
	CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/second/hv_second_secfpn_6x8_80e_kitti-3d-3class.py $< --out $@ --eval mAP --logfile second-eval.pkl

results/kitti/pointpillars.pkl: checkpoints/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20200620_230421-aa0f3adb.pth results
	CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class.py $< --out $@ --eval mAP --logfile pointpillars-eval.pkl

results/kitti/parta2.pkl: checkpoints/hv_PartA2_secfpn_2x8_cyclic_80e_kitti-3d-3class_20200620_230724-a2672098.pth results
	CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/parta2/hv_PartA2_secfpn_2x8_cyclic_80e_kitti-3d-3class.py $< --out $@ 

results/kitti/dynamic_voxelization.pkl: checkpoints/dv_second_secfpn_2x8_cosine_80e_kitti-3d-3class_20200620_231010-6aa607d3.pth results
	CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/dynamic_voxelization/dv_second_secfpn_2x8_cosine_80e_kitti-3d-3class.py $< --out $@

results/kitti/mvx_net.pkl: checkpoints/dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class_20200621_003904-10140f2d.pth results
	CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/mvxnet/dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class.py $< --out $@

results/nuscenes/pointpillars.pkl: checkpoints/hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d_20200620_230725-0817d270.pth results
	CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/pointpillars/hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d.py $< --out $@


second.visualize: results/kitti/second.pkl
	mkdir -p visualization/second/
	python tools/misc/visualize_results.py configs/second/hv_second_secfpn_6x8_80e_kitti-3d-3class.py --result $< --show-dir visualization/second/

pointpillars.visualize: results/kitti/pointpillars.pkl
	mkdir -p visualization/pointpillars/
	python tools/misc/visualize_results.py configs/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class.py --result $< --show-dir visualization/pointpillars/

parta2.visualize: results/kitti/parta2.pkl
	mkdir -p visualization/parta2/
	python tools/misc/visualize_results.py configs/parta2/hv_PartA2_secfpn_2x8_cyclic_80e_kitti-3d-3class.py --result $< --show-dir visualization/parta2/

dynamic_voxelization.visualize: results/kitti/dynamic_voxelization.pkl
	mkdir -p visualization/dynamic_voxelization/
	python tools/misc/visualize_results.py configs/dynamic_voxelization/dv_second_secfpn_2x8_cosine_80e_kitti-3d-3class.py --result $< --show-dir visualization/dynamic_voxelization/

mvx_net.visualize: results/kitti/mvx_net.pkl
	mkdir -p visualization/mvx_net/
	python tools/misc/visualize_results.py configs/mvxnet/dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class.py --result $< --show-dir visualization/mvx_net/

kitti.extract_objects: checkpoints/hv_second_secfpn_6x8_80e_kitti-3d-3class_20200620_230238-9208083a.pth
	CUDA_VISIBLE_DEVICES=7 python tools/extract_human_scans.py configs/second/hv_second_secfpn_6x8_80e_kitti-3d-3class.py $<
