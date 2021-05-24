second.pkl:
	make -C ../../checkpoints hv_second_secfpn_6x8_80e_kitti-3d-3class_20200620_230238-9208083a.pth
	cd ../.. && \
	CUDA_VISIBLE_DEVICES=0 python tools/test.py \
	configs/second/hv_second_secfpn_6x8_80e_kitti-3d-3class.py \
	checkpoints/hv_second_secfpn_6x8_80e_kitti-3d-3class_20200620_230238-9208083a.pth \
	--out $@ --eval mAP --logfile second-eval.pkl

pointpillars.pkl:
	make -C ../../checkpoints hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20200620_230421-aa0f3adb.pth
	cd ../.. && \
	CUDA_VISIBLE_DEVICES=0 python tools/test.py \
	configs/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class.py \
	checkpoints/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20200620_230421-aa0f3adb.pth \
 	--out $@ --eval mAP --logfile pointpillars-eval.pkl

parta2.pkl:
	make -C ../../checkpoints hv_PartA2_secfpn_2x8_cyclic_80e_kitti-3d-3class_20200620_230724-a2672098.pth
	cd ../.. && \
	CUDA_VISIBLE_DEVICES=0 python tools/test.py \
	configs/parta2/hv_PartA2_secfpn_2x8_cyclic_80e_kitti-3d-3class.py \
	checkpoints/hv_PartA2_secfpn_2x8_cyclic_80e_kitti-3d-3class_20200620_230724-a2672098.pth \
 	--out $@ --eval mAP --logfile parta2-eval.pkl

dynamic_voxelization.pkl: checkpoints/dv_second_secfpn_2x8_cosine_80e_kitti-3d-3class_20200620_231010-6aa607d3.pth results
	CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/dynamic_voxelization/dv_second_secfpn_2x8_cosine_80e_kitti-3d-3class.py $< --out $@

mvx_net.pkl: checkpoints/dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class_20200621_003904-10140f2d.pth results
	CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/mvxnet/dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class.py $< --out $@

results/nuscenes/pointpillars.pkl: checkpoints/hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d_20200620_230725-0817d270.pth results
	CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/pointpillars/hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d.py $< --out $@
