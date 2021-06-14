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

gpu=0
gpus=0
epoch=10

checkpoints/%.pkl:
	make -C checkpoints $(notdir $@)

results/kitti/%.pkl: 
	make -C results/kitti $(notdir $@)

results/nus/%.pkl: 
	make -C results/nus $(notdir $@)

pointpillars.kitti.train:
	mkdir -p checkpoints/pointpillars-kitti/
	CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class.py --work-dir checkpoints/pointpillars-kitti/

snet.train:
	mkdir -p checkpoints/snet-kitti/
	CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/snet/snet-kitti-3d-3class.py --work-dir checkpoints/snet-kitti/

3dssd.train:
	CUDA_VISIBLE_DEVICES=$(gpus) ./tools/dist_train.sh configs/3dssd/3dssd_nus-3d.py 4
	#CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh configs/3dssd/3dssd_kitti-3d-3class.py 4
	#CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/3dssd/3dssd_nus-3d.py --work-dir 3dssd_nus-3d/

3dssd.test:
	#CUDA_VISIBLE_DEVICES=$(gpu) python tools/test.py configs/3dssd/3dssd_kitti-3d-3class.py work_dirs/3dssd_kitti-3d-3class/epoch_6.pth --eval mAP --out work_dirs/3dssd_kitti-3d-3class/eval.pkl
	#CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_test.sh configs/3dssd/3dssd_kitti-3d-car.py 4 --no-validate
	CUDA_VISIBLE_DEVICES=$(gpus) ./tools/dist_test.sh configs/3dssd/3dssd_kitti-3d-3class.py work_dirs/3dssd_kitti-3d-3class/epoch_150.pth 4 --eval mAP --out work_dirs/3dssd_kitti-3d-3class/eval.pkl

centerpoint-voxel.train:
	mkdir -p checkpoints/centerpoint-voxel
	CUDA_VISIBLE_DEVICES=$(gpu) python tools/train.py configs/centerpoint/centerpoint_01voxel_second_secfpn_4x8_cyclic_20e_nus.py --work-dir checkpoints/centerpoint-voxel/

second.visualize: 
	mkdir -p visualization/second/
	python tools/misc/visualize_results.py configs/second/hv_second_secfpn_6x8_80e_kitti-3d-3class.py --result results/kitti/second.pkl --show-dir visualization/second/

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
	python tools/extract_human_scans.py configs/second/hv_second_secfpn_6x8_80e_kitti-3d-3class.py $<
