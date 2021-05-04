#!/bin/bash

#pip3 cache purge
pip3 uninstall -y numpy mmdet3d mmdet mmcv-full
pip3 install numpy==1.19.5

pip3 install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.1/index.html --no-cache-dir --no-binary

pip3 install git+https://github.com/open-mmlab/mmdetection.git --no-cache-dir --no-binary

pip3 install -v -e . 
