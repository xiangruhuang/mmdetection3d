#!/bin/bash

pip3 uninstall -y numpy mmdet3d mmdet mmcv-full
pip3 install numpy

pip3 install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu112/torch1.9.0/index.html

pip3 install git+https://github.com/open-mmlab/mmdetection.git

python setup.py develop
