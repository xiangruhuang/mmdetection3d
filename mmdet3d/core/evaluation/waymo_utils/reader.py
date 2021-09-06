import numpy as np
import glob
from waymo_open_dataset.protos import metrics_pb2
import tensorflow as tf
from waymo_open_dataset import dataset_pb2 as open_dataset
from tqdm import tqdm

tfrecord_files = glob.glob(
    '/mnt/xrhuang/datasets/waymo/waymo_format/validation/*.tfrecord')

gt_bin_file = '/mnt/xrhuang/mmdetection3d/data/waymo/waymo_format/gt.bin'

byte=bytearray(open(gt_bin_file, 'rb').read())
objs=metrics_pb2.Objects()
objs.ParseFromString(byte)

context_names = []
timestamps = []
frames = []
for i, tf_file in enumerate(tfrecord_files):
    file_data = tf.data.TFRecordDataset(tf_file, compression_type='')
    for frame_num, frame_data in enumerate(tqdm(file_data)):
        frame = open_dataset.Frame()

        frame.ParseFromString(bytearray(frame_data.numpy()))
        #context_names.append(frame.context.name)
        #timestamps.append(frame.timestamp_micros)
        frames.append((frame.context.name, frame.timestamp_micros))

import ipdb; ipdb.set_trace()
frame_set = set(frames)

frame_dict = {}
selected_objs = metrics_pb2.Objects()

for i, obj in enumerate(objs.objects):
    ts = obj.frame_timestamp_micros
    name = obj.context_name
    if (name, ts) in frame_set:
        selected_objs.objects.append(obj)

import ipdb; ipdb.set_trace()
with open('selected.bin', 'wb') as fout:
    fout.write(selected_objs.SerializeToString())

