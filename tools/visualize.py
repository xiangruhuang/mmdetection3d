import argparse
import polyscope as ps
import pickle
import os
import numpy as np

from mmcv import Config, DictAction
from mmdet3d.datasets import build_dataloader, build_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize Results')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('result', help='saved result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--logfile', type=str,
        help='place to holder evaluation results')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where results will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both specified, '
            '--options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args

def main():
    args = parse_args()
    ps.init()
    cfg = Config.fromfile(args.config)
    dataset = build_dataset(cfg.data.test)
    print(dataset.CLASSES)
    with open(args.result, 'rb') as fin:
        results = pickle.load(fin)
    R = np.array([[1, 0, 0], [0, 0, 1], [0,-1,0]])
    for i, data in enumerate(dataset):
        ps.remove_all_structures()
        scene = dataset.scenes[i]
        name=f'sample-{i}'
        if len(scene.keys()) <= 1:
            continue
        for k, val in scene.items():
            name += f'-{k}-{val}'
            dataset.samples[k][val]
            points = dataset.load_points(k, val).tensor.cpu().numpy()
            points = R.dot(points.T).T
            ptr = ps.register_point_cloud(f'{k}-{val}', points+np.array([10, 0, 10*dataset.cat2id[k]]))
            idx = dataset.cat2id[k] * 100 + val
            gt_labels = dataset[idx]['gt_labels'].data.cpu().numpy()
            pred = results[idx]['pred'].cpu().numpy()
            acc = (gt_labels == pred).astype(np.float32)
            ptr.add_scalar_quantity('acc', acc, enabled=True)
            
        points = data['points'].data.cpu().numpy()
        points = R.dot(points.T).T
        gt_labels = data['gt_labels'].data.cpu().numpy()
        pred = results[i]['pred'].cpu().numpy()
        acc = (gt_labels == pred).astype(np.float32)
        if acc.mean() > 1 - 1e-6:
            continue
        ptr = ps.register_point_cloud(f'sample-{i}', points)
        ptr.add_scalar_quantity('gt', gt_labels, enabled=True)
        ptr.add_scalar_quantity('acc', acc, enabled=True)
        ps.show()
        
if __name__ == '__main__':
    main()
