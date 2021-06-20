import copy
import numpy as np
import torch
import polyscope as ps

try:
    import open3d as o3d
    from open3d import geometry
except ImportError:
    raise ImportError(
        'Please run "pip install open3d" to install open3d first.')

class Visualizer(object):
    r"""Online visualizer implemented with PolyScope.

    Args:
        points (numpy.array, shape=[N, 3+C]): Points to visualize. The Points
            cloud is in mode of Coord3DMode.DEPTH (please refer to
            core.structures.coord_3d_mode).
        bbox3d (numpy.array, shape=[M, 7]): 3d bbox (x, y, z, dx, dy, dz, yaw)
            to visualize. The 3d bbox is in mode of Box3DMode.DEPTH with
            gravity_center (please refer to core.structures.box_3d_mode).
            Default: None.
        save_path (str): path to save visualized results. Default: None.
        points_size (int): the size of points to show on visualizer.
            Default: 2.
        point_color (tuple[float]): the color of points.
            Default: (0.5, 0.5, 0.5).
        bbox_color (tuple[float]): the color of bbox. Default: (0, 1, 0).
        points_in_box_color (tuple[float]):
            the color of points which are in bbox3d. Default: (1, 0, 0).
        rot_axis (int): rotation axis of bbox. Default: 2.
        center_mode (bool): indicate the center of bbox is bottom center
            or gravity center. avaliable mode
            ['lidar_bottom', 'camera_bottom']. Default: 'lidar_bottom'.
        mode (str):  indicate type of the input points, avaliable mode
            ['xyz', 'xyzrgb']. Default: 'xyz'.
    """

    def __init__(self,
                 points,
                 bbox3d=None,
                 save_path=None,
                 radius=0.0001,
                 point_color=(0.5, 0.5, 0.5),
                 bbox_color=(0, 1, 0),
                 points_in_box_color=(1, 0, 0),
                 rot_axis=2,
                 center_mode='lidar_bottom',
                 mode='xyz'):
        super(Visualizer, self).__init__()
        assert 0 <= rot_axis <= 2

        self.class2color = {
                'pedestrian': (1,0,0),
                'car': (0,0,1),
                'truck': (0,0,1),
                'bus': (0,0,1),
                'trailer': (0.0,0,0),
                'barrier': (0.0,0,0.0),
                'motorcycle': (0, 1, 0),
                'bicycle': (0,1,0),
                'traffic_cone': (0,0,0),
                'construction_vehicle': (0,0,0),
                'background': (0.5, 0.5, 0.5),
                'Unknown': (0, 0, 0),
            }
        self.points_by_class = {c: None for c in self.class2color.keys()}

        points = points[:, :3]
        ps.init()
        ps.remove_all_structures()
        self.bbox_count = 0

        self.radius = radius
        self.point_color = point_color
        self.bbox_color = bbox_color
        self.points_in_box_color = points_in_box_color
        self.rot_axis = rot_axis
        self.center_mode = center_mode
        self.mode = mode
        self.seg_num = 0

        if isinstance(points, torch.Tensor):
            points = points.numpy()
        # draw points
        self.bg_points = points
        self.draw_points(points, 'background',
            point_color=self.point_color, radius=self.radius)

        # draw boxes
        if bbox3d is not None:
            self.draw_bboxes(bbox3d,
                bbox_color, points_in_box_color, rot_axis,
                center_mode, mode)

    def draw_data(self, gt_names, gt_bboxes, pred_bboxes=None):
        self.add_bboxes(bbox3d=gt_bboxes,
            bbox_color=(0, 0, 1), cls_names=gt_names)

    def draw_points(self, points, cls_name,
                    point_color=(0.5, 0.5, 0.5),
                    radius=0.0001):
        """Draw points on visualizer.

        Args:
            points (numpy.array | torch.tensor, shape=[N, 3+C]):
                points to visualize.
            cls_name (str): name of the class.
            point_color (tuple[float]) : color of points.
                Default: (0.5, 0.5, 0.5).
            radius (float): the size of points to show on visualizer.
                Default: 0.0001.

        Returns:
            ps_pcd: polyscope point cloud interactive object.
        """
        if isinstance(points, torch.Tensor):
            points = points.cpu().numpy()

        point_color = (0.5, 0.5, 0.5)
        if self.points_by_class.get(cls_name, None) is None:
            self.points_by_class[cls_name] = points
        else:
            self.points_by_class[cls_name] = np.concatenate(
                [self.points_by_class[cls_name], points], axis=0)
        ps_pcd = ps.register_point_cloud(
            cls_name, self.points_by_class[cls_name], radius=radius,
            color=self.class2color[cls_name] if 
                cls_name in self.class2color.keys()
                else self.class2color['Unknown'])

        return ps_pcd
    
    def draw_bboxes(self, bbox3d,
                    bbox_color=(0, 1, 0),
                    points_in_box_color=(1, 0, 0),
                    rot_axis=2,
                    center_mode='lidar_bottom',
                    cls_names=None):
        """Draw bbox on visualizer and change the color of points inside bbox3d.

        Args:
            bbox3d (numpy.array | torch.tensor, shape=[M, 7]):
                3d bbox (x, y, z, dx, dy, dz, yaw) to visualize.
            points_colors (numpy.array): color of each points.
            pcd (:obj:`open3d.geometry.PointCloud`): point cloud. Default: None.
            bbox_color (tuple[float]): the color of bbox. Default: (0, 1, 0).
            points_in_box_color (tuple[float], or list[tuple[float]]):
                the color of points inside each bbox3d. Default: (1, 0, 0).
            rot_axis (int): rotation axis of bbox. Default: 2.
            center_mode (bool): indicate the center of bbox is bottom center
                or gravity center. avaliable mode
                ['lidar_bottom', 'camera_bottom']. Default: 'lidar_bottom'.
        """
        if isinstance(bbox3d, torch.Tensor):
            bbox3d = bbox3d.cpu().numpy()
        bbox3d = bbox3d.copy()

        for i in range(len(bbox3d)):
            if isinstance(points_in_box_color, list):
                in_box_color = np.array(points_in_box_color[i])
            else:
                in_box_color = np.array(points_in_box_color)
            center = bbox3d[i, 0:3]
            dim = bbox3d[i, 3:6]
            yaw = np.zeros(3)
            yaw[rot_axis] = -bbox3d[i, 6]
            rot_mat = geometry.get_rotation_matrix_from_xyz(yaw)

            if center_mode == 'lidar_bottom':
                center[rot_axis] += dim[
                    rot_axis] / 2  # bottom center to gravity center
            elif center_mode == 'camera_bottom':
                center[rot_axis] -= dim[
                    rot_axis] / 2  # bottom center to gravity center
            box3d = geometry.OrientedBoundingBox(center, rot_mat, dim)

            line_set = geometry.LineSet.create_from_oriented_bounding_box(box3d)
            cls_name = cls_names[i]
            ps_line_set = ps.register_curve_network(
                    f'box{self.bbox_count}-{cls_names[i]}',
                    np.array(line_set.points),
                    np.array(line_set.lines),
                    radius=0.0003,
                    color=np.array(bbox_color)
                    )
            self.bbox_count += 1

            # change the color of points which are in box
            indices = box3d.get_point_indices_within_bounding_box(
                o3d.utility.Vector3dVector(self.bg_points)
            )
            points_in_box = self.bg_points[indices]
            self.draw_points(points_in_box, cls_name)
            self.bg_points = np.delete(self.bg_points, indices, axis=0)
            ps.register_point_cloud('background', self.bg_points)

    def add_bboxes(self, bbox3d, cls_names, bbox_color=None, points_in_box_color=None):
        """Add bounding box to visualizer.

        Args:
            bbox3d (numpy.array, shape=[M, 7]):
                3D bbox (x, y, z, dx, dy, dz, yaw) to be visualized.
                The 3d bbox is in mode of Box3DMode.DEPTH with
                gravity_center (please refer to core.structures.box_3d_mode).
            bbox_color (tuple[float]): the color of bbox. Defaule: None.
            points_in_box_color (tuple[float]): the color of points which
                are in bbox3d. Defaule: None.
            cls_name (str): the name of the class
        """
        if bbox_color is None:
            bbox_color = self.bbox_color

        points_in_box_color = []
        for cls_name in cls_names:
            points_in_box_color.append(self.class2color.get(cls_name, self.class2color['Unknown']))

        self.draw_bboxes(
            bbox3d, bbox_color, points_in_box_color, self.rot_axis,
            self.center_mode, cls_names
        )

    def show(self, save_path=None):
        """Visualize the points cloud.

        """

        ps.show()
