import open3d as o3d
import numpy as np
import sys, os
project_path=os.path.abspath(__file__)
project_path=os.path.dirname(project_path)
project_path=os.path.dirname(project_path)
sys.path.append(project_path)
import linalg
import geometry.util as geo_util

class PinholeCamera:
  def __init__(self, extrinsic=None):
    self.vis = o3d.visualization.Visualizer()
    self.vis.create_window('ply', 320, 240, # width, height
                                  50, 50, False)
    self.T = np.eye(4)

  def transform(self, transformation):
    self.T = transformation.dot(self.T)
    #ctr = self.vis.get_view_control()
    #param = ctr.convert_to_pinhole_camera_parameters()
    #param.extrinsic = transformation.dot(param.extrinsic)
    #ctr.convert_from_pinhole_camera_parameters(param)
    #ctr.rotate(10.0, 0.0)

  """ Project Triangle Mesh to depth image.
  Input:
    mesh: o3d.geometry.TriangleMesh object
    intersecting_triangles: if True, we are also collecting correspondences.
  Output:
    depth: np.ndarray of shape [W, H].
  Additional Output (if intersecting_triangles=True):
    points3d: np.ndarray of shape [M, 3].
              point cloud in 3D, M is number of valid pixels/points.
    correspondences: np.ndarray of shape [M] and dtype np.int32
                     each entry lies in range [-1, N],
                     -1 indicates invalid.
    valid_pixel_indices: np.ndarray of shape [M, 2],
                         each row contains a valid pixel coordinates.
  """
  def project(self, mesh, intersecting_triangles=False):
    mesh.transform(self.T)
    import time
    t1 = time.time()
    self.vis.add_geometry(mesh)
    depth = self.vis.capture_depth_float_buffer(True)
    depth = np.array(depth)
    ctr = self.vis.get_view_control()
    pinhole_params = ctr.convert_to_pinhole_camera_parameters()
    intrinsic = pinhole_params.intrinsic.intrinsic_matrix
    extrinsic = pinhole_params.extrinsic.dot(self.T)
    if intersecting_triangles:
      """ retrieve camera extrinsic """
      R = extrinsic[:3, :3]
      trans = extrinsic[:3, 3]
      """ Hash depth pixels """
      valid_idx = np.where(depth > 1e-7)
      x, y = np.meshgrid(np.arange(depth.shape[0]),
                         np.arange(depth.shape[1]), indexing='ij')
      z = depth[valid_idx]
      x = x[valid_idx]
      y = y[valid_idx]
      points3d = np.stack([y*z, x*z, z], axis=1)
      points3d = R.T.dot(np.linalg.pinv(intrinsic).dot(points3d.T)-
                         trans[:, np.newaxis]).T

      """ valid indices """
      vertices = np.array(mesh.vertices)
      from sklearn.neighbors import NearestNeighbors as NN
      tree = NN(n_neighbors=1, algorithm='kd_tree', n_jobs=10).fit(vertices)
      dists, indices = tree.kneighbors(points3d)
      assert dists.mean() < 0.1

      correspondences = indices[:, 0]

      self.vis.remove_geometry(mesh)
      valid_pixel_indices = np.stack([valid_idx[0], valid_idx[1]], axis=1)
      mesh.transform(np.linalg.inv(self.T))
      return depth, extrinsic, intrinsic, points3d, correspondences, valid_pixel_indices
    else:
      self.vis.remove_geometry(mesh)
      mesh.transform(np.linalg.inv(self.T))
      return depth, extrinsic, intrinsic

if __name__ == '__main__':
  import matplotlib.pyplot as plt
  camera = PinholeCamera()
  mesh_male = o3d.io.read_triangle_mesh('example_data/mesh_male.ply')
  """ Depth Images """
  T = geo_util.pack(linalg.rodriguez(np.random.randn(3)),
                    np.random.randn(3))
  camera.transform(T)
  depth_image, extrinsic, intrinsic = camera.project(mesh_male)

  """ Inverse """
  points = linalg.depth2pointcloud(depth_image, extrinsic, intrinsic)
  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(points)
  o3d.draw_geometries([mesh_male, pcd])
  pixels = linalg.pointcloud2pixel(points, extrinsic, intrinsic).astype(np.int32)
  import ipdb; ipdb.set_trace()
  plt.imshow(depth_image)
  plt.show()
  depth_image[:] = 0.0
  depth_image[(pixels[:, 0], pixels[:, 1])] = 100.0
  plt.imshow(depth_image)
  plt.show()
