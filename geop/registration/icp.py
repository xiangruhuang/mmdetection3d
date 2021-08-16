import numpy as np
import torch
import open3d as o3d
import sys
from sklearn.neighbors import NearestNeighbors as NN
import geop.geometry.util as gutil
from torch_scatter import scatter_add, scatter_mean
from torch_geometric.nn import knn

""" Iterative Closest Points (ICP) Method according to point-to-plane metric.
    Inputs:
        source: o3d.geometry.PointCloud
        target: o3d.geometry.PointCloud
        sigma: soft-thresholding [default 0.01]
        max_iter: maximum number of iterations [default 100]
        stopping_threshold: stopping threshold for ICP algorithm [default 1e-4]
    Outputs:
        transform: np.ndarray of shape [4, 4].
                   Transformation from source to target.
"""
def icp_reweighted(source, target, sigma=0.01, max_iter = 100,
                   stopping_threshold=1e-4):
    """ If target has no normals, estimate """
    if np.array(target.normals).shape[0] == 0:
        search_param = o3d.geometry.KDTreeSearchParamHybrid(
                                        radius=0.2, max_nn=30)
        o3d.estimate_normals(target, search_param=search_param)

    tree = NN(n_neighbors=1, algorithm='kd_tree', n_jobs=10)
    tree = tree.fit(np.array(target.points))
    n = np.array(source.points).shape[0]
    normals = np.array(target.normals)
    points = np.array(target.points)
    weights = np.zeros(n)
    errors = []
    transform = np.eye(4)

    for itr in range(max_iter):
        p = np.array(source.points)
        R, trans = gutil.unpack(transform)
        p = (R.dot(p.T) + trans.reshape((3, 1))).T
        _, indices = tree.kneighbors(p)

        """ (r X pi + pi + t - qi)^T ni """
        """( <r, (pi X ni)> + <t, ni> + <pi-qi, ni> )^2"""
        """ (<(r; t), hi> + di)^2 """
        nor = normals[indices[:, 0], :]
        q = points[indices[:, 0], :]
        d = np.sum(np.multiply(p-q, nor), axis=1) #[n]
        h = np.zeros((n, 6))
        h[:, :3] = np.cross(p, nor)
        h[:, 3:] = nor
        weight = (sigma**2)/(np.square(d)+sigma**2)
        H = np.multiply(h.T, weight).dot(h)
        g = -h.T.dot(np.multiply(d, weight))
        delta = np.linalg.solve(H + np.eye(H.shape[0])*0.1, g)
        errors = np.abs(d)
        mean_err = np.mean(errors)
        if (itr + 1 == max_iter) or \
                (np.linalg.norm(delta, 2) < stopping_threshold):
            print('iter=%d, delta=%f, mean error=%f, median error=%f' % (
                    itr, np.linalg.norm(delta, 2),
                    np.mean(errors), np.median(errors)))
            break
        trans = delta[3:]
        R = gutil.rodrigues(delta[:3])
        T = gutil.pack(R, trans)
        transform = T.dot(transform)

    return transform, mean_err

def batched_icp(source_points, target_points, target_normals,
                point2cluster, init_transf=None,
                sigma=0.05, max_iter=40,
                stopping_threshold=1e-2,
                device='cpu'):
    """
    Args:
        source_points (torch.Tensor, shape=[N1, 3]): moving frame
        target_points (torch.Tensor, shape=[N2, 3]): reference frame
        target_normals (torch.Tensor, shape=[N2, 3]): normal
        point2cluster (torch.Tensor, shape=[N1]): cluster assignments ([M])

    Returns:
        transform (torch.Tensor, shape=[M, 4, 4]): per cluster rigid
            transformation

    """
    source_points = torch.as_tensor(source_points).float().to(device)
    target_points = torch.as_tensor(target_points).float().to(device)
    target_points_cpu = target_points.detach().cpu()
    target_normals = torch.as_tensor(target_normals).float().to(device)
    point2cluster = torch.as_tensor(point2cluster).long().to(device)
    num_clusters = point2cluster.max()+1
    if init_transf is None:
        init_transf = torch.eye(4).repeat(num_clusters, 1, 1).float().to(device) # [M, 4, 4]
    else:
        init_transf = init_transf.float().to(device)
    
    transform = init_transf.clone()
    lamb = 0.1
    active_mask = torch.as_tensor([True], dtype=torch.bool
                      ).repeat(num_clusters) # [M]

    import time
    for sigma in [0.05, 0.2, 1.0]:
        p = source_points.clone()
        R, trans = gutil.unpack(transform) # [M, 3, 3], [M, 3]
        R_p, trans_p = R[point2cluster], trans[point2cluster] # [N, 3, 3], [N, 3]
        p = R_p.matmul(p.unsqueeze(-1)).squeeze(-1) + trans_p # [N, 3]
        for itr in range(max_iter):
            active_mask_p = active_mask[point2cluster] # [N]
            p_active = p[active_mask_p]
            t0 = time.time()
            e0, e1 = knn(target_points_cpu, p_active.detach().cpu(), 1).to(device)
            #print(f'itr={itr}, p_active={active_mask_p.float().sum()}, time={(time.time()-t0):.4f}')
            nor = target_normals[e1] # [N, 3]
            q = target_points[e1] # [N, 3]
            d = ((p_active - q) * nor).sum(-1) # [N]
            h = torch.zeros(p_active.shape[0], 6) # [N, 6]
            h = torch.cat([torch.cross(p_active, nor), nor], dim=-1)
            point2cluster_active = point2cluster[active_mask_p]
            #h[:, :3] = torch.cross(p, nor)
            #h[:, 3:] = nor
            weight = (sigma**2)/(d.square()+sigma**2) # [N]
            hhT = h.unsqueeze(-1).matmul(h.unsqueeze(-2)) # [N, 6, 1] @ [N, 1, 6]
            hhT = (hhT * weight.unsqueeze(-1).unsqueeze(-1)).view(-1, 36) # [N, 36]
            H = scatter_add(hhT.view(-1, 36).double(), point2cluster_active,
                    dim=0, dim_size=num_clusters).view(-1, 6, 6) # [M, 6, 6]
            #H = (h.T * weight).mm(h) # [6, 6]
            g = -h * (d * weight).unsqueeze(-1) # [N, 6]
            g = scatter_add(g.double(), point2cluster_active, dim=0,
                            dim_size=num_clusters) # [M, 6]
            # torch.linalg.solve is not working...
            
            H = H + lamb*torch.eye(6).repeat(H.shape[0], 1, 1).to(H.device)
            if H.is_cuda:
                delta = torch.linalg.lstsq(H, g).solution.float() # [M, 6]
            else:
                delta = np.linalg.solve(H.numpy(), g.numpy())
                delta = torch.as_tensor(delta).float().to(H.device)
            
            #g = -h.T.mm((d * weight).unsqueeze(-1)).squeeze(-1) # [6, 1]
            
            trans = delta[..., 3:] # [M, 3]
            R = gutil.axis_angle_to_matrix(delta[..., :3]) #[M, 3, 3]
            T = gutil.pack_torch(R, trans) # [M, 4, 4]
            transform = T.matmul(transform) # [M, 4, 4]
            #T = gutil.pack(R, trans) # [M, 4, 4]
            R_p, trans_p = R[point2cluster], trans[point2cluster] # [N, 3, 3], [N, 3]
            p = R_p.matmul(p.unsqueeze(-1)).squeeze(-1) + trans_p # [N, 3]
            
            active_mask = delta.norm(p=2, dim=-1) > stopping_threshold
            if delta.norm(p=2, dim=-1).max() <= stopping_threshold:
                break
            
        e0, e1 = knn(target_points_cpu, p.detach().cpu(), 1).to(device)
        nor = target_normals[e1] # [N, 3]
        q = target_points[e1] # [N, 3]
        d = ((p - q) * nor).sum(-1) # [N]
        errors = scatter_mean(d.abs(), point2cluster, dim=0, dim_size=num_clusters) #[M]
        active_mask[errors > 0.1] = True
        active_mask[errors <= 0.1] = False
        if sigma < 0.8:
            transform[active_mask, :, :] = init_transf[active_mask]

    return transform.detach().cpu()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='reweighted ICP algorithm')
    parser.add_argument('--source', type=str,
                        help='source point cloud or mesh in .ply format')
    parser.add_argument('--target', type=str,
                        help='target point cloud or mesh in .ply format')
    args = parser.parse_args()

    source = o3d.io.read_point_cloud(args.source)
    try:
        mesh = o3d.read_triangle_mesh(args.target)
        if np.array(mesh.triangles).shape[0] == 0:
            assert False
        v = np.array(mesh.vertices)
        tri = np.array(mesh.triangles)
        v1 = v[tri[:, 0], :]
        v2 = v[tri[:, 1], :]
        v3 = v[tri[:, 2], :]
        normals = np.cross(v1-v3, v2-v3)
        normals = (normals.T / np.linalg.norm(normals, 2, axis=1)).T
        centers = (v1+v2+v3)/3.0

        target = o3d.PointCloud()
        target.points = o3d.utility.Vector3dVector(centers)
        target.normals = o3d.utility.Vector3dVector(normals)
    except:
        target = o3d.io.read_point_cloud(args.target)
        search_param = o3d.geometry.KDTreeSearchParamHybrid(
                                        radius=0.2, max_nn=30)
        target.estimate_normals(search_param=search_param)

    transformation = icp_reweighted(source, target)
    source.transform(transformation)
    o3d.visualization.draw_geometries([source, target])
