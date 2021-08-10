import numpy as np
import torch

def cross_op(r):
  """
  Return the cross operator as a matrix
  i.e. for input vector r \in \R^3
  output rX s.t. rX.dot(v) = np.cross(r, v)
  where rX \in \R^{3 X 3}
  """
  rX = np.zeros((3, 3))
  rX[0, 1] = -r[2]
  rX[0, 2] = r[1]
  rX[1, 2] = -r[0]
  rX = rX - rX.T
  return rX

def cross_op_torch(r):
  """
  Return the cross operator as a matrix
  i.e. for input vector r \in \R^3
  output rX s.t. rX.dot(v) = np.cross(r, v)
  where rX \in \R^{3 X 3}
  """
  if len(r.shape) > 1:
      rX = torch.zeros(r.shape[0], 3, 3).to(r)
      rX[..., 0, 1] = -r[..., 2]
      rX[..., 0, 2] = r[..., 1]
      rX[..., 1, 2] = -r[..., 0]
      rX = rX - rX.transpose(-1, -2)
  else:
      rX = torch.zeros(3, 3).to(r)
      rX[0, 1] = -r[2]
      rX[0, 2] = r[1]
      rX[1, 2] = -r[0]
      rX = rX - rX.T
  return rX

def rodrigues(r):
  """
  Return the rotation matrix R as a function of (axis, angle)
  following Rodrigues rotation theorem.
  (axis, angle) are represented by an input vector r, where
  axis = r / l2_norm(r) and angle = l2_norm(r)
  """
  theta = np.linalg.norm(r, 2)
  if theta < 1e-12:
    return np.eye(3)
  k = r / theta
  """ Rodrigues """
  R = np.cos(theta)*np.eye(3) + np.sin(theta)*cross_op(k) + (1-np.cos(theta))*np.outer(k, k)
  return R

def pack(R, t):
  """
    T = [[R, t]; [0, 1]]

  """
  T = np.eye(4)
  T[:3, :3] = R
  T[:3, 3] = t.reshape(3)
  return T

def pack_torch(R, t):
  """
    T = [[R, t]; [0, 1]]

  """
  T = t.unsqueeze(-1) # [..., 3, 1]
  T = torch.cat([R, t.unsqueeze(-1)], dim=-1) # [..., 3, 4]
  T_bottom = torch.zeros(R.shape[0], 1, 4).to(T) # [..., 1, 4]
  T_bottom[..., 0, 3] = 1.0
  T = torch.cat([T, T_bottom], dim=-2) # [..., 4, 4]
  return T

def unpack(T):
  """ R = T[:3, :3]; t = T[:3, 3]
  """
  R = T[..., :3, :3]
  t = T[..., :3, 3]
  return R, t

def matrix_to_axis_angle(R):
    """Convert Rotation Matrices to Axis-Angle representation.
    
    R.diag = (1-cost) r * r + cost 1
    R.off_diag = sint r + (1-cost) (r r^T - r*r)

    Args:
        R (torch.Tensor or np.ndarray, shape=[..., 3, 3]): rotation matrices
        
    """
    if isinstance(R, np.ndarray):
        n_dim = len(R.shape)
        Rs = R - R.transpose([i for i in range(n_dim-2)] + [-1, -2]) # 
        # Rs.diag = 2 cost + 2 (1-cost) k * k
        # Rs.off_diag = 
        sint_r = np.array([Rs[..., 2, 1], Rs[..., 0, 2], Rs[..., 1, 0]]) # [..., 3]
        sint = np.linalg.norm(sint_kX, ord=2, axis=-1) # [...]
        r = np.zeros(R[..., 0].shape) # [..., 3]
        r = sint_kX / sint[:, np.newaxis] # [..., 3]
    else:
        # torch.Tensor
        if len(R.shape) > 2:
            Rs = R - R.transpose(-1, -2)
            sint_r = torch.stack([Rs[..., 2, 1], Rs[..., 0, 2], Rs[..., 1, 0]],
                                 dim=-1)/2.0 # [..., 3]
            cost = ((R.reshape(-1, 9)[..., [0, 4, 8]].sum(-1)-1.0)/2.0).clip(-1, 1)
            theta = cost.arccos()
            sint = sint_r.norm(p=2, dim=-1).clip(0, 1)
            valid_mask = sint > 1e-6
            r = torch.zeros(list(R.shape[:-2])+[3]).to(R)
            r[valid_mask] = sint_r[valid_mask] / sint[valid_mask].unsqueeze(-1)
            r = r * theta.unsqueeze(-1)
        else:
            Rs = R - R.transpose(-1, -2)
            sint_r = torch.stack([Rs[..., 2, 1], Rs[..., 0, 2], Rs[..., 1, 0]],
                               dim=-1)/2.0 # [3]
            cost = ((R.reshape(-1, 9)[..., [0, 4, 8]].sum(-1)-1.0)/2.0).clip(-1, 1)
            sint = sint_r.norm(p=2, dim=-1).clip(0, 1)
            theta = cost.arccos()
            r = torch.zeros(3).to(R) # [..., 3]
            if sint > 1e-6:
                r = sint_r / sint * theta
    return r
        
def axis_angle_to_matrix(r):
    """
    Return the rotation matrix R as a function of (axis, angle)
    following Rodrigues rotation theorem.
    (axis, angle) are represented by an input vector r, where
    axis = r / l2_norm(r) and angle = l2_norm(r)
    """
    if len(r.shape) == 1:
        theta = r.norm(p=2)
        R = torch.eye(3).to(r)
        if theta > 1e-6:
            k = r / theta
            """ Rodrigues """
            cost = torch.cos(theta)
            sint = torch.sin(theta)
            R = cost*torch.eye(3).to(cost) + sint*cross_op_torch(k) + \
                (1.0-cost)*(k.unsqueeze(-1) * k.unsqueeze(-2))
    else:
        theta = r.norm(p=2, dim=-1)
        #theta = np.linalg.norm(r, ord=2, axis=-1) # [...]
        valid_mask = theta > 1e-6
        R = torch.eye(3).repeat(r.shape[0], 1, 1).to(r)

        k = r[valid_mask] / theta[valid_mask].unsqueeze(-1)
        """ Rodrigues """
        cost = torch.cos(theta[valid_mask]).unsqueeze(-1).unsqueeze(-1)
        sint = torch.sin(theta[valid_mask]).unsqueeze(-1).unsqueeze(-1)
        R[valid_mask] = cost*torch.eye(3).to(cost) + sint*cross_op_torch(k) + \
                        (1.0-cost)*(k.unsqueeze(-1) * k.unsqueeze(-2))
    return R
    
