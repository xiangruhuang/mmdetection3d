import numpy as np
import open3d as o3d

def laplacian_embedding(mesh, rank=30):
  faces = np.array(mesh.triangles)
  N = np.array(mesh.vertices).shape[0]
  A = np.zeros((N, N))
  for i in range(3):
    for j in range(3):
      if i == j:
        continue
      A[(faces[:, i], faces[:, j])] = 1.0
  A = A + A.T
  diag = A.dot(np.ones(N))
  L = np.diag(diag) - A
  eigvals, eigvecs = np.linalg.eigh(L)
  embedding = eigvecs[:, 1:(rank+1)]
  return embedding

def floyd(mesh):
  faces = np.array(mesh.triangles)
  N = np.array(mesh.vertices).shape[0]
  Dist = np.zeros((N, N)) + 1e10
  for i in range(N):
    Dist[i, i] = 0.0
  for i in range(3):
    for j in range(3):
      if i == j:
        continue
      Dist[(faces[:, i], faces[:, j])] = 1.0
  #for k in range(N):
  #  print(k, N)
  #  for i in range(N):
  #    for j in range(N):
  #      if (i == j) or (i == k) or (j == k):
  #        continue
  #      if Dist[i, j] > Dist[i, k] + Dist[k, j]:
  #        Dist[i, j] = Dist[i, k] + Dist[k, j]
  return Dist

if __name__ == '__main__':
  import scipy.io as sio
  mesh = o3d.io.read_triangle_mesh('example_data/mesh_female.ply')
  embedding = laplacian_embedding(mesh, rank=128)
  embedding = (embedding - embedding.min(0)[np.newaxis, :])/(embedding.max(0)-embedding.min(0))[np.newaxis, :]
  import ipdb; ipdb.set_trace()
  sio.savemat('smpl_laplacian_embedding_128.mat', {'male': embedding, 'female': embedding})
  #N = np.array(mesh.vertices).shape[0]
  #Dist = np.loadtxt('floyd_results').reshape((6890, 6890))
  #dists = []
  #dists = Dist.reshape(-1)
  #feature_dists = np.linalg.norm(embedding[np.newaxis, :, :] - embedding[:, np.newaxis, :], 2, axis=-1).reshape(-1)

  #import matplotlib.pyplot as plt
  #feature_dists = np.array(feature_dists)
  #dists = np.array(dists)
  #random_idx = np.random.permutation(N*N)[:100000]
  #plt.scatter(dists[random_idx], feature_dists[random_idx], 1.0)
  #plt.savefig('hey2.png')
  #plt.show()
