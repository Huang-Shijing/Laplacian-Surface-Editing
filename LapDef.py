import numpy as np
from scipy.sparse import coo_matrix, block_diag
from scipy.sparse.linalg import lsqr
from collections import defaultdict

# https://github.com/SecretMG/Laplacian-Mesh-Deformation
class LaplacianDeformation:
    def __init__(self, vps, faces) -> None:
        self.vps = vps
        self.faces = faces
        self.adj_info = self.get_adj_info()
        self.mode = 'mean'
        self.dim = vps.shape[1]
    
    def get_adj_info(self):
        adj_dic = defaultdict(set)
        for idx, face in enumerate(self.faces):
            for f in face:
                adj_dic[idx].add(f)
        return adj_dic

    def get_Ls_matrix(self, anchor_ids, anchor_weight=None):
        k = anchor_ids.shape[0]
        n = self.vps.shape[0]
        # 生成稀疏矩阵，不建议使用np.zeros依次赋值
        data = []
        I = [] # 行序号
        J = []  # 列序号
        for i in range(n):
            neighbors = [v for v in self.adj_info[i]]  # ids
            degree = len(neighbors)
            # data += [degree] + [-1] * degree  # D-A
            data += [1] + [-1 / degree] * degree  # (I - D^{-1}A)
            I += [i] * (degree + 1)
            J += [i] + neighbors
        # 为了anchor增加的方程组，组成了超定方程 
        for i in range(k):
            if anchor_weight is None:
                data += [1]
            else:
                data += [anchor_weight[i]]
            I += [n+i]
            J += [anchor_ids[i]] 
        # [coco_matrix] https://blog.csdn.net/kittyzc/article/details/126077002
        Ls = coo_matrix((data, (I, J)), shape=(n+k, n)).todense()
        return Ls

    def solve(self, anchors, anchor_ids):
        k = anchor_ids.shape[0]
        n = self.vps.shape[0]
        Ls = self.get_Ls_matrix(anchor_ids)
        print(Ls)
        delta = Ls.dot(self.vps) # n+k, dim
        # print(delta.shape);exit()
        # 为后k行的原始顶点坐标修改为anchor的坐标（即修改为人为指定的已知条件）
        for i in range(k):
            delta[n+i] = anchors[i]  
            # update mesh vertices with least-squares solution
        res = np.zeros((n, self.dim))
        delta = np.array(delta)
        for i in range(self.dim):
            # res[:, i] = lsqr(Ls, delta[:, i])[0] # n,
            res[:, i] = np.linalg.pinv(Ls).dot(delta[:, i])
        return res

# class LaplacianDeformationWithT(LaplacianDeformation):
#     def __init__(self, vps, faces) -> None:
#         super().__init__(vps, faces)
#         self.extra_A = None

#     def get_Ls_matrix(self):
#         n = self.vps.shape[0]
#         # 生成稀疏矩阵，不建议使用np.zeros依次赋值
#         data = []
#         I = [] # 行序号
#         J = []  # 列序号
#         for i in range(n):
#             neighbors = [v for v in self.adj_info[i]]  # ids
#             degree = len(neighbors)
#             # data += [degree] + [-1] * degree  # D-A
#             data += [1] + [-1 / degree] * degree  # (I - D^{-1}A)
#             I += [i] * (degree + 1)
#             J += [i] + neighbors
#         Ls = coo_matrix((data, (I, J)), shape=(n, n)).todense()
#         return Ls

#     def solve(self, anchors, anchor_ids, anchor_weight=1):
#         k = anchor_ids.shape[0]
#         n = self.vps.shape[0]
#         Ls = self.get_Ls_matrix() # n, n
#         delta = Ls.dot(self.vps) # n, dim
#         LS = np.zeros([self.dim*n, self.dim*n])
#         for idx in range(self.dim):
#             LS[idx*n:(idx+1)*n, idx*n:(idx+1)*n] = (-1) * Ls
#         # [
#         #   [s, -w],
#         #   [w, s],
#         # ]
#         for idx in range(n):
#             neighbors = [v for v in self.adj_info[idx]]  # ids
#             ring = np.array([idx] + neighbors)
#             V_ring = self.vps[ring]
#             n_ring = V_ring.shape[0]
#             A = np.zeros((n_ring * self.dim, 4))
#             for j in range(n_ring):
#                 A[j]          = [V_ring[j,0], -V_ring[j,1], 1, 0]
#                 A[j+n_ring]   = [V_ring[j,1], V_ring[j,0],  0, 1]
#                 # A[j+2*n_ring] = [V_ring[j,2],  V_ring[j,1], -V_ring[j, 0],           0 , 0, 0, 1]
#             inv_A = np.linalg.pinv(A)  # 4, n_ring*2
#             s = inv_A[0]
#             w = inv_A[1]
#             T_delta = np.stack(
#                 [
#                     delta[idx, 0] * s - delta[idx, 1] * w,
#                     delta[idx, 0] * w + delta[idx, 1] * s
#                 ], axis=0
#             ) # 2, 2*n_ring
#             LS[idx, np.hstack([ring, ring+n,])] += T_delta[0]
#             LS[idx+n, np.hstack([ring, ring+n])] += T_delta[1]

#         constraint_coef = []
#         constraint_b = []
#         for idx in range(k):
#             vps_idx = anchor_ids[idx]
#             tmp_coeff_x = np.zeros((self.dim * n))
#             tmp_coeff_x[vps_idx] = anchor_weight
#             tmp_coeff_y = np.zeros((self.dim * n))
#             tmp_coeff_y[vps_idx+n] = anchor_weight
            
#             constraint_coef.append(tmp_coeff_x)
#             constraint_coef.append(tmp_coeff_y)
#             constraint_b.append(anchor_weight * anchors[idx, 0])
#             constraint_b.append(anchor_weight * anchors[idx, 1])
#         constraint_coef = np.matrix(constraint_coef)
#         constraint_b = np.array(constraint_b)
#         A = np.vstack([LS, constraint_coef])
#         b = np.hstack([np.zeros(self.dim * n), constraint_b])
#         spA = coo_matrix(A).todense()
#         # 求解的时候保证shape为二维
#         V_prime = lsqr(spA, b)[0]
#         # V_prime = np.linalg.lstsq(spA, b.reshape(-1, 1), rcond=None)[0]
#         # V_prime = np.linalg.pinv(spA).dot(b)
#         new_pnts = []
#         for idx in range(n):
#             new_pnts.append(list(V_prime[idx + i*n] for i in range(self.dim)))
#         new_pnts = np.array(new_pnts)
#         return new_pnts
       
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    x = np.linspace(0, 40, 100)
    y = np.sin(0.5*x)
    anchor_ids = np.array([100, 1]) - 1
    anchors = np.array([x[-1], y[-1]+10, x[0], y[0]]).reshape(-1, 2)
    vps = np.stack((x, y), axis=1) # n, 2
    faces = []
    for idx in range(len(x)):
        if idx == 0:
            faces.append((idx+1,))
        elif idx == (len(x)-1):
            faces.append((idx-1,))           
        else:
            faces.append((idx-1, idx+1))
    # model = LaplacianDeformationWithT(vps, faces)
    model = LaplacianDeformation(vps, faces)
    new_pnts = model.solve(anchors, anchor_ids)
    plt.scatter(x, y)
    plt.scatter(new_pnts[:, 0], new_pnts[:, 1])
    plt.show()
