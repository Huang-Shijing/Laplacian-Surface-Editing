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
        # fruits = ['apple', 'banana', 'cherry']
        # for index, fruit in enumerate(fruits):
        #     print(f"Index: {index}, Fruit: {fruit}")
        # 输出
        # Index: 0, Fruit: apple
        # Index: 1, Fruit: banana
        # Index: 2, Fruit: cherry
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
            #第i行有degree + 1个元素非零
            I += [i] * (degree + 1)
            #对角线元素加上邻接点
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

    def get_delta(self , anchor_ids):
        Ls = self.get_Ls_matrix(anchor_ids)
        # print(Ls)
        delta = Ls.dot(self.vps) # n+k, dim
        return Ls , delta
    
    def solve(self, anchors, anchor_ids , Ls=None , delta=None):
        k = anchor_ids.shape[0]
        #节点数量
        n = self.vps.shape[0]
        if Ls is None or delta is None:
            Ls , delta = self.get_delta(anchor_ids)
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
       
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    x = np.linspace(0, 40, 100)
    y = np.sin(0.5*x)
    #锚点
    anchor_ids = np.array([100, 1]) - 1
    anchors = np.array([x[-1], y[-1]+10, x[0], y[0]]).reshape(-1, 2)
    #坐标
    vps = np.stack((x, y), axis=1) # n, 2
    faces = []
    #相邻节点
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
