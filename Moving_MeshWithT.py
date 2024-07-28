import numpy as np
from PLOT import plot_gridWithT
from read_grid import Coord , wallNodes , farwallNodes , Grid
# from LapDef import LaplacianDeformation
from LapDefT import LaplacianDeformationWithT

xCoord = np.array([Coord[:, 0]]).T
yCoord = np.array([Coord[:, 1]]).T
nNodes = Coord.shape[0]
nWallNodes = len(wallNodes)

xCoord_new = np.array([xCoord[:, 0]]).T #避免改变原数组
yCoord_new = np.array([yCoord[:, 0]]).T


# 对物面点进行变形
for i in range(nWallNodes):
    wall_index = wallNodes[i] - 1
    dy = 0.2 * np.sin(-2*np.pi * xCoord_new[wall_index] )
    yCoord_new[wall_index] += dy

#锚点，远场边界点固定，物面边界点运动
anchor_ids = np.concatenate((wallNodes -1 , farwallNodes - 1))
wallCoord_new = np.concatenate((xCoord_new[wallNodes-1] , yCoord_new[wallNodes - 1]) ,axis=1)    
farwallCoord_new = np.concatenate(([Coord[farwallNodes-1,0]] , [Coord[farwallNodes-1,1]]) ,axis=0)
anchors = np.concatenate((wallCoord_new , farwallCoord_new.T))
faces = []
for i in range(nNodes):
    nerb = []
    for j in range(len(Grid)):
        if Grid[j,0]  == i + 1:
            nerb.append(Grid[j,1] - 1)
        elif Grid[j,1]  == i + 1:
            nerb.append(Grid[j,0] - 1)
    nerb.sort()
    faces.append(nerb)

# model = LaplacianDeformation(Coord, faces)
model = LaplacianDeformationWithT(Coord, faces)
new_pnts = model.solve(anchors, anchor_ids)

# 绘制网格
plot_gridWithT(Grid, new_pnts[:, 0], new_pnts[:, 1])