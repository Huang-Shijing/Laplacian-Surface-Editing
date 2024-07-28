import numpy as np
from PLOT import plot_grid
from read_grid import Coord , wallNodes , farwallNodes , Grid
from LapDef import LaplacianDeformation

xCoord = np.array([Coord[:, 0]]).T
yCoord = np.array([Coord[:, 1]]).T
nNodes = Coord.shape[0]
nWallNodes = len(wallNodes)

#参数
lamda = 1      #波长
c = 0.1        #波速
T = 2.0        #周期
t = 0          #起始时间
dt = 0.5       #时间间隔
r0 = 10.0      #紧支半径
basis = 11      #基函数类型

while t < 10:
    dy = np.zeros((nWallNodes,1))
    xCoord_new = np.array([xCoord[:, 0]]).T #避免改变原数组
    yCoord_new = np.array([yCoord[:, 0]]).T
    t = t + dt

    # 对物面点进行变形
    for i in range(nWallNodes):
        wall_index = wallNodes[i] - 1
        xCoord_new[wall_index] = xCoord[wall_index]
    nose_x = min(xCoord_new[wallNodes-1])

    for i in range(nWallNodes):
        wall_index = wallNodes[i] - 1
        x = xCoord_new[wall_index] - nose_x
        y = yCoord[wall_index]

        #此处表示已知所有物面点的位移
        A = min(1, t / T) * (0.02 - 0.0825 * x + 0.1625 * x**2)
        B = A * np.sin(2 * np.pi / lamda * (x - c * t))
        dy[i,0] = B[0]
        yCoord_new[wall_index] = yCoord[wall_index] + B[0]

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

    model = LaplacianDeformation(Coord, faces)
    new_pnts = model.solve(anchors, anchor_ids)

    # 绘制网格
    plot_grid(Grid, new_pnts[:, 0], new_pnts[:, 1], nose_x)
    if input(f"t={t} 按任意键继续生成下一时刻网格，或输入q退出循环:\n") == 'q':
        break

