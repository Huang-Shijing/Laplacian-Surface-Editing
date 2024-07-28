import matplotlib.pyplot as plt

fig = None
callback_connected = False

# 鼠标滚轮放大缩放
def call_back(event):
    axtemp = event.inaxes
    if axtemp is None:
        return

    x_min, x_max = axtemp.get_xlim()
    y_min, y_max = axtemp.get_ylim()
    xfanwei = (x_max - x_min) / 10
    yfanwei = (y_max - y_min) / 10

    if event.button == 'up':
        axtemp.set_xlim(x_min + xfanwei, x_max - xfanwei)
        axtemp.set_ylim(y_min + yfanwei, y_max - yfanwei)
    elif event.button == 'down':
        axtemp.set_xlim(x_min - xfanwei, x_max + xfanwei)
        axtemp.set_ylim(y_min - yfanwei, y_max + yfanwei)

    fig.canvas.draw_idle()

#绘制网格
def plot_grid(grid, x_coord, y_coord, nose_x):
    #为了同时实现滚轮放大缩放以及不同时刻图像表示在同一个图像位置上以便动态观察
    global fig, callback_connected
    if fig is None:
        fig = plt.figure()
        if not callback_connected:
            fig.canvas.mpl_connect('scroll_event', call_back)
            callback_connected = True
    else:
        plt.figure(fig.number)

    plt.clf()  # 清除当前图形
    for i in range(len(grid)):
        node1 = int(grid[i, 0])
        node2 = int(grid[i, 1])
        xx = [x_coord[node1 - 1], x_coord[node2 - 1]]
        yy = [y_coord[node1 - 1], y_coord[node2 - 1]]
        #节点是否在物面上
        if grid[i, 2] == 3: 
            plt.plot(xx, yy, '-k', linewidth=1.5)
        else:
            plt.plot(xx, yy, '-r' )

    plt.axis('equal')
    plt.axis([nose_x - 0.5, nose_x + 1.5, -0.7, 0.7])
    plt.pause(0.001)

def plot_gridWithT(grid, x_coord, y_coord):
    global fig
    fig = plt.figure()
    fig.canvas.mpl_connect('scroll_event', call_back)
    plt.clf()  # 清除当前图形
    for i in range(len(grid)):
        node1 = int(grid[i, 0])
        node2 = int(grid[i, 1])
        xx = [x_coord[node1 - 1], x_coord[node2 - 1]]
        yy = [y_coord[node1 - 1], y_coord[node2 - 1]]
        #节点是否在物面上
        if grid[i, 2] == 3: 
            plt.plot(xx, yy, '-k', linewidth=1.5)
        else:
            plt.plot(xx, yy, '-r' )

    plt.axis('equal')
    plt.axis([ -1 ,  1 , -0.7 , 0.7])
    plt.show()