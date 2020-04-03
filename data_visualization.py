from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
import numpy as np
import tqdm
import time
import threading

fig = plt.figure()
ax = plt.gca(projection='3d')
ani = FuncAnimation(fig, animate, interval=1)
plt.show()

def animate(i):
    ax.clear()
    ax.scatter3D(x, y, z, color='black', label='Joints')
    ax.plot3D(x, y, z, c='b', label='Trajectory')
    ax.set_title(title, size=10)
    ax.legend(loc=2, prop={'size':7})
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim([-60, 60])
    ax.set_ylim([-60, 60])
    ax.set_zlim([0, 60])
