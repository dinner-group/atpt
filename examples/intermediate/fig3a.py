import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import marching_cubes


def main():
    u = np.load("model/u.npy")
    xc = np.load("model/xc.npy")
    yc = np.load("model/yc.npy")
    zc = np.load("model/zc.npy")

    # compute isosurface
    verts, faces, _, _ = marching_cubes(u, -3, spacing=(0.1, 0.1, 0.1))
    verts += np.array([xc[0], yc[0], zc[0]])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2])
    ax.view_init(elev=45, azim=-120)
    ax.set_proj_type("ortho")
    ax.set_box_aspect([2, 3, 3])
    ax.set_xlim(-1, 3)
    ax.set_ylim(-3, 3)
    ax.set_zlim(-3, 3)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_zlabel("$x_3$")
    ax.text(0, 2, 2, "$A$", ha="center", va="center")
    ax.text(0, -2, -2, "$B$", ha="center", va="center")
    ax.text(0, 0, 2, "$S_1$", ha="center", va="center")
    ax.text(0, 0, -2, "$S_2$", ha="center", va="center")
    ax.text(2, 0, 0, "$C$", ha="center", va="center")
    fig.savefig("fig3a.png")


if __name__ == "__main__":
    main()
