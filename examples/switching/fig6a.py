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
    ax.view_init(elev=60, azim=-105)
    ax.set_proj_type("ortho")
    ax.set_box_aspect([2, 3, 3])
    ax.set_xlim(-2, 2)
    ax.set_ylim(-3, 3)
    ax.set_zlim(-3, 3)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_zlabel("$x_3$")
    ax.text(0, 3, 0, "$A$", c="k", ha="center", va="center")
    ax.text(0, -3, 0, "$B$", c="k", ha="center", va="center")
    ax.text(-1.25, 1.5, -1, "$C_1$", c="k", ha="center", va="center")
    ax.text(1.25, 1.5, 1, "$C_2$", c="k", ha="center", va="center")
    ax.text(-1.25, -1.5, 1, "$C_3$", c="k", ha="center", va="center")
    ax.text(1.25, -1.5, -1, "$C_4$", c="k", ha="center", va="center")
    fig.savefig("fig6a.png")


if __name__ == "__main__":
    main()
