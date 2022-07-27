import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle
from matplotlib.patches import Ellipse


def main():
    w = np.load("model/w.npy")
    xe = np.load("model/xe.npy")
    ye = np.load("model/ye.npy")
    area = np.outer(xe[1:] - xe[:-1], ye[1:] - ye[:-1])
    wproj = np.sum(w, axis=-1) / np.sum(w) / area

    fig, ax = plt.subplots()
    pcm = ax.pcolormesh(xe, ye, wproj.T, norm=LogNorm(vmin=1e-6, vmax=2))
    ax.set_aspect("equal")
    ax.set_xlim(-2, 2)
    ax.set_ylim(-3, 3)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.text(0, 2.5, "$A$", c="k", ha="center", va="center")
    ax.text(0, -2.5, "$B$", c="k", ha="center", va="center")
    ax.text(-1.5, 1.5, "$C_1$", c="k", ha="center", va="center")
    ax.text(1.5, 1.5, "$C_2$", c="k", ha="center", va="center")
    ax.text(-1.5, -1.5, "$C_3$", c="k", ha="center", va="center")
    ax.text(1.5, -1.5, "$C_4$", c="k", ha="center", va="center")
    ax.add_patch(Circle((0, -3), 1, fc="none", ec="k"))
    ax.add_patch(Circle((0, 3), 1, fc="none", ec="k"))
    ax.add_patch(Ellipse((-2, 2), np.sqrt(2), np.sqrt(8), 45, fc="none", ec="k"))
    ax.add_patch(Ellipse((2, -2), np.sqrt(2), np.sqrt(8), 45, fc="none", ec="k"))
    ax.add_patch(Ellipse((2, 2), np.sqrt(2), np.sqrt(8), -45, fc="none", ec="k"))
    ax.add_patch(Ellipse((-2, -2), np.sqrt(2), np.sqrt(8), -45, fc="none", ec="k"))
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label(r"$\rho_{\theta}$")
    fig.savefig("fig6b.png")


if __name__ == "__main__":
    main()
