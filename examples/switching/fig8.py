import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import LogNorm
from scipy.interpolate import interpn


def main():
    xe = np.load("model/xe.npy")
    ye = np.load("model/ye.npy")
    xc = np.load("model/xc.npy")
    yc = np.load("model/yc.npy")
    area = np.outer(xe[1:] - xe[:-1], ye[1:] - ye[:-1])
    rho = [
        np.load("reaction/rho.npy")[None],
        np.load("pathway14/rho.npy"),
        np.load("pathway23/rho.npy"),
        np.load("pathway13/rho.npy"),
        np.load("pathway24/rho.npy"),
    ]
    tm = [
        np.load("reaction/tm.npy")[None],
        np.load("pathway14/tm.npy"),
        np.load("pathway23/tm.npy"),
        np.load("pathway13/tm.npy"),
        np.load("pathway24/tm.npy"),
    ]
    tp = [
        np.load("reaction/tp.npy")[None],
        np.load("pathway14/tp.npy"),
        np.load("pathway23/tp.npy"),
        np.load("pathway13/tp.npy"),
        np.load("pathway24/tp.npy"),
    ]
    jx = [
        np.load("reaction/jx.npy")[None],
        np.load("pathway14/jx.npy"),
        np.load("pathway23/jx.npy"),
        np.load("pathway13/jx.npy"),
        np.load("pathway24/jx.npy"),
    ]
    jy = [
        np.load("reaction/jy.npy")[None],
        np.load("pathway14/jy.npy"),
        np.load("pathway23/jy.npy"),
        np.load("pathway13/jy.npy"),
        np.load("pathway24/jy.npy"),
    ]

    fig, axes = plt.subplots(3, 5, figsize=(7, 5), sharex="col", sharey="row", constrained_layout=True)

    for j in range(5):
        proj = np.sum(rho[j], axis=(0, -1)) / area
        pcm = axes[0, j].pcolormesh(xe, ye, proj.T, norm=LogNorm(vmin=1e-6, vmax=1e-2))
    cbar = fig.colorbar(pcm, ax=axes[0])
    cbar.set_label(r"$\rho_\theta$")

    for j in range(5):
        proj = np.sum(np.where(rho[j] == 0.0, 0.0, rho[j] * tm[j]), axis=(0, -1)) / np.sum(rho[j], axis=(0, -1))
        pcm = axes[1, j].pcolormesh(xe, ye, proj.T, vmin=0, vmax=21)
    cbar = fig.colorbar(pcm, ax=axes[1])
    cbar.set_label(r"$A_\theta[m_{-}]$")

    for j in range(5):
        proj = np.sum(np.where(rho[j] == 0.0, 0.0, rho[j] * tp[j]), axis=(0, -1)) / np.sum(rho[j], axis=(0, -1))
        pcm = axes[2, j].pcolormesh(xe, ye, proj.T, vmin=0, vmax=21)
    cbar = fig.colorbar(pcm, ax=axes[2])
    cbar.set_label(r"$A_\theta[m_{+}]$")

    xj = np.linspace(-2, 2, 4 * 3 + 1)[1:-1]
    yj = np.linspace(-3, 3, 6 * 3 + 1)[1:-1]
    xs, ys = np.meshgrid(xj, yj, indexing="ij")
    xys = np.stack((xs, ys), axis=-1)
    jmax = [1.23e-3, 6.02e-4, 6.02e-4, 1.06e-4, 1.06e-4]
    for j in range(5):
        projx = np.sum(jx[j], axis=(0, -1)) / area
        projy = np.sum(jy[j], axis=(0, -1)) / area
        projx = interpn((xc, yc), projx, xys, method="linear")
        projy = interpn((xc, yc), projy, xys, method="linear")
        jnorm = np.sqrt(projx**2 + projy**2)
        jxdir = projx / jnorm
        jydir = projy / jnorm
        for i in range(3):
            axes[i, j].quiver(
                xs,
                ys,
                jxdir,
                jydir,
                jnorm,
                pivot="middle",
                angles="xy",
                cmap=LinearSegmentedColormap.from_list("wa", [(1, 1, 1, 0), (1, 1, 1, 1)]),
                scale=15,
                scale_units="inches",
                units="inches",
                width=0.01,
                clim=(0, jmax[j]),
            )

    for i in range(3):
        for j in range(5):
            ax = axes[i, j]
            ax.set_aspect("equal")
            ax.set_xlim(-2, 2)
            ax.set_ylim(-3, 3)
            ax.text(0, 2.5, "$A$", c="k", ha="center", va="center")
            ax.text(0, -2.5, "$B$", c="k", ha="center", va="center")
            ax.text(-1.5, 1.5, "$C_1$", c="w", ha="center", va="center")
            ax.text(1.5, 1.5, "$C_2$", c="w", ha="center", va="center")
            ax.text(-1.5, -1.5, "$C_3$", c="w", ha="center", va="center")
            ax.text(1.5, -1.5, "$C_4$", c="w", ha="center", va="center")

    for j in range(5):
        axes[-1, j].set_xlabel("$x_1$")
    for i in range(3):
        axes[i, 0].set_ylabel("$x_2$")

    axes[0, 0].set_title("Reaction")
    axes[0, 1].set_title("Pathway I")
    axes[0, 2].set_title("Pathway II")
    axes[0, 3].set_title("Pathway III")
    axes[0, 4].set_title("Pathway IV")

    fig.savefig("fig8.png")


if __name__ == "__main__":
    main()
