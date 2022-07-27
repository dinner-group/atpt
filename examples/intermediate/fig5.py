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
    rho = np.concatenate(
        [
            [np.load("reaction/rho.npy")],
            np.load("uncatalyzed/rho.npy"),
            np.load("catalyzed/rho.npy"),
        ]
    )
    qp = np.concatenate(
        [
            [np.load("reaction/qp.npy")],
            np.load("uncatalyzed/qp.npy"),
            np.load("catalyzed/qp.npy"),
        ]
    )
    tp = np.concatenate(
        [
            [np.load("reaction/tp.npy")],
            np.load("uncatalyzed/tp.npy"),
            np.load("catalyzed/tp.npy"),
        ]
    )
    jx = np.concatenate(
        [
            [np.load("reaction/jx.npy")],
            np.load("uncatalyzed/jx.npy"),
            np.load("catalyzed/jx.npy"),
        ]
    )
    jy = np.concatenate(
        [
            [np.load("reaction/jy.npy")],
            np.load("uncatalyzed/jy.npy"),
            np.load("catalyzed/jy.npy"),
        ]
    )

    fig, axes = plt.subplots(3, 5, figsize=(6, 5), sharex="col", sharey="row", constrained_layout=True)

    for j in range(5):
        proj = np.sum(rho[j], axis=-1) / area
        pcm = axes[0, j].pcolormesh(xe, ye, proj.T, norm=LogNorm(vmin=1e-6, vmax=1e-2))
    cbar = fig.colorbar(pcm, ax=axes[0])
    cbar.set_label(r"$\rho_{\theta}$")

    for j in range(5):
        proj = np.sum(rho[j] * qp[j], axis=-1) / np.sum(rho[j], axis=-1)
        pcm = axes[1, j].pcolormesh(xe, ye, proj.T, vmin=0, vmax=1)
    cbar = fig.colorbar(pcm, ax=axes[1])
    cbar.set_label(r"$A_{\theta}[q_{+}]$")

    for j in range(5):
        proj = np.sum(rho[j] * tp[j], axis=-1) / np.sum(rho[j], axis=-1)
        pcm = axes[2, j].pcolormesh(xe, ye, proj.T, vmin=0, vmax=8)
    cbar = fig.colorbar(pcm, ax=axes[2])
    cbar.set_label(r"$A_{\theta}[m_{+}]$")

    xj = np.linspace(-1, 3, 4 * 3 + 1)[1:-1]
    yj = np.linspace(-3, 3, 6 * 3 + 1)[1:-1]
    xs, ys = np.meshgrid(xj, yj, indexing="ij")
    xys = np.stack((xs, ys), axis=-1)
    for j in range(5):
        projx = np.sum(jx[j], axis=-1) / area
        projy = np.sum(jy[j], axis=-1) / area
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
                clim=(0, 4.5e-4),
            )

    for i in range(3):
        for j in range(5):
            ax = axes[i, j]
            ax.set_aspect("equal")
            ax.set_xlim(-1, 3)
            ax.set_ylim(-3, 3)
            ax.text(0, 2, "$A$", ha="center", va="center")
            ax.text(0, -2, "$B$", ha="center", va="center")
            ax.text(2, 0, "$C$", ha="center", va="center")

    for j in range(5):
        axes[-1, j].set_xlabel("$x_1$")
    for i in range(3):
        axes[i, 0].set_ylabel("$x_2$")

    axes[0, 0].set_title("Reaction")
    axes[0, 1].set_title("Uncatalyzed")
    axes[0, 2].set_title("$Y_t = 1$")
    axes[0, 3].set_title("Catalyzed\n$Y_t = 2$")
    axes[0, 4].set_title("$Y_t = 3$")

    fig.savefig("fig5.png")


if __name__ == "__main__":
    main()
