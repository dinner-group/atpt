import numpy as np
import scipy.sparse as sps

from atpt import fdm


def main():
    # parameters
    kT = 1.0
    out_dir = "model"
    np.save(f"{out_dir}/kT.npy", kT)

    # cell edges
    xe = np.linspace(-2.0, 4.0, 61)
    ye = np.linspace(-4.0, 4.0, 81)
    ze = np.linspace(-4.0, 4.0, 81)
    np.save(f"{out_dir}/xe.npy", xe)
    np.save(f"{out_dir}/ye.npy", ye)
    np.save(f"{out_dir}/ze.npy", ze)

    # coordinates
    xc = 0.5 * (xe[1:] + xe[:-1])
    yc = 0.5 * (ye[1:] + ye[:-1])
    zc = 0.5 * (ze[1:] + ze[:-1])
    x, y, z = np.meshgrid(xc, yc, zc, indexing="ij")
    np.save(f"{out_dir}/xc.npy", xc)
    np.save(f"{out_dir}/yc.npy", yc)
    np.save(f"{out_dir}/zc.npy", zc)
    np.save(f"{out_dir}/x.npy", x)
    np.save(f"{out_dir}/y.npy", y)
    np.save(f"{out_dir}/z.npy", z)

    # potential
    box = ((x - 1) / 2) ** 4 + (y / 3) ** 4 + (z / 3) ** 4
    a = -3 * np.exp(-(x**2 + (y - 2) ** 2 + (z - 2) ** 2))
    b = -3 * np.exp(-(x**2 + (y + 2) ** 2 + (z + 2) ** 2))
    c = -np.exp(-((x - 2) ** 2 + y**2))
    s1 = -2 * np.exp(-(x**2 + y**2 + (z - 2) ** 2))
    s2 = -2 * np.exp(-(x**2 + y**2 + (z + 2) ** 2))
    u = 5 * (box + a + b + c + s1 + s2)
    np.save(f"{out_dir}/u.npy", u)

    # change of measure
    w = np.exp(-u / kT)
    w /= np.sum(w)
    np.save(f"{out_dir}/w.npy", w)

    # generator
    gen = fdm.generator_from_potential_3d(u, kT, 0.1, 0.1, 0.1)
    sps.save_npz(f"{out_dir}/gen.npz", gen)

    # states
    in_a = x**2 + (y - 2) ** 2 <= 0.5**2
    in_b = x**2 + (y + 2) ** 2 <= 0.5**2
    in_c = (x - 2) ** 2 + y**2 <= 0.5**2
    np.save(f"{out_dir}/in_a.npy", in_a)
    np.save(f"{out_dir}/in_b.npy", in_b)
    np.save(f"{out_dir}/in_c.npy", in_c)

    # reaction

    out_dir = "reaction"

    in_d = ~(in_a | in_b)

    qm = fdm.backward_feynman_kac(gen, w, in_d, 0.0, np.where(in_a, 1.0, 0.0))
    qp = fdm.forward_feynman_kac(gen, w, in_d, 0.0, np.where(in_b, 1.0, 0.0))
    np.save(f"{out_dir}/qm.npy", qm)
    np.save(f"{out_dir}/qp.npy", qp)

    rate = fdm.rate(gen, qp, qm, w)
    np.save(f"{out_dir}/rate.npy", rate)

    rho = w * qm * qp
    np.save(f"{out_dir}/rho.npy", rho)

    jx = fdm.current(gen, qp, qm, w, x)
    jy = fdm.current(gen, qp, qm, w, y)
    np.save(f"{out_dir}/jx.npy", jx)
    np.save(f"{out_dir}/jy.npy", jy)

    qm_tm = fdm.backward_feynman_kac(gen, w, in_d, qm, np.zeros(in_d.shape))
    qp_tp = fdm.forward_feynman_kac(gen, w, in_d, qp, np.zeros(in_d.shape))
    np.save(f"{out_dir}/tm.npy", qm_tm / qm)
    np.save(f"{out_dir}/tp.npy", qp_tp / qp)

    # uncatalyzed pathway

    out_dir = "uncatalyzed"

    in_d = ~(in_a | in_b | in_c)
    in_abd = in_a | in_b | in_d

    a = np.array([in_a])
    b = np.array([in_b])
    d = np.array([in_d])

    m = np.full((1, 1), 0, dtype=object)
    m[0, 0] = fdm.spouter(gen, np.multiply, in_abd, in_abd)

    pm = fdm.vector_backward_feynman_kac(gen, w, m, d, 0.0, np.where(d, 0.0, 1.0))
    pp = fdm.vector_forward_feynman_kac(gen, w, m, d, 0.0, np.where(d, 0.0, 1.0))
    np.save(f"{out_dir}/pm.npy", pm)
    np.save(f"{out_dir}/pp.npy", pp)

    pm_qm = fdm.vector_backward_feynman_kac(gen, w, m, d, 0.0, np.where(a, 1.0, 0.0))
    pp_qp = fdm.vector_forward_feynman_kac(gen, w, m, d, 0.0, np.where(b, 1.0, 0.0))
    np.save(f"{out_dir}/qm.npy", pm_qm / pm)
    np.save(f"{out_dir}/qp.npy", pp_qp / pp)

    rate = fdm.pathway_rate(gen, pp_qp, pm_qm, w, m)
    np.save(f"{out_dir}/rate.npy", rate)

    rho = w * pm_qm * pp_qp
    np.save(f"{out_dir}/rho.npy", rho)

    jx = fdm.pathway_current(gen, pp_qp, pm_qm, w, m, x)
    jy = fdm.pathway_current(gen, pp_qp, pm_qm, w, m, y)
    np.save(f"{out_dir}/jx.npy", jx)
    np.save(f"{out_dir}/jy.npy", jy)

    pm_qm_tm = fdm.vector_backward_feynman_kac(gen, w, m, d, pm_qm, np.zeros(d.shape))
    pp_qp_tp = fdm.vector_forward_feynman_kac(gen, w, m, d, pp_qp, np.zeros(d.shape))
    np.save(f"{out_dir}/tm.npy", pm_qm_tm / pm_qm)
    np.save(f"{out_dir}/tp.npy", pp_qp_tp / pp_qp)

    # catalyzed pathway

    out_dir = "catalyzed"

    in_d = ~(in_a | in_b | in_c)
    in_abd = in_a | in_b | in_d
    in_cd = in_c | in_d

    zeros = np.zeros_like(in_d)
    a = np.array([in_a, zeros, zeros])
    b = np.array([zeros, zeros, in_b])
    d = np.array([in_d, in_cd, in_d])

    m = np.full((3, 3), 0, dtype=object)
    m[0, 0] = fdm.spouter(gen, np.multiply, in_abd, in_d)
    m[0, 1] = fdm.spouter(gen, np.multiply, in_abd, in_c)
    m[1, 1] = fdm.spouter(gen, np.multiply, in_cd, in_cd)
    m[1, 2] = fdm.spouter(gen, np.multiply, in_c, in_abd)
    m[2, 2] = fdm.spouter(gen, np.multiply, in_d, in_abd)

    pm = fdm.vector_backward_feynman_kac(gen, w, m, d, 0.0, np.where(d, 0.0, 1.0))
    pp = fdm.vector_forward_feynman_kac(gen, w, m, d, 0.0, np.where(d, 0.0, 1.0))
    np.save(f"{out_dir}/pm.npy", pm)
    np.save(f"{out_dir}/pp.npy", pp)

    pm_qm = fdm.vector_backward_feynman_kac(gen, w, m, d, 0.0, np.where(a, 1.0, 0.0))
    pp_qp = fdm.vector_forward_feynman_kac(gen, w, m, d, 0.0, np.where(b, 1.0, 0.0))
    np.save(f"{out_dir}/qm.npy", pm_qm / pm)
    np.save(f"{out_dir}/qp.npy", pp_qp / pp)

    rate = fdm.pathway_rate(gen, pp_qp, pm_qm, w, m)
    np.save(f"{out_dir}/rate.npy", rate)

    rho = w * pm_qm * pp_qp
    np.save(f"{out_dir}/rho.npy", rho)

    jx = fdm.pathway_current(gen, pp_qp, pm_qm, w, m, x)
    jy = fdm.pathway_current(gen, pp_qp, pm_qm, w, m, y)
    np.save(f"{out_dir}/jx.npy", jx)
    np.save(f"{out_dir}/jy.npy", jy)

    pm_qm_tm = fdm.vector_backward_feynman_kac(gen, w, m, d, pm_qm, np.zeros(d.shape))
    pp_qp_tp = fdm.vector_forward_feynman_kac(gen, w, m, d, pp_qp, np.zeros(d.shape))
    np.save(f"{out_dir}/tm.npy", pm_qm_tm / pm_qm)
    np.save(f"{out_dir}/tp.npy", pp_qp_tp / pp_qp)


if __name__ == "__main__":
    main()
