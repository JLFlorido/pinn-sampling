import deepxde as dde
import numpy as np
from deepxde.backend import tf
import skopt
from distutils.version import LooseVersion
import matplotlib.pyplot as plt
import time

dde.config.set_default_float("float64")


def quasirandom(n_samples, sampler):
    space = [(-1.0, 1.0), (0.0, 1.0)]
    if sampler == "LHS":
        sampler = skopt.sampler.Lhs(
            lhs_type="centered", criterion="maximin", iterations=1000
        )
    elif sampler == "Halton":
        sampler = skopt.sampler.Halton(min_skip=-1, max_skip=-1)
    elif sampler == "Hammersley":
        sampler = skopt.sampler.Hammersly(min_skip=-1, max_skip=-1)
    elif sampler == "Sobol":
        # Remove the first point [0, 0, ...] and the second point [0.5, 0.5, ...], which
        # are too special and may cause some error.
        if LooseVersion(skopt.__version__) < LooseVersion("0.9"):
            sampler = skopt.sampler.Sobol(min_skip=2, max_skip=2, randomize=False)
        else:
            sampler = skopt.sampler.Sobol(skip=0, randomize=False)
            return np.array(sampler.generate(space, n_samples + 2)[2:])
    return np.array(sampler.generate(space, n_samples))


def gen_testdata():
    data = np.load("src/burgers/Burgers.npz")
    t, x, exact = data["t"], data["x"], data["usol"].T
    xx, tt = np.meshgrid(x, t)
    X = np.vstack((np.ravel(xx), np.ravel(tt))).T
    y = exact.flatten()[:, None]
    return X, y, xx, tt


def main(NumDomain, method):
    def pde(x, y):
        dy_x = dde.grad.jacobian(y, x, i=0, j=0)
        dy_t = dde.grad.jacobian(y, x, i=0, j=1)
        dy_xx = dde.grad.hessian(y, x, i=0, j=0)
        return dy_t + y * dy_x - 0.01 / np.pi * dy_xx

    geom = dde.geometry.Interval(-1, 1)
    timedomain = dde.geometry.TimeDomain(0, 1)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    if method == "Grid":
        data = dde.data.TimePDE(
            geomtime, pde, [], num_domain=NumDomain, train_distribution="uniform"
        )
    elif method == "Random":
        data = dde.data.TimePDE(
            geomtime, pde, [], num_domain=NumDomain, train_distribution="pseudo"
        )
    elif method in ["LHS", "Halton", "Hammersley", "Sobol"]:
        sample_pts = quasirandom(NumDomain, method)
        data = dde.data.TimePDE(
            geomtime,
            pde,
            [],
            num_domain=0,
            train_distribution="uniform",
            anchors=sample_pts,
        )

    net = dde.maps.FNN([2] + [64] * 3 + [1], "tanh", "Glorot normal")

    def output_transform(x, y):
        return -tf.sin(np.pi * x[:, 0:1]) + (1 - x[:, 0:1] ** 2) * (x[:, 1:]) * y

    net.apply_output_transform(output_transform)

    model = dde.Model(data, net)

    model.compile("adam", lr=1e-3)
    model.train(epochs=15000)  # Originally 15000
    model.compile("L-BFGS")
    losshistory, train_state = model.train()

    X, y_true, xx, tt = gen_testdata()  # xx and tt are meshgrid shaped for pcolormesh
    y_pred = model.predict(X)
    error = dde.metrics.l2_relative_error(y_true, y_pred)
    # # plot u(t,x) distribution as a color-map
    # fig = plt.figure(figsize=(10, 8), dpi=50)
    # y_plot = y_pred.reshape(
    #     (100, 256)
    # )  # y is flat for error comparison, needs to be matrix for pcolormesh
    # plt.pcolormesh(tt, xx, y_plot, cmap="rainbow")
    # plt.xlabel("t")
    # plt.ylabel("x")
    # cbar = plt.colorbar(pad=0.05, aspect=10)
    # cbar.set_label("u(t,x)")
    # cbar.mappable.set_clim(-1, 1)
    print("L2 relative error:", error)
    # This function saves and plots data from the history, instead of plotting the
    #  last prediction that is used for the error. This is why the data is at the
    # last training points and not in a grid.
    dde.saveplot(
        losshistory,
        train_state,
        issave=True,
        isplot=False,
        output_dir="results/raw",
    )
    return error


if __name__ == "__main__":
    time_cost = []
    final_errors = []
    for n in range(1):
        start_t = time.time()
        # main(NumDomain=5000, method="Grid")
        # main(NumDomain=5000, method='Random')
        # main(NumDomain=5000, method='LHS')
        # main(NumDomain=5000, method='Halton')
        final_error_only  = main(NumDomain=10000, method="Hammersley")
        # main(NumDomain=5000, method='Sobol')
        final_errors.append(final_error_only)
        time_cost.append((time.time() - start_t))

        print("Finished run #{}".format(n+1))
        print("Time taken: {:.02f}s".format(time.time() - start_t))
        print("\n--------------------------------------------\n")
    np.savetxt(f"results/raw/time_Ham_10k.txt", time_cost)
    np.savetxt(f"results/raw/error_Ham_10k.txt", final_errors)
    print("Files saved")
