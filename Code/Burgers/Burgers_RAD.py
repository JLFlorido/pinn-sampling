import deepxde as dde
import numpy as np
import tensorflow as tf

dde.config.set_default_float("float64")
dde.optimizers.config.set_LBFGS_options(maxiter=1000)


def gen_testdata():
    data = np.load("Burgers.npz")
    t, x, exact = data["t"], data["x"], data["usol"].T
    xx, tt = np.meshgrid(x, t)
    X = np.vstack((np.ravel(xx), np.ravel(tt))).T
    y = exact.flatten()[:, None]
    return X, y


def main(k, c):
    NumDomain = 2000

    def pde(x, y):
        dy_x = dde.grad.jacobian(y, x, i=0, j=0)
        dy_t = dde.grad.jacobian(y, x, i=0, j=1)
        dy_xx = dde.grad.hessian(y, x, i=0, j=0)
        return dy_t + y * dy_x - 0.01 / np.pi * dy_xx

    X_test, y_true = gen_testdata()

    geom = dde.geometry.Interval(-1, 1)
    timedomain = dde.geometry.TimeDomain(0, 1)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)
    data = dde.data.TimePDE(geomtime, pde, [], num_domain=NumDomain, num_test=10000, train_distribution="pseudo")

    net = dde.maps.FNN([2] + [64] * 3 + [1], "tanh", "Glorot normal")

    def output_transform(x, y):
        return -tf.sin(np.pi * x[:, 0:1]) + (1 - x[:, 0:1] ** 2) * (x[:, 1:]) * y

    net.apply_output_transform(output_transform)

    model = dde.Model(data, net)

    model.compile("adam", lr=0.001)
    model.train(epochs=15000)
    model.compile("L-BFGS")
    model.train()
    y_pred = model.predict(X_test)
    l2_error = dde.metrics.l2_relative_error(y_true, y_pred)
    error = [np.array([l2_error])]
    print(f"l2_relative_error: {l2_error}")

    for i in range(100):
        X = geomtime.random_points(100000)
        Y = np.abs(model.predict(X, operator=pde)).astype(np.float64)
        err_eq = np.power(Y, k) / np.power(Y, k).mean() + c
        err_eq_normalized = (err_eq / sum(err_eq))[:, 0]
        X_ids = np.random.choice(a=len(X), size=NumDomain, replace=False, p=err_eq_normalized)
        X_selected = X[X_ids]
        data.replace_with_anchors(X_selected)

        model.compile("adam", lr=0.001)
        model.train(epochs=1000)
        model.compile("L-BFGS")
        losshistory, train_state = model.train()

        y_pred = model.predict(X_test)
        l2_error = dde.metrics.l2_relative_error(y_true, y_pred)
        error.append(np.array([l2_error]))
        print(f"l2_relative_error: {l2_error}")

    error = np.array(error)
    dde.saveplot(losshistory, train_state, issave=True, isplot=True)
    np.savetxt(f'error_RAD_k_{k}_c_{c}.txt', error)
    return error


if __name__ == "__main__":
    main(c=1, k=1)
