"""RAR_D_debugging.py Going to see if I can get this to work.

"""

import deepxde as dde
import numpy as np
from deepxde.backend import tf
import time
import pdb

dde.config.set_default_float("float64")
dde.optimizers.config.set_LBFGS_options(maxiter=100)


def gen_testdata():
    data = np.load("src/burgers/Burgers.npz")
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
    data = dde.data.TimePDE(
        geomtime,
        pde,
        [],
        num_domain=NumDomain // 2,
        train_distribution="pseudo",
    )

    net = dde.maps.FNN([2] + [64] * 3 + [1], "tanh", "Glorot normal")

    print(f"anchors size after: {data.anchors}")
    print(f"X_train size after: {data.train_x.shape}")
    print(f"X_train_all size after: {data.train_x_all.shape}")
    def output_transform(x, y):
        return -tf.sin(np.pi * x[:, 0:1]) + (1 - x[:, 0:1] ** 2) * (x[:, 1:]) * y

    net.apply_output_transform(output_transform)

    model = dde.Model(data, net)
    model.compile("adam", lr=0.001)
    model.train(epochs=100)
    model.compile("L-BFGS")  # This seems to be taking more than 1000 even with maxiter in
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
        X_ids = np.random.choice(
            a=len(X), size=((NumDomain//200)), replace=False, p=err_eq_normalized
        )
        X_selected=X[X_ids]
        print("This is still loop #{}, about to add the anchors\n".format(i))
        data.add_anchors(X_selected)
        print(f"X_selected size: {X_selected.shape}")
        print(f"anchors size after: {data.anchors.shape}")
        print(f"X_train size after: {data.train_x.shape}")
        print(f"X_train_all size after: {data.train_x_all.shape}")

        model.compile("adam", lr=0.001)
        model.train(epochs=1000)
        print("L-BFGS going for 1000")
        model.compile("L-BFGS")
        losshistory, train_state = model.train()
        #Inspect Train_state and whether it updates appropriately

        y_pred = model.predict(X_test)
        l2_error = dde.metrics.l2_relative_error(y_true, y_pred)
        error.append(l2_error)
        print("!\nFinished loop #{}\n".format(i))
        print(f"l2_relative_error: {l2_error}")

    error = np.array(error)
    dde.saveplot(losshistory, train_state, issave=False, isplot=True)

    del model
    
    return error, l2_error


if __name__ == "__main__":
    _, _ = main(c=2, k=0)
    # np.savetxt(f"results/raw/error_and_time/time_RAR-D_default3.txt", time_cost)
    # np.savetxt(f"results/raw/error_and_time/error_RAR-D_default3.txt", error)