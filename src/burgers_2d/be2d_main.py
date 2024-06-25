"""Run PINN to solve Burger's Equation in 2D using adaptive resampling.
be2d_main.py. 

Usage:
    be2d_main.py [--k=<hyp_k>] [--c=<hyp_c>] [--N=<NumDomain>] [--L=<NumResamples> ] [--IM=<InitialMethod>] [--DEP=<depth>] [--INP1=<input1>]
    be2d_main.py -h | --help
Options:
    -h --help                   Display this help message
    --k=<hyp_k>                 Hyperparameter k [default: 1]
    --c=<hyp_c>                 Hyperparameter c [default: 1]
    --N=<NumDomain>             Number of collocation points for training [default: 1000]
    --L=<NumResamples>          Number of times points are resampled [default: 10]
    --IM=<InitialMethod>        Initial distribution method from: "Grid","Random","LHS", "Halton", "Hammersley", "Sobol" [default: Hammersley]
    --DEP=<depth>               Depth of the network [default: 3]
    --INP1=<input1>             Info source, "uxt", "uxut1" etc... [default: residual]
"""
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Initial imports and some function definitions.
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------

import os
from docopt import docopt
import skopt
from distutils.version import LooseVersion
import deepxde as dde
from deepxde.backend import tf
import numpy as np
import time
import matplotlib.pyplot as plt

# os.environ['DDE_BACKEND'] = 'tensorflow.compat.v1'
# dde.config.set_default_float("float64")
dde.optimizers.config.set_LBFGS_options(maxiter=1000)

def gen_testdata(): # This function opens the ground truth solution at a fixed uniform set of points.
    X = np.load("./src/burgers_2d/X.npy")
    X = X[0,:]
    Y = np.load("./src/burgers_2d/Y.npy")
    Y = Y[:,0]
    T = np.arange(0,1.1,0.1)
    xx,yy,tt = np.meshgrid(X,Y,T)
    xyt = np.vstack((np.ravel(xx), np.ravel(yy),  np.ravel(tt))).T
    results_u = np.load("./src/burgers_2d/results_u.npy")
    u = results_u.flatten()[:,None]
    results_v = np.load("./src/burgers_2d/results_v.npy")
    v = results_v.flatten()[:,None]
    u = np.squeeze(u)
    v = np.squeeze(v)
    
    # indices_x0 = np.where(xyt[:, 0] == 0)[0] # This finds the coordinates where x and y = 0, 1
    # indices_x1 = np.where(xyt[:, 0] == 1)[0]
    # indices_y0 = np.where(xyt[:, 1] == 0)[0]
    # indices_y1 = np.where(xyt[:, 1] == 1)[0]

    return xyt, u, v

def quasirandom(n_samples, sampler): # This function creates pseudorandom distributions if initial method is specified.
    space = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
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

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Main code start
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def main(k=1, c=1, NumDomain=2000, NumResamples=100, method="Random", depth=3, input1="residual"): # Main Code
    print(f"k equals {k}")
    print(f"c equals {c}")
    print(f"NumDomain equals {NumDomain}")
    print(f"Method equals {method}")
    print(f"Depth equals {depth}")
    print(f"Input1 equals {input1}")
    start_t = time.time() #Start time.

    def pde(x, u): # Define Burgers PDE; x has components x,y,t. u has components u and v.
        u_vel, v_vel = u[:, 0:1], u[:, 1:2] # This is needed for calculation of pde_u and pde_v
        du_x = dde.grad.jacobian(u, x, i=0, j=0) # For Jacobian, i dicates u or v, j dictates x, y or t
        du_t = dde.grad.jacobian(u, x, i=0, j=2)
        du_xx = dde.grad.hessian(u, x,component=0, i=0, j=0) # For hessian, component dictates u or v and i,j dictates x,y,t
        du_y = dde.grad.jacobian(u, x, i=0, j=1)
        du_yy = dde.grad.hessian(u, x, component=0, i=1, j=1)
        pde_u = du_t + (u_vel * du_x) + (v_vel * du_y) - (0.01 / np.pi * du_xx) - (0.01 / np.pi * du_yy)
        
        dv_x = dde.grad.jacobian(u, x, i=1, j=0)
        dv_t = dde.grad.jacobian(u, x, i=1, j=2)
        dv_xx = dde.grad.hessian(u, x, component=1, i=0, j=0)
        dv_y = dde.grad.jacobian(u, x, i=1, j=1)
        dv_yy = dde.grad.hessian(u, x, component=1, i=1, j=1)
        pde_v = dv_t + (u_vel * dv_x) + (v_vel * dv_y) - (0.01 / np.pi * dv_xx) - (0.01 / np.pi * dv_yy)
        return [pde_u, pde_v]

    def u_xy(x, u):
        u_vel = u[:, 0:1]
        return dde.grad.hessian(u_vel, x, i=0, j=1)
    def u_xt(x, u):
        u_vel = u[:, 0:1]
        return dde.grad.hessian(u_vel, x, i=0, j=2)
    def u_yt(x, u):
        u_vel = u[:, 0:1]
        return dde.grad.hessian(u_vel, x, i=1, j=2)
    def u_xyt(x, u):
        uxy= u_xy(x, u)
        return dde.grad.jacobian(uxy, x, i=0, j=2)
    def v_xy(x, u):
        v_vel = u[:, 1:2]
        return dde.grad.hessian(v_vel, x, i=0, j=1)
    def v_xt(x, u):
        v_vel = u[:, 1:2]
        return dde.grad.hessian(v_vel, x, i=0, j=2)
    def v_yt(x, u):
        v_vel = u[:, 1:2]
        return dde.grad.hessian(v_vel, x, i=1, j=2)
    def v_xyt(x, u):
        vxy = v_xy(x, u)
        return dde.grad.jacobian(vxy, x, i=0, j=1)
    def pde_u_xy(x, u):
        pde_u = pde(x,u)
        pde_u = pde_u[0]
        return dde.grad.hessian(pde_u, x, component = 0, i=0, j=1)
    def pde_u_xt(x, u):
        pde_u = pde(x,u)
        pde_u = pde_u[0]
        return dde.grad.hessian(pde_u, x, component = 0, i=0, j=2)
    def pde_u_yt(x, u):
        pde_u = pde(x,u)
        pde_u = pde_u[0]
        return dde.grad.hessian(pde_u, x, component = 0, i=1, j=2)

    def pde_u_xyt(x,u):
        pde_uxy = pde_u_xy(x,u)
        return dde.grad.jacobian(pde_uxy,x,i=0,j=2)
    def pde_v_xy(x,u):
        pde_v = pde(x, u)
        pde_v = pde_v[1]
        return dde.grad.hessian(pde_v, x, i=0, j=1)
    def pde_v_xt(x,u):
        pde_v = pde(x, u)
        pde_v = pde_v[1]
        return dde.grad.hessian(pde_v, x, i=0, j=2)
    def pde_v_yt(x,u):
        pde_v = pde(x, u)
        pde_v = pde_v[1]
        return dde.grad.hessian(pde_v, x, i=1, j=2)
    def pde_v_xyt(x,u):
        pde_vxy = pde_v_xy(x, u)
        return dde.grad.jacobian(pde_vxy, x, i=0, j=2)
    
    # This chunk of code describes the problem using dde structure. Varies depending on prescribed initial distribution.
    spacedomain = dde.geometry.Rectangle(xmin=[0, 0],xmax=[1,1]) # The x,y domain is a rectangle with corners 0,0 to 1,1.
    timedomain = dde.geometry.TimeDomain(0, 1) # Time domain is a line from 0 to 1.
    geomtime = dde.geometry.GeometryXTime(spacedomain, timedomain)

    xyt, u_true, v_true = gen_testdata()

    if method == "Grid":
        data = dde.data.TimePDE(
            geomtime, pde, [], num_domain=NumDomain, num_test=10000, train_distribution="uniform"
        )
    elif method == "Random":
        data = dde.data.TimePDE(
            geomtime, pde, [], num_domain=NumDomain, num_test=10000, train_distribution="pseudo"
        )
    elif method in ["LHS", "Halton", "Hammersley", "Sobol"]:
        sample_pts = quasirandom(NumDomain, method)
        data = dde.data.TimePDE(
            geomtime,
            pde,
            [],
            num_domain=0, # Raises eyebrows. Need to check this doesn't cause issue.
            num_test=10000,
            train_distribution="uniform",
            anchors=sample_pts,
        )
    # net = dde.maps.FNN([3] + [64] * depth + [2], "tanh", "Glorot normal") # 3 Input nodes for x,y and t; 2 outputs for u and v.
    net = dde.nn.FNN([3] + [64] * depth + [2], "tanh", "Glorot normal") # 3 Input nodes for x,y and t; 2 outputs for u and v.

    def output_transform(x, y): # BC        
        y1 = y[:, 0:1]
        y2 = y[:, 1:2]
        return tf.concat(
                [
                tf.sin(np.pi * x[:, 0:1]) * tf.cos(np.pi * x[:, 1:2]) + (tf.sin(np.pi * x[:, 0:1])) * (x[:, 2:3])* y1,
                tf.cos(np.pi * x[:, 0:1]) * tf.sin(np.pi * x[:, 1:2]) + (tf.sin(np.pi * x[:, 1:2])) * (x[:, 2:3])* y2
                ],
                axis=1)

    net.apply_output_transform(output_transform)
    
    model = dde.Model(data, net)

    # # Extract current points and plot them - for checking.
    # points_before = data.train_points()
    # points_figure = plt.figure(figsize=(6, 5))
    # ax = points_figure.add_subplot(projection='3d')
    # ax.scatter(points_before[:, 0], points_before[:, 1], points_before[:, 2])
    # plt.show()
    
    # Initial Training before resampling
    print("Initial 15000 Adam steps")
    model.compile("adam", lr=0.001)
    model.train(epochs=15000, display_every=1000)    

    #   # ------ Testing guiding information; the time taken to evaluate a given thing 
    # X = geomtime.random_points(1000)
    # time00 = time.time()
    # Y = np.abs(model.predict(X, operator=pde)).astype(np.float64)
    # time01= time.time()
    # print(f"pde shape check:{Y.shape}")
    # print(f'took {time01-time00}')

    # time10= time.time()
    # Y = np.abs(model.predict(X, operator=pde_v_xy)).astype(np.float64)
    # time11= time.time()
    # print(f"pde_v_xy shape check:{Y.shape}")
    # print(f'took {time11-time10}')
    # time20= time.time()
    # Y = np.abs(model.predict(X, operator=pde_v_xyt)).astype(np.float64)
    # time21= time.time()
    # print(f"pde_v_xyt shape check:{Y.shape}")
    # print(f'took {time21-time20}')
    # quit()
    # # --------------- End of testing code snippet
    # print("Initial L-BFGS steps")
    # model.compile("L-BFGS")
    # model.train(display_every=1000)
    # Measuring error after initial phase. This information is not used by network to train.
    # print(xyt.shape)
    pred = model.predict(xyt)
    u_pred = pred[:,0]
    v_pred = pred[:,1]

    l2_error_u = dde.metrics.l2_relative_error(u_true, u_pred)
    l2_error_v = dde.metrics.l2_relative_error(v_true, v_pred)

    print("Finished initial steps. ")
    print(f"l2_relative_error_u: {l2_error_u}")
    print(f"l2_relative_error_v: {l2_error_v}")

    for i in range(NumResamples): # Resampling loop begins. 100 is default, first run took ~4 hours...
        X = geomtime.random_points(100000)
        # --- Below, all the different info sources for resampling
        if input1 == "residual" or input1 == "pde":
            Y = np.abs(model.predict(X, operator=pde)).astype(np.float64)
            Y = np.add(Y[0,:],Y[1,:])
        elif input1 == "uv_xy":
            Y1 = np.abs(model.predict(X, operator=u_xy)).astype(np.float64)
            Y2 = np.abs(model.predict(X, operator=v_xy)).astype(np.float64)
            Y = np.add(Y1,Y2)
        elif input1 == "uv_xt":
            Y1 = np.abs(model.predict(X, operator=u_xt)).astype(np.float64)
            Y2 = np.abs(model.predict(X, operator=v_xt)).astype(np.float64)
            Y = np.add(Y1,Y2)
        elif input1 == "uv_yt":
            Y1 = np.abs(model.predict(X, operator=u_yt)).astype(np.float64)
            Y2 = np.abs(model.predict(X, operator=v_yt)).astype(np.float64)
            Y = np.add(Y1,Y2)
        elif input1 == "uv_xyt":
            Y1 = np.abs(model.predict(X, operator=u_xyt)).astype(np.float64)
            Y2 = np.abs(model.predict(X, operator=v_xyt)).astype(np.float64)
            Y = np.add(Y1,Y2)
        elif input1 == "pde_uvxy":
            Y1 = np.abs(model.predict(X, operator=pde_u_xy))
            Y2 = np.abs(model.predict(X, operator=pde_v_xy))
            Y = np.add(Y1,Y2)
        elif input1 == "pde_uvxt":
            Y1 = np.abs(model.predict(X, operator=pde_u_xt))
            Y2 = np.abs(model.predict(X, operator=pde_v_xt))
            Y = np.add(Y1,Y2)
        elif input1 == "pde_uvyt":
            Y1 = np.abs(model.predict(X, operator=pde_u_yt))
            Y2 = np.abs(model.predict(X, operator=pde_v_yt))
            Y = np.add(Y1,Y2)
        elif input1 == "pde_uvxyt":
            Y1 = np.abs(model.predict(X, operator=pde_u_xyt))
            Y2 = np.abs(model.predict(X, operator=pde_v_xyt))
            Y = np.add(Y1,Y2)
        
        err_eq = np.power(Y, k) / np.power(Y, k).mean() + c
        err_eq_normalized = (err_eq / sum(err_eq))[:, 0]
        X_ids = np.random.choice(a=len(X), size=NumDomain, replace=False, p=err_eq_normalized)
        X_selected = X[X_ids]

        # # Extract current points and plot them - for checking.
        # points_figure = plt.figure(figsize=(6, 5))
        # ax = points_figure.add_subplot(projection='3d')
        # ax.scatter(X_selected[:, 0], X_selected[:, 1], X_selected[:, 2])
        # plt.show()
        # quit()

        data.replace_with_anchors(X_selected) # Change current points with selected X points

        model.compile("adam", lr=0.001)
        model.train(epochs=1000, display_every=300000)
        model.compile("L-BFGS")
        model.train(display_every=300000)

        print("!\nFinished loop #{}\n".format(i+1))

        pred = model.predict(xyt)
        u_pred = pred[:,0]
        v_pred = pred[:,1]

        l2_error_u = dde.metrics.l2_relative_error(u_true, u_pred)
        l2_error_v = dde.metrics.l2_relative_error(v_true, v_pred)

        print(f"l2_relative_error_u: {l2_error_u}")
        print(f"l2_relative_error_v: {l2_error_v}")

    y_pred = model.predict(xyt)
    u_pred = y_pred[:,0]
    v_pred = y_pred[:,1]
    error_final_u = dde.metrics.l2_relative_error(u_true, u_pred)
    error_final_v = dde.metrics.l2_relative_error(v_true, v_pred)
    print(f"error_final_u is: {error_final_u}")
    print(f"error_final_v is: {error_final_v}")


    # -------------------------------------------------
    # Figures for checking u, v, and bc qualitatively, as error was quite far.
    # Plotting both u and v to check.
    # fig = plt.figure(figsize=(14, 10))

    # ax1 = fig.add_subplot(221, projection='3d')
    # ax1.scatter(xyt[:, 0], xyt[:, 1], xyt[:, 2], c=u_pred)
    # ax1.set_title('u_pred')
    # ax1.set_xlabel('x')
    # ax1.set_ylabel('y')
    # ax1.set_zlabel('t')

    # ax2 = fig.add_subplot(222, projection='3d')
    # ax2.scatter(xyt[:, 0], xyt[:, 1], xyt[:, 2], c=u_true)
    # ax2.set_title('u_true')
    # ax2.set_xlabel('x')
    # ax2.set_ylabel('y')
    # ax2.set_zlabel('t')

    # ax3 = fig.add_subplot(223, projection='3d')
    # ax3.scatter(xyt[:, 0], xyt[:, 1], xyt[:, 2], c=v_pred)
    # ax3.set_title('v_pred')
    # ax3.set_xlabel('x')
    # ax3.set_ylabel('y')
    # ax3.set_zlabel('t')

    # ax4 = fig.add_subplot(224, projection='3d')
    # ax4.scatter(xyt[:, 0], xyt[:, 1], xyt[:, 2], c=v_true)
    # ax4.set_title('v_true')
    # ax4.set_xlabel('x')
    # ax4.set_ylabel('y')
    # ax4.set_zlabel('t')

    # plt.tight_layout()
    # plt.show()

    # Here plotting slices to check B.C.s
    # fig1, axs = plt.subplots(1, 2, figsize=(10, 5))
    # axs[0].scatter(xyt[0::11,0], xyt[0::11,1], c=u_pred[0::11], marker='o')
    # axs[0].set_title('u_pred t=0')
    # axs[0].set_xlabel('X')
    # axs[0].set_ylabel('Y')

    # axs[1].scatter(xyt[0::11,0], xyt[0::11,1], c=u_true[0::11], marker='o')
    # axs[1].set_title('u_true t=0')
    # axs[1].set_xlabel('X')
    # axs[1].set_ylabel('Y')

    # # Second Figure
    # fig2, axs2 = plt.subplots(1, 2, figsize=(10, 5))
    # axs2[0].scatter(xyt[indices_x0,2], xyt[indices_x0,1], c=u_pred[indices_x0], marker='o')
    # axs2[0].set_title('u_pred x=0')
    # axs2[0].set_xlabel('t')
    # axs2[0].set_ylabel('Y')

    # axs2[1].scatter(xyt[indices_x0,2], xyt[indices_x0,1], c=u_true[indices_x0], marker='o')
    # axs2[1].set_title('u_true x=0')
    # axs2[1].set_xlabel('t')
    # axs2[1].set_ylabel('Y')

    # # Third Figure
    # fig3, axs3 = plt.subplots(1, 2, figsize=(10, 5))
    # scatter32 = axs3[0].scatter(xyt[indices_x1,2], xyt[indices_x1,1], c=u_pred[indices_x1], marker='o')
    # cbar32 = plt.colorbar(scatter32, ax=axs3[1])
    # axs3[0].set_title('u_pred x=1')
    # axs3[0].set_xlabel('t')
    # axs3[0].set_ylabel('X')

    # scatter33 = axs3[1].scatter(xyt[indices_x1,2], xyt[indices_x1,1], c=u_true[indices_x1], marker='o')
    # cbar33 = plt.colorbar(scatter33, ax=axs3[1])
    # axs3[1].set_title('u_true x=1')
    # axs3[1].set_xlabel('t')
    # axs3[1].set_ylabel('X')
    
    # plt.tight_layout()
    # plt.show()

    # The following works for X, Y meshgrid, for a specific t. 
    # Would require re-creating the meshgrid from xyt, and appropriately selecting u_pred, so might not be worth the hassle
    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # ax.plot_surface(X, Y, u_pred, cmap='viridis')
    # ax.set_zlim(-1, 1)

    # ax.set_xlabel("x")
    # ax.set_ylabel("y")

    # ax.elev = 35
    # ax.azim = -30
    # plt.show()
    # ------------------------- end of figures.
    time_taken = (time.time()-start_t)
    
    # dde.saveplot(losshistory, train_state, issave=False, isplot=False, 
    #              loss_fname=f"replacement_{input1}_D{depth}_{method}_k{k}c{c}_N{NumDomain}_L{NumResamples}_loss_info.dat", 
    #              train_fname=f"replacement_{input1}_D{depth}_{method}_k{k}c{c}_N{NumDomain}_L{NumResamples}_finalpoints.dat", 
    #              test_fname=f"replacement_{input1}_D{depth}_{method}_k{k}c{c}_N{NumDomain}_L{NumResamples}_finalypred.dat",
    #              output_dir="../results/additional_info")
    return error_final_u, error_final_v, time_taken
 
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Calling main and saving results
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    args = docopt(__doc__)
    c=float(args['--c'])
    k=float(args['--k'])
    NumDomain=int(args['--N'])
    NumResamples=int(args['--L'])
    method=str(args['--IM'])
    depth=int(args["--DEP"])
    input1=str(args["--INP1"])

    # Run main code, save 3 results
    error_final_u, error_final_v, time_taken = main(c=c, k=k, NumDomain=NumDomain,NumResamples=NumResamples,method=method, depth=depth, input1=input1) # Run main, record error history and final accuracy.
    print(f'Time taken was: {time_taken}')
    print(f'Error_u was: {error_final_u}')
    print(f'Error_v was: {error_final_v}')

    # To ensure no error from save.txt
    if np.isscalar(time_taken):
        time_taken = np.atleast_1d(time_taken)
    if np.isscalar(error_final_u):
        error_final_u = np.atleast_1d(error_final_u)
    if np.isscalar(error_final_v):
        error_final_v = np.atleast_1d(error_final_v)

    # Directory to save to
    output_dir = "../pinn-sampling/src/burgers_2d" 
    # File name
    error_u_fname = f"be2d_{input1}_D{depth}_{method}_k{k}c{c}_N{NumDomain}_L{NumResamples}_error_final_u.txt"
    error_v_fname = f"be2d_{input1}_D{depth}_{method}_k{k}c{c}_N{NumDomain}_L{NumResamples}_error_final_v.txt"
    time_taken_fname = f"be2d_{input1}_D{depth}_{method}_k{k}c{c}_N{NumDomain}_L{NumResamples}_time_taken.txt"
    
    # If results directory does not exist, this creates it
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Define the full file paths
    error_u_fname = os.path.join(output_dir, error_u_fname)
    error_v_fname = os.path.join(output_dir, error_v_fname)
    time_taken_fname = os.path.join(output_dir, time_taken_fname)
    
    # Define function to append to file. The try/exception is in case the file was locked for saving by a different job.
    def append_to_file(file_path, data):
        try:    
            with open(file_path, 'ab') as file:
                np.savetxt(file,data)
        except Exception as e:
            print(f"An exception occurred: {e}")
            print("Retrying in 5 seconds...")
            time.sleep(5)
            try:
                with open(file_path, 'ab') as file:
                    np.savetxt(file, data)
            except Exception as e2:
                print(f"An exception occurred again: {e2}")

    # Use function to append to file.
    append_to_file(error_u_fname, error_final_u)
    append_to_file(error_v_fname, error_final_v)
    append_to_file(time_taken_fname, time_taken)