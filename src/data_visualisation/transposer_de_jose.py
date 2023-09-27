import numpy as np
import os

# fname_steps = "testing_folder/losses/Convergence_2layer_Random_k1c1_N2000_L100_step_hist.txt"
# fname_locale = "testing_folder/losses/Convergence_2layer_Random_k1c1_N2000_L100_error_hist_local.txt"
# fname_globale = "testing_folder/losses/Convergence_2layer_Random_k1c1_N2000_L100_error_hist.txt"

# fname_steps2 = "testing_folder/losses/Convergence_Dense1_Random_k1c1_N2000_L100_step_hist.txt"
# fname_locale2 = "testing_folder/losses/Convergence_Dense1_Random_k1c1_N2000_L100_error_hist_local.txt"
# fname_globale2 = "testing_folder/losses/Convergence_Dense1_Random_k1c1_N2000_L100_error_hist.txt"

fname_steps3 = "testing_folder/losses/RAD_residual_Random_k1.0c1.0_N2000_L100_step_hist.txt"
fname_locale3 = "testing_folder/losses/RAD_residual_Random_k1.0c1.0_N2000_L100_error_hist_local.txt"
fname_globale3 = "testing_folder/losses/RAD_residual_Random_k1.0c1.0_N2000_L100_error_hist.txt"

# step_hist = np.loadtxt(fname_steps)
# error_hist_local = np.loadtxt(fname_locale)
# error_hist = np.loadtxt(fname_globale)

# error_curves = np.column_stack(
#     (
#         np.array(step_hist),
#         np.array(error_hist_local),
#         np.array(error_hist),
#     )
# )

# step_hist2 = np.loadtxt(fname_steps2)
# error_hist_local2 = np.loadtxt(fname_locale2)
# error_hist2 = np.loadtxt(fname_globale2)

# error_curves2 = np.column_stack(
#     (
#         np.array(step_hist2),
#         np.array(error_hist_local2),
#         np.array(error_hist2),
#     )
# )
step_hist3 = np.loadtxt(fname_steps3, delimiter=",", dtype=float)
error_hist_local3 = np.loadtxt(fname_locale3, delimiter=",")
error_hist3 = np.loadtxt(fname_globale3, delimiter=",")

error_curves3 = np.column_stack(
    (
        np.array(step_hist3),
        np.array(error_hist_local3),
        np.array(error_hist3),
    )
)

if not(os.path.exists("results/raw/errors_losses")):
    os.mkdir("results/raw/errors_losses")

# output_fname="./results/raw/errors_losses/REP_2layer_Random_k1c1_N2000_L100_error_curves.txt"
# output_fname2="./results/raw/errors_losses/REP_Dense1_Random_k1c1_N2000_L100_error_curves.txt"
output_fname3="./results/raw/errors_losses/REP_residual_Random_k1.0c1.0_N2000_L100_error_curves.txt"
# np.savetxt(output_fname, error_curves, delimiter=",")
# np.savetxt(output_fname2, error_curves2, delimiter=",")
np.savetxt(output_fname3, error_curves3, delimiter=",")