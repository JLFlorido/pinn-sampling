Script Description:
   REP*.py = Replacement of points
   REF*.py = Refining of points. Keeping old ones and adding new ones.
   *_errors.py = Saves step, error and error at collocation points throughout simulation.
   *Local.py = Runs locally, parameters defined within code.
   *HPC.py = Set to run on hpc, hyper parameters defined with job script.
   GradCurvEstimates.py was a specific code that plotted the gradient estimates...
   ... at different steps. Was used to evaluate ability of NN to rpedict gradient.
   ResVsError.py produced data to compare error in u to residual before resampling occured.

Current work:
   Modify _errors.py scripts so that they output data in columns. Use np.vstack
   Modify errors to use new ground truth.