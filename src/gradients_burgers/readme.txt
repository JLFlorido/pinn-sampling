Modification of Wu code, using solution gradients.
Started from code in 'burgers' folder:
    Need to check filepaths and filenames are updated to not refer to old folder.
    At some point will have to investigate docopts and adjust this to work in arc.
For now, focus is on using RAD.py:
    Simplify so it runs quickly.
    Figure out how to obtain u-gradients
    Test by trying to plot " after initial training phase.
    Then substitute residual info with u-gradients in resampling.
        " Could get exaggerated u-gradient info input manually to test if point resampling working as intended.