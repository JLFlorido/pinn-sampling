import numpy as np
import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


"""
Train your model
Once you've created your TensorFlow session, you can train your model as you normally would. 
TensorFlow will automatically use your GPU to speed up the training process.
    
"""