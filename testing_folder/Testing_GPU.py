import numpy as np
import tensorflow as tf

print(tf.config.list_physical_devices('GPU'))
quit()
"""
Create a Tensorflow Session
    
"""

gpu_device = tf.config.list_physical_devices('GPU')[0]
tf.config.experimental.set_memory_growth(gpu_device, True)



"""
Train your model
Once you've created your TensorFlow session, you can train your model as you normally would. 
TensorFlow will automatically use your GPU to speed up the training process.
    
"""