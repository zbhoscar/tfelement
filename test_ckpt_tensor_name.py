from tensorflow.python import pywrap_tensorflow
import os

dir =  '/home/zbh/Dropbox/tf_models/180929175948'
checkpoint_path = os.path.join(dir, "180929175948.ckpt-29400")
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()
for key in var_to_shape_map:
    print("tensor_name: ", key)
    #print(reader.get_tensor(key))
