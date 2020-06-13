
''' Need test later '''

''' 
Currently these functions are in "train.py", they can be move to util.py later
1. Function to Convert the weights/bias parameters (list of tensors --- trainable_variables) to a flattened 1D Numpy array
2. Function to Convert the calculated gradients (list of tensors --- gradients) to a flattened 1D Numpy array
3. Function to pack the 1D parameters to 2D numpy array
'''

import numpy as np
from pandas import HDFStore


class CL_Utilities:
    def __init__(self):
        pass


    def write_to_hdf5_on_disk(self, file_name, writing_dataframe, episode=None):
        if episode == 0: # create HDFStore container
            print(file_name)
            hdf_container = HDFStore(file_name +'.h5')
            # print('--------------- Container type: {}'.format(type(hdf_container)))
            hdf_container.put(str(episode), writing_dataframe, format='table', data_columns=True)
            hdf_container.close()
        else:
            with HDFStore(file_name +'.h5', mode='a') as store:
                store.append(str(episode), writing_dataframe, append=True, format='table', data_columns=True)

