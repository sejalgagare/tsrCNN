

"""
Course:  Convolutional Neural Networks for Image Classification

Section-3
Convert downloaded dataset to use it for Classification

Description:
Modify images of Traffic Signs to use them for classification
Save prepared dataset

File: convert_ts_dataset.py
"""


# Algorithm:
# --> Cutting Traffic Signs from images
# --> Saving collected Traffic Signs into multiple HDF5 files
# --> Collecting all arrays from created HDF5 files into united Numpy arrays
# --> Shuffling data along the first axis
# --> Splitting arrays into train, validation and test
# --> Saving prepared arrays into one HDF5 binary file
#
# Result: HDF5 binary file with Traffic Sign dataset


# Importing needed libraries
import pandas as pd
import numpy as np
import h5py
import cv2
import os

from sklearn.utils import shuffle
from tqdm import tqdm


"""
Start of:
Cutting Traffic Signs from images
Saving collected Traffic Signs into multiple HDF5 files
"""

# Using method 'os.walk' to iterate all directories and all files
# It starts from specified directory 'GTSRB'
for current_dir, dirs, files in os.walk('GTSRB'):
    # Iterating all files
    for f in files:
        # Checking if filename ends with '.csv'
        if f.endswith('.csv'):
            # Preparing path to current annotation csv file
            # (!) On Windows, it might need to change
            # this: + '/' +
            # to this: + '\' +
            # or to this: + '\\' +
            path_to_annotation = current_dir + '/' + f

            # Getting Pandas dataFrame from current annotation csv file
            a = pd.read_csv(path_to_annotation, sep=';')

            # Getting number of rows from current Pandas dataFrame
            a_rows = a.shape[0]

            # Preparing zero-valued Numpy array for cut objects
            # Shape: image number, height, width, number of channels
            x_train = np.zeros((1, 48, 48, 3))

            # Preparing zero-valued Numpy array for classes' numbers
            # Shape: class's number
            y_train = np.zeros(1)

            # Preparing temp zero-valued Numpy array for current cut object
            # Shape: image number, height, width, number of channels
            x_temp = np.zeros((1, 48, 48, 3))

            # Preparing temp zero-valued Numpy array for class's number
            # Shape: class's number
            y_temp = np.zeros(1)

            # Defining boolean variable to track arrays' shapes
            first_ts = True

            # Iterating all rows from current Pandas dataFrame
            # Wrapping the loop with 'tqdm' in order to see progress in real time
            for i in tqdm(range(a_rows)):
                # Preparing path to current image file
                # (!) On Windows, it might need to change
                # this: + '/' +
                # to this: + '\' +
                # or to this: + '\\' +
                path_to_image = current_dir + '/' + a.loc[i, 'Filename']

                # Reading current image by OpenCV library
                # In this way image is opened already as Numpy array
                # (!) OpenCV by default reads images in BGR order of channels
                image_array = cv2.imread(path_to_image)

                # Swapping channels from BGR to RGB by OpenCV function
                image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

                # Getting class index of current Traffic Sign
                class_index = a.loc[i, 'ClassId']

                # Getting coordinates of current Traffic Sign
                x_min = a.loc[i, 'Roi.X1']
                y_min = a.loc[i, 'Roi.Y1']
                x_max = a.loc[i, 'Roi.X2']
                y_max = a.loc[i, 'Roi.Y2']

                # Cutting Traffic Sign from the image
                cut_ts = image_array[y_min:y_max, x_min:x_max]

                # Resizing cut Traffic Sign to 48 by 48 pixels size
                cut_ts = cv2.resize(cut_ts,
                                    (48, 48),
                                    interpolation=cv2.INTER_CUBIC)

                # Checking if it is the first Traffic Sign for current iteration
                if first_ts:
                    # Assigning to the first position first Traffic Sign
                    x_train[0, :, :, :] = cut_ts

                    # Assigning to the first position its class index
                    y_train[0] = class_index

                    # Changing boolean variable
                    first_ts = False

                # Collecting next Traffic Signs into temp arrays
                # Concatenating arrays vertically
                else:
                    # Assigning to temp array current Traffic Sign
                    x_temp[0, :, :, :] = cut_ts

                    # Assigning to temp array its class index
                    y_temp[0] = class_index

                    # Concatenating vertically temp arrays to main arrays
                    x_train = np.concatenate((x_train, x_temp), axis=0)
                    y_train = np.concatenate((y_train, y_temp), axis=0)

            # Preparing name for the binary file
            # Slicing only name from current csv filename without extension
            file_name = f[:-4]

            # Saving prepared Numpy arrays into HDF5 binary file
            # Initiating File object
            # Creating and opening file in writing mode by 'w'
            # (!) On Windows, it might need to change
            # this: + '/' +
            # to this: + '\' +
            # or to this: + '\\' +
            with h5py.File('GTSRB' + '/' + file_name + '.hdf5', 'w') \
                    as intermediate_f:
                # Calling methods to create datasets of given shapes and types
                # Saving Numpy arrays for training
                intermediate_f.create_dataset('x_train', data=x_train, dtype='f')
                intermediate_f.create_dataset('y_train', data=y_train, dtype='i')

"""
End of:
Cutting Traffic Signs from images
Saving collected Traffic Signs into multiple HDF5 files
"""


"""
Start of:
Collecting all arrays from created HDF5 files into united Numpy arrays
"""

# Opening one of the 44 saved dataset from HDF5 binary file
# Initiating File object
# Opening file in reading mode by 'r'
# (!) On Windows, it might need to change
# this: + '/' +
# to this: + '\' +
# or to this: + '\\' +
with h5py.File('GTSRB/GT-final_test.hdf5', 'r') as f:
    # Extracting saved arrays for training by appropriate keys
    # Saving them into new variables
    x_train = f['x_train']  # HDF5 dataset
    y_train = f['y_train']  # HDF5 dataset

    # Converting them into Numpy arrays
    x_train = np.array(x_train)  # Numpy arrays
    y_train = np.array(y_train)  # Numpy arrays


# Opening rest 43 saved datasets from HDF5 binary files
# Using method 'os.walk' to iterate all directories and all files
# It starts from specified directory 'GTSRB'
for current_dir, dirs, files in os.walk('GTSRB'):
    # Iterating all files
    for f in files:
        # Checking if filename ends with '.hdf5'
        if f.endswith('.hdf5') and f != 'GT-final_test.hdf5':
            # Initiating File object
            # Opening file in reading mode by 'r'
            # (!) On Windows, it might need to change
            # this: + '/' +
            # to this: + '\' +
            # or to this: + '\\' +
            with h5py.File('GTSRB' + '/' + f, 'r') as intermediate_f:
                # Extracting saved arrays for training by appropriate keys
                # Saving them into new variables
                x_temp = intermediate_f['x_train']  # HDF5 dataset
                y_temp = intermediate_f['y_train']  # HDF5 dataset

                # Converting them into Numpy arrays
                x_temp = np.array(x_temp)  # Numpy arrays
                y_temp = np.array(y_temp)  # Numpy arrays

            # Concatenating vertically temp arrays to main arrays
            x_train = np.concatenate((x_train, x_temp), axis=0)
            y_train = np.concatenate((y_train, y_temp), axis=0)

            # Check point
            # Tracking progress of processing current HDF5 file
            print('Done: ', f)

"""
End of:
Collecting all arrays from created HDF5 files into united Numpy arrays
"""


"""
Start of:
Shuffling data along the first axis
"""

# Shuffling data along the first axis
# Saving appropriate connection: image --> label
x_train, y_train = shuffle(x_train, y_train)

"""
End of:
Shuffling data along the first axis
"""


"""
Start of:
Splitting arrays into train, validation and test
"""

# Check point
# Showing total number of collected images
print()
print(x_train.shape)
print(y_train.shape)
print()


# Slicing first 30% of elements from Numpy arrays for training
# Assigning sliced elements to temp Numpy arrays
x_temp = x_train[:int(x_train.shape[0] * 0.3), :, :, :]
y_temp = y_train[:int(y_train.shape[0] * 0.3)]


# Slicing last 70% of elements from Numpy arrays for training
# Re-assigning sliced elements to train Numpy arrays
x_train = x_train[int(x_train.shape[0] * 0.3):, :, :, :]
y_train = y_train[int(y_train.shape[0] * 0.3):]


# Slicing first 80% of elements from temp Numpy arrays
# Assigning sliced elements to validation Numpy arrays
x_validation = x_temp[:int(x_temp.shape[0] * 0.8), :, :, :]
y_validation = y_temp[:int(y_temp.shape[0] * 0.8)]


# Slicing last 20% of elements from temp Numpy arrays
# Assigning sliced elements to test Numpy arrays
x_test = x_temp[int(x_temp.shape[0] * 0.8):, :, :, :]
y_test = y_temp[int(y_temp.shape[0] * 0.8):]

"""
End of:
Splitting arrays into train, validation and test
"""


"""
Start of:
Saving final, prepared Numpy arrays into one HDF5 binary file
"""

# Saving prepared Numpy arrays into HDF5 binary file
# Initiating File object
# Creating file with name 'dataset_ts.hdf5'
# Opening it in writing mode by 'w'
with h5py.File('dataset_ts.hdf5', 'w') as f:
    # Calling methods to create datasets of given shapes and types
    # Saving Numpy arrays for training
    f.create_dataset('x_train', data=x_train, dtype='f')
    f.create_dataset('y_train', data=y_train, dtype='i')

    # Saving Numpy arrays for validation
    f.create_dataset('x_validation', data=x_validation, dtype='f')
    f.create_dataset('y_validation', data=y_validation, dtype='i')

    # Saving Numpy arrays for testing
    f.create_dataset('x_test', data=x_test, dtype='f')
    f.create_dataset('y_test', data=y_test, dtype='i')

"""
End of:
Saving final, prepared Numpy arrays into one HDF5 binary file
"""


"""
Some comments
Function 'cv2.resize' resizes an image down to or up to the specified size.
Interpolations:
    interpolation=cv2.INTER_AREA
        Shrink an image

    interpolation=cv2.INTER_CUBIC
        Enlarge an image

    interpolation=cv2.INTER_LINEAR
        Bilinear interpolation

More details and examples are here:
print(help(cv2.resize))
https://docs.opencv.org/4.3.0/da/d54/group__imgproc__transform.html

"""
