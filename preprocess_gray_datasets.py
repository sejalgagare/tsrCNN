

"""
Course:  Convolutional Neural Networks for Image Classification

Section-4
Construct set of datasets with grayscale images

Description:
Produce datasets from prepared ones by applying preprocessing techniques
Save set of processed datasets in grayscale

File: preprocess_gray_datasets.py
"""


# Algorithm:
# --> Setting up full paths

# --> Preprocessing Traffic Signs Dataset
#
# Result: 5 new HDF5 binary files for every processed dataset


# Importing needed libraries
import numpy as np
import h5py
import cv2


"""
Start of:
Setting up full paths
"""

# Full or absolute path to 'Section2' with Custom and CIFAR-10 datasets
# (!) On Windows, the path should look like following:
# r'C:\Users\your_name\PycharmProjects\CNNCourse\Section2'
# or:
# 'C:\\Users\\your_name\\PycharmProjects\\CNNCourse\\Section2'
full_path_to_Section2 = \
    'D:\Programming\Jupiter Notebook\Project\Section2'


# Full or absolute path to 'Section3' with Traffic Signs dataset
# (!) On Windows, the path should look like following:
# r'C:\Users\your_name\PycharmProjects\CNNCourse\Section3'
# or:
# 'C:\\Users\\your_name\\PycharmProjects\\CNNCourse\\Section3'
full_path_to_Section3 = \
    'D:\Programming\Jupiter Notebook\Project\Section3'

"""
End of:
Setting up full paths
"""


"""
Start of:
Preprocessing Traffic Signs Dataset
"""

# Opening saved Traffic Signs Dataset from HDF5 binary file
# Initiating File object
# Opening file in reading mode by 'r'
# (!) On Windows, it might need to change
# this: + '/' +
# to this: + '\' +
# or to this: + '\\' +
with h5py.File(full_path_to_Section3 + '/' + 'dataset_ts.hdf5', 'r') as f:
    # Extracting saved arrays for training by appropriate keys
    # Saving them into new variables
    x_train = f['x_train']  # HDF5 dataset
    y_train = f['y_train']  # HDF5 dataset
    # Converting them into Numpy arrays
    x_train = np.array(x_train)  # Numpy arrays
    y_train = np.array(y_train)  # Numpy arrays

    # Extracting saved arrays for validation by appropriate keys
    # Saving them into new variables
    x_validation = f['x_validation']  # HDF5 dataset
    y_validation = f['y_validation']  # HDF5 dataset
    # Converting them into Numpy arrays
    x_validation = np.array(x_validation)  # Numpy arrays
    y_validation = np.array(y_validation)  # Numpy arrays

    # Extracting saved arrays for testing by appropriate keys
    # Saving them into new variables
    x_test = f['x_test']  # HDF5 dataset
    y_test = f['y_test']  # HDF5 dataset
    # Converting them into Numpy arrays
    x_test = np.array(x_test)  # Numpy arrays
    y_test = np.array(y_test)  # Numpy arrays


# Converting all images to GRAY by OpenCV function
x_train = np.array(
    list(map(lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), x_train)))
x_validation = np.array(
    list(map(lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), x_validation)))
x_test = np.array(
    list(map(lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), x_test)))

# Extending dimension from (n, height, width) to (n, height, width, one channel)
x_train = x_train[:, :, :, np.newaxis]
x_validation = x_validation[:, :, :, np.newaxis]
x_test = x_test[:, :, :, np.newaxis]


# Check point
# Showing shapes of Numpy arrays with GRAY images
print('Numpy arrays of Traffic Signs Dataset')
print(x_train.shape)
print(x_validation.shape)
print(x_test.shape)
print()


# Implementing normalization by dividing images pixels on 255.0
# Purpose: to make computation more efficient by reducing values between 0 and 1
x_train_255 = x_train / 255.0
x_validation_255 = x_validation / 255.0
x_test_255 = x_test / 255.0


# Saving processed Numpy arrays into new HDF5 binary file
# Initiating File object
# Creating file with name 'dataset_ts_gray_255.hdf5'
# Opening it in writing mode by 'w'
# (!) On Windows, it might need to change
# this: + '/' +
# to this: + '\' +
# or to this: + '\\' +
with h5py.File('ts' + '/' + 'dataset_ts_gray_255.hdf5', 'w') as f:
    # Calling methods to create datasets of given shapes and types
    # Saving Numpy arrays for training
    f.create_dataset('x_train', data=x_train_255, dtype='f')
    f.create_dataset('y_train', data=y_train, dtype='i')

    # Saving Numpy arrays for validation
    f.create_dataset('x_validation', data=x_validation_255, dtype='f')
    f.create_dataset('y_validation', data=y_validation, dtype='i')

    # Saving Numpy arrays for testing
    f.create_dataset('x_test', data=x_test_255, dtype='f')
    f.create_dataset('y_test', data=y_test, dtype='i')


# Calculating Mean Image from training dataset
# (!) We calculate Mean Image only from training dataset
# And apply it to all sub-datasets
mean_gray_dataset_ts = np.mean(x_train_255, axis=0)  # (48, 48, 1)

# Implementing normalization by subtracting Mean Image
# Purpose: to centralize the data dispersion around zero, that, in turn,
# is needed for training with respect to learnability and accuracy
# The images themselves are no longer interpretable to human eyes
# Pixels' values are now in some range (from negative to positive),
# where the mean lies at zero
x_train_255_mean = x_train_255 - mean_gray_dataset_ts
x_validation_255_mean = x_validation_255 - mean_gray_dataset_ts
x_test_255_mean = x_test_255 - mean_gray_dataset_ts

# Saving Mean Image to use it later in the course
# Initiating File object
# Creating file with name 'mean_gray_dataset_ts.hdf5'
# Opening it in writing mode by 'w'
# (!) On Windows, it might need to change
# this: + '/' +
# to this: + '\' +
# or to this: + '\\' +
with h5py.File('ts' + '/' + 'mean_gray_dataset_ts.hdf5', 'w') as f:
    # Calling methods to create datasets of given shapes and types
    # Saving Numpy array for Mean Image
    f.create_dataset('mean', data=mean_gray_dataset_ts, dtype='f')


# Saving processed Numpy arrays into new HDF5 binary file
# Initiating File object
# Creating file with name 'dataset_ts_gray_255_mean.hdf5'
# Opening it in writing mode by 'w'
# (!) On Windows, it might need to change
# this: + '/' +
# to this: + '\' +
# or to this: + '\\' +
with h5py.File('ts' + '/' + 'dataset_ts_gray_255_mean.hdf5', 'w') as f:
    # Calling methods to create datasets of given shapes and types
    # Saving Numpy arrays for training
    f.create_dataset('x_train', data=x_train_255_mean, dtype='f')
    f.create_dataset('y_train', data=y_train, dtype='i')

    # Saving Numpy arrays for validation
    f.create_dataset('x_validation', data=x_validation_255_mean, dtype='f')
    f.create_dataset('y_validation', data=y_validation, dtype='i')

    # Saving Numpy arrays for testing
    f.create_dataset('x_test', data=x_test_255_mean, dtype='f')
    f.create_dataset('y_test', data=y_test, dtype='i')


# Calculating Standard Deviation from training dataset
# (!) We calculate Standard Deviation only from training dataset
# And apply it to all sub-datasets
std_gray_dataset_ts = np.std(x_train_255_mean, axis=0)  # (48, 48, 1)

# Implementing preprocessing by dividing on Standard Deviation
# Purpose: to scale pixels' values to a smaller range, that, in turn,
# is needed for training with respect to learnability and accuracy
x_train_255_mean_std = x_train_255_mean / std_gray_dataset_ts
x_validation_255_mean_std = x_validation_255_mean / std_gray_dataset_ts
x_test_255_mean_std = x_test_255_mean / std_gray_dataset_ts

# Saving Standard Deviation to use it later in the course
# Initiating File object
# Creating file with name 'std_gray_dataset_ts.hdf5'
# Opening it in writing mode by 'w'
# (!) On Windows, it might need to change
# this: + '/' +
# to this: + '\' +
# or to this: + '\\' +
with h5py.File('ts' + '/' + 'std_gray_dataset_ts.hdf5', 'w') as f:
    # Calling methods to create datasets of given shapes and types
    # Saving Numpy array for Mean Image
    f.create_dataset('std', data=std_gray_dataset_ts, dtype='f')


# Saving processed Numpy arrays into new HDF5 binary file
# Initiating File object
# Creating file with name 'dataset_ts_gray_255_mean_std.hdf5'
# Opening it in writing mode by 'w'
# (!) On Windows, it might need to change
# this: + '/' +
# to this: + '\' +
# or to this: + '\\' +
with h5py.File('ts' + '/' + 'dataset_ts_gray_255_mean_std.hdf5', 'w') as f:
    # Calling methods to create datasets of given shapes and types
    # Saving Numpy arrays for training
    f.create_dataset('x_train', data=x_train_255_mean_std, dtype='f')
    f.create_dataset('y_train', data=y_train, dtype='i')

    # Saving Numpy arrays for validation
    f.create_dataset('x_validation', data=x_validation_255_mean_std, dtype='f')
    f.create_dataset('y_validation', data=y_validation, dtype='i')

    # Saving Numpy arrays for testing
    f.create_dataset('x_test', data=x_test_255_mean_std, dtype='f')
    f.create_dataset('y_test', data=y_test, dtype='i')


# Check point
# Printing some values from matrices
print('Original:            ', x_train_255[0, 0, :5, 0])
print('- Mean Image:        ', x_train_255_mean[0, 0, :5, 0])
print('/ Standard Deviation:', x_train_255_mean_std[0, 0, :5, 0])
print()

# Check point
# Printing some values of Mean Image and Standard Deviation
print('Mean Image:          ', mean_gray_dataset_ts[0, :5, 0])
print('Standard Deviation:  ', std_gray_dataset_ts[0, :5, 0])
print()


"""
End of:
Preprocessing Traffic Signs Dataset
"""


"""
Some comments
Function 'map' applies function to every item.
More details and examples are here:
print(help(map))
https://docs.python.org/3/library/functions.html#map


Lambda expressions are used to create anonymous functions.
More details and examples are here:
https://docs.python.org/3/reference/expressions.html#lambda
https://docs.python.org/3/tutorial/controlflow.html#lambda-expressions

"""
