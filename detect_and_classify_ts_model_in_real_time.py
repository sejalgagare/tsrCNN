

"""
Course:  Convolutional Neural Networks for Image Classification

Section-6
Combine: Detection & Classification in Real Time by camera

Description:
Apply simple object detection by colour thresholding in Real Time
Classify detected fragment in Real Time

File: detect_and_classify_ts_model_in_real_time.py
"""


# Algorithm:
# --> Setting up full paths
# --> Loading saved model
# --> Loading and assigning best weights
# --> Preparing labels
# --> Loading saved Mean Image
# --> Preparing function to plot bar chart
# --> Preparing OpenCV windows to be shown
# --> Reading frames from camera in the loop
#     --> Detecting object
#     --> Cutting detected fragment
#         --> Preprocessing cut fragment
#         --> Implementing forward pass
#         --> Showing OpenCV windows with results
#         --> Calculating FPS rate
#
# Result: OpenCV windows with classification results


# Importing needed libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import h5py
import cv2
import io

from keras.models import load_model

from timeit import default_timer as timer


"""
Start of:
Setting up full paths
"""

# Full or absolute path to 'Section3' with labels for Traffic Signs dataset
# (!) On Windows, the path should look like following:
# r'C:\Users\your_name\PycharmProjects\CNNCourse\Section3'
# or:
# 'C:\\Users\\your_name\\PycharmProjects\\CNNCourse\\Section3'
full_path_to_Section3 = \
    'D:\Programming\Jupiter Notebook\Project\Section3'

# Full or absolute path to 'Section4' with preprocessed datasets
# (!) On Windows, the path should look like following:
# r'C:\Users\your_name\PycharmProjects\CNNCourse\Section4'
# or:
# 'C:\\Users\\your_name\\PycharmProjects\\CNNCourse\\Section4'
full_path_to_Section4 = \
    'D:\Programming\Jupiter Notebook\Project\Section4'

# Full or absolute path to 'Section5' with designed models
# (!) On Windows, the path should look like following:
# r'C:\Users\your_name\PycharmProjects\CNNCourse\Section5'
# or:
# 'C:\\Users\\your_name\\PycharmProjects\\CNNCourse\\Section5'
full_path_to_Section5 = \
    'D:\Programming\Jupiter Notebook\Project\Section5'

"""
End of:
Setting up full paths
"""


"""
Start of:
Loading saved model
"""

# Loading model
# (!) On Windows, it might need to change
# this: + '/' +
# to this: + '\' +
# or to this: + '\\' +
model = load_model(full_path_to_Section5 + '/' +
                   'ts' + '/' +
                   'model_1_ts_rgb.h5')

# Check point
print('Model is successfully loaded')

"""
End of:
Loading saved model
"""


"""
Start of:
Loading and assigning best weights
"""

# Loading and assigning best weights
# (!) On Windows, it might need to change
# this: + '/' +
# to this: + '\' +
# or to this: + '\\' +
model.load_weights('ts' + '/' + 'w_1_ts_rgb_255_mean.h5')

# Check point
print('Best weights are loaded and assigned')

"""
End of:
Loading and assigning best weights
"""


"""
Start of:
Preparing labels
"""

# Preparing labels for Traffic Signs dataset
# Getting Pandas dataFrame with labels
# (!) On Windows, it might need to change
# this: + '/' +
# to this: + '\' +
# or to this: + '\\' +
labels = pd.read_csv(full_path_to_Section3 + '/' + 'classes_names.csv', sep=',')

# Converting into Numpy array
labels = np.array(labels.loc[:, 'SignName']).flatten()

# Check point
print('Labels are ready')

"""
End of:
Preparing labels
"""


"""
Start of:
Loading saved Mean Image
"""

# Opening saved Mean Image for RGB Traffic Signs dataset
# Initiating File object
# Opening file in reading mode by 'r'
# (!) On Windows, it might need to change
# this: + '/' +
# to this: + '\' +
# or to this: + '\\' +
with h5py.File(full_path_to_Section4 + '/' +
               'ts' + '/' +
               'mean_rgb_dataset_ts.hdf5', 'r') as f:
    # Extracting saved array for Mean Image
    # Saving it into new variable
    mean_rgb = f['mean']  # HDF5 dataset
    # Converting it into Numpy array
    mean_rgb = np.array(mean_rgb)  # Numpy arrays

"""
End of:
Loading saved Mean Image
"""


"""
Start of:
Preparing function to plot bar chart
"""


# Defining function to plot bar chart with scores values
def bar_chart(obtained_scores):
    # Arranging X axis
    x_positions = np.arange(obtained_scores.size)

    # Creating bar chart
    bars = plt.bar(x_positions, obtained_scores, align='center', alpha=0.6)

    # Highlighting the highest bar
    bars[np.argmax(obtained_scores)].set_color('red')

    # Giving names to axes
    plt.xlabel('Class', fontsize=20)
    plt.ylabel('Value', fontsize=20)

    # Giving name to bar chart
    plt.title('Obtained Scores', fontsize=20)

    # Adjusting borders of the plot
    plt.tight_layout(pad=2.5)

    # Initializing object of the buffer
    b = io.BytesIO()

    # Saving bar chart into the buffer
    plt.savefig(b, format='png', dpi=200)

    # Closing plot with bar chart
    plt.close()

    # Moving pointer to the beginning of the buffer
    b.seek(0)

    # Reading bar chart from the buffer
    bar_image = np.frombuffer(b.getvalue(), dtype=np.uint8)

    # Closing buffer
    b.close()

    # Decoding buffer
    bar_image = cv2.imdecode(bar_image, 1)

    # Returning Numpy array with bar chart
    return bar_image


# Check point
print('Function to plot Bar Chart is successfully defined')


"""
End of:
Preparing function to plot bar chart
"""


"""
Start of:
Preparing OpenCV windows to be shown
"""

# Giving names to the windows
# Specifying that windows are resizable

# Window to show current view from camera in Real Time
cv2.namedWindow('Current view', cv2.WINDOW_NORMAL)

# Window to show cut fragment
cv2.namedWindow('Cut fragment', cv2.WINDOW_NORMAL)

# Window to show classification result
cv2.namedWindow('Classified as', cv2.WINDOW_NORMAL)

# Window to show bar chart with scores
cv2.namedWindow('Scores', cv2.WINDOW_NORMAL)

# Check point
print('OpenCV windows are ready')

"""
End of:
Preparing OpenCV windows to be shown
"""


"""
Start of:
Reading frames from camera in the loop
"""

# Defining 'VideoCapture' object
# to read stream video from camera
# Index of the built-in camera is usually 0
# Try to select other cameras by passing 1, 2, 3, etc.
camera = cv2.VideoCapture(0)

# Defining counter for FPS (Frames Per Second)
counter = 0

# Starting timer for FPS
# Getting current time point in seconds
fps_start = timer()


# Creating image with black background
temp = np.zeros((720, 1280, 3), np.uint8)


# Defining loop to catch frames
while True:
    # Capturing frames one-by-one from camera
    _, frame_bgr = camera.read()

    """
    Start of:
    Detecting object
    """

    # Converting caught frame to HSV colour space
    frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # Applying mask with founded boundary numbers
    mask = cv2.inRange(frame_hsv,
                       (0, 130, 80),
                       (180, 255, 255))

    # Finding contours
    # All found contours are placed into a list
    # Every individual contour is a Numpy array of (x, y) coordinates,
    # that represent boundary points of detected object
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # Sorting contours from biggest to smallest
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    """
    End of:
    Detecting object
    """

    """
    Start of:
    Classifying detected object
    """

    # If any contour is found, extracting coordinates of the biggest one
    if contours:
        # Getting rectangle coordinates and spatial size of the biggest contour
        # Function 'cv2.boundingRect()' returns an approximate rectangle,
        # that covers the region around found contour
        (x_min, y_min, box_width, box_height) = cv2.boundingRect(contours[0])

        # Drawing obtained rectangle on the current BGR frame
        cv2.rectangle(frame_bgr, (x_min, y_min),
                      (x_min + box_width, y_min + box_height),
                      (230, 161, 0), 3)

        # Putting text above rectangle
        cv2.putText(frame_bgr, 'Detected', (x_min - 5, y_min - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (230, 161, 0), 2)

        """
        Start of:
        Cutting detected fragment
        """

        # Cutting detected fragment from BGR frame
        cut_fragment_bgr = frame_bgr[y_min + int(box_height * 0.1):
                                     y_min + box_height - int(box_height * 0.1),
                                     x_min + int(box_width * 0.1):
                                     x_min + box_width - int(box_width * 0.1)]

        """
        End of:
        Cutting detected fragment
        """

        """
        Start of:
        Preprocessing caught frame
        """

        # Swapping channels from BGR to RGB by OpenCV function
        frame_rgb = cv2.cvtColor(cut_fragment_bgr, cv2.COLOR_BGR2RGB)

        # Resizing frame to 48 by 48 pixels size
        frame_rgb = cv2.resize(frame_rgb,
                               (48, 48),
                               interpolation=cv2.INTER_CUBIC)

        # Implementing normalization by dividing image's pixels on 255.0
        frame_rgb_255 = frame_rgb / 255.0

        # Implementing normalization by subtracting Mean Image
        frame_rgb_255_mean = frame_rgb_255 - mean_rgb

        # Extending dimension from (height, width, channels)
        # to (1, height, width, channels)
        frame_rgb_255_mean = frame_rgb_255_mean[np.newaxis, :, :, :]

        """
        End of:
        Preprocessing caught frame
        """

        """
        Start of:
        Implementing forward pass
        """

        # Testing RGB Traffic Signs model trained on dataset:
        # dataset_ts_rgb_255_mean.hdf5
        # Caught frame is preprocessed in the same way
        # Measuring classification time
        start = timer()
        scores = model.predict(frame_rgb_255_mean)
        end = timer()

        # Scores are given as 43 numbers of predictions for each class
        # Getting index of only one class with maximum value
        prediction = np.argmax(scores)

        """
        End of:
        Implementing forward pass
        """

        """
        Start of:
        Showing OpenCV windows
        """

        # Showing current view from camera in Real Time
        # Pay attention! 'cv2.imshow' takes images in BGR format
        cv2.imshow('Current view', frame_bgr)

        # Showing cut fragment
        cv2.imshow('Cut fragment', cut_fragment_bgr)

        # Changing background to BGR(230, 161, 0)
        # B = 230, G = 161, R = 0
        temp[:, :, 0] = 230
        temp[:, :, 1] = 161
        temp[:, :, 2] = 0

        # Adding text with current label
        cv2.putText(temp, labels[int(prediction)], (100, 200),
                    cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

        # Adding text with obtained confidence score to image with label
        cv2.putText(temp, 'Score : ' + '{0:.5f}'.format(scores[0][prediction]),
                    (100, 450), cv2.FONT_HERSHEY_DUPLEX, 4, (255, 255, 255),
                    4, cv2.LINE_AA)

        # Adding text with time spent for classification to image with label
        cv2.putText(temp, 'Time  : ' + '{0:.5f}'.format(end - start),
                    (100, 600), cv2.FONT_HERSHEY_DUPLEX, 4, (255, 255, 255),
                    4, cv2.LINE_AA)

        # Showing image with respect to classification results
        cv2.imshow('Classified as', temp)

        # Showing bar chart
        cv2.imshow('Scores', bar_chart(scores[0]))

        """
        End of:
        Showing OpenCV windows
        """

    # If no contour is found, showing OpenCV windows with information
    else:
        # Showing current view from camera in Real Time
        # Pay attention! 'cv2.imshow' takes images in BGR format
        cv2.imshow('Current view', frame_bgr)

        # Changing background to BGR(230, 161, 0)
        # B = 230, G = 161, R = 0
        temp[:, :, 0] = 230
        temp[:, :, 1] = 161
        temp[:, :, 2] = 0

        # Adding text with information
        cv2.putText(temp, 'No object', (100, 450),
                    cv2.FONT_HERSHEY_DUPLEX, 4, (255, 255, 255), 4, cv2.LINE_AA)

        # Showing information in prepared OpenCV windows
        cv2.imshow('Cut fragment', temp)
        cv2.imshow('Classified as', temp)
        cv2.imshow('Scores', temp)

    """
    End of:
    Classifying detected object
    """

    """
    Start of:
    Calculating FPS
    """

    # Increasing counter for FPS
    counter += 1

    # Stopping timer for FPS
    # Getting current time point in seconds
    fps_stop = timer()

    # Checking if timer reached 1 second
    # Comparing
    if fps_stop - fps_start >= 1.0:
        # Showing FPS rate
        print('FPS rate is: ', counter)

        # Reset FPS counter
        counter = 0

        # Restart timer for FPS
        # Getting current time point in seconds
        fps_start = timer()

    """
    End of:
    Calculating FPS
    """

    # Breaking the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

"""
End of:
Reading frames from camera in the loop
"""


# Releasing camera
camera.release()

# Destroying all opened OpenCV windows
cv2.destroyAllWindows()


"""
Some comments
Simple object detection by colour thresholding with OpenCV function 'cv2.inRange'
More details and examples are here:
https://docs.opencv.org/3.4/da/d97/tutorial_threshold_inRange.html


OpenCV function 'cv2.findContours'
More details and examples are here:
https://docs.opencv.org/4.0.0/d4/d73/tutorial_py_contours_begin.html


'io.BytesIO' object is an in-memory binary stream.
More details and examples are here:
print(help(io))
print(help(io.BytesIO()))
https://docs.python.org/3.7/library/io.html


Function 'np.frombuffer' interprets a buffer as a 1-dimensional array.
More details and examples are here:
print(help(np.frombuffer))
https://numpy.org/doc/stable/reference/generated/numpy.frombuffer.html


Function 'cv2.imdecode' reads image from a buffer.
More details are here:
print(help(cv2.imdecode))


Function 'tight_layout' adjusts padding of the plot.
More details are here:
print(help(plt.tight_layout))
https://matplotlib.org/3.2.1/api/_as_gen/matplotlib.pyplot.tight_layout.html


Function 'cv2.putText' adds text to images.
More details and examples are here:
print(help(cv2.putText))
https://docs.opencv.org/master/dc/da5/tutorial_py_drawing_functions.html


To set height and width of the captured frames add following lines
right after VideoCapture object:

camera = cv2.VideoCapture(0)
camera.set(3, 1280)
camera.set(4, 720)

equivalent is:

camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

More details and examples are here:
print(help(cv2.VideoCapture.set))
https://docs.opencv.org/3.4/d8/dfe/classcv_1_1VideoCapture.html

Pay attention! This properties might work or not.
The property value has to be accepted by the capture device (camera).

"""
