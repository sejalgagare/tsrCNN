

"""
Course:  Convolutional Neural Networks for Image Classification

Description:
Getting full or absolute path to the current directory

File: getting_full_path.py
"""


# Algorithm:
# --> Copy/Paste the file into needed directory
# --> Open Terminal Window or Anaconda Prompt
# --> Activate environment 'cnncpu' or 'cnngpu'
# --> Activate needed directory by command: cd
# --> Run the file by command: python getting_full_path.py
# --> Copy received full path
#
# Result: Full or absolute path to the current directory


# Importing needed library
import os

# Getting full or absolute path to the current directory
# By using 'os.path.dirname(os.path.abspath(__file__))'
# we get full path to the directory in which this file is located
current_dir = os.path.dirname(os.path.abspath(__file__))

print(current_dir)
