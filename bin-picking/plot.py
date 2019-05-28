#!/usr/bin/env python

import rospy
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import MultiArrayDimension
import time

import numpy as np
import matplotlib.pyplot as plt

def callback(data):

    plt.clf() #clear previous data

    #get data from files
    x_0, y_0, z_0 = np.loadtxt('src/bin-picking/training_data/cat_0.dat', delimiter=';', unpack=True)
    x_1, y_1, z_1 = np.loadtxt('src/bin-picking/training_data/cat_1.dat', delimiter=';', unpack=True)
    x_2, y_2, z_2 = np.loadtxt('src/bin-picking/training_data/cat_2.dat', delimiter=';', unpack=True)
    x_3, y_3, z_3 = np.loadtxt('src/bin-picking/training_data/cat_3.dat', delimiter=';', unpack=True)
    x_4, y_4, z_4 = np.loadtxt('src/bin-picking/training_data/cat_4.dat', delimiter=';', unpack=True)
    x_5, y_5, z_5 = np.loadtxt('src/bin-picking/training_data/cat_5.dat', delimiter=';', unpack=True)

    '''
    #plot color and ratio
    plt.scatter(x_0,y_0, label='class_0 (Jolly Cola)', color='orange')
    plt.scatter(x_1,y_1, label='class_1 (Topform)', color='green')
    plt.scatter(x_2,y_2, label='class_2 (Small energy drink)', color='grey')
    plt.scatter(x_3,y_3, label='class_3 (Red Bull)', color='blue')
    plt.scatter(x_4,y_4, label='class_4 (Large energy drink)', color='yellow')
    plt.scatter(x_5,y_5, label='class_5 (Booster)', color='black')
    point = [data.data[0], data.data[1]]
    plt.xlabel('Average hue-value')
    plt.ylabel('Length-width ratio /3.24*100')
    plt.title('Plot of features (color and length-width-ratio)')
    plt.legend(loc='lower right')
    '''

    #plot color and area
    plt.scatter(x_0,z_0, label='class_0 (Jolly Cola)', color='orange')
    plt.scatter(x_1,z_1, label='class_1 (Topform)', color='green')
    plt.scatter(x_2,z_2, label='class_2 (Small energy drink)', color='grey')
    plt.scatter(x_3,z_3, label='class_3 (Red Bull)', color='blue')
    plt.scatter(x_4,z_4, label='class_4 (Large energy drink)', color='yellow')
    plt.scatter(x_5,z_5, label='class_5 (Booster)', color='black')
    point = [data.data[0], data.data[2]]
    plt.xlabel('Average hue-value')
    plt.ylabel('Area of bounding box /12150*100')
    plt.title('Plot of features (color and area)')
    plt.legend(loc='upper right')

    '''
    #plot ratio and area
    plt.scatter(y_0,z_0, label='class_0 (Jolly Cola)', color='orange')
    plt.scatter(y_1,z_1, label='class_1 (Topform)', color='green')
    plt.scatter(y_2,z_2, label='class_2 (Small energy drink)', color='grey')
    plt.scatter(y_3,z_3, label='class_3 (Red Bull)', color='blue')
    plt.scatter(y_4,z_4, label='class_4 (Large energy drink)', color='yellow')
    plt.scatter(y_5,z_5, label='class_5 (Booster)', color='black')
    point = [data.data[1], data.data[2]]
    plt.xlabel('Length-width ratio /3.24*100')
    plt.ylabel('Area of bounding box /12150*100')
    plt.title('Plot of features (length-width-ratio and area)')
    plt.legend(loc='upper left')
    '''

    #display testing point/object/features
    scat_point = plt.scatter(point[0],point[1], label='detected object', color='red', s=100)

    plt.show()

def plotter():
    rospy.init_node('plotter', anonymous=False)

    rospy.Subscriber("/Feat", Float32MultiArray, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    plotter()
