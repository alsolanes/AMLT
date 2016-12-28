# importing all the required components, you may also use scikit for a direct implementation.
import copy
import math
import random
import time
import sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import decimal


#used for randomising U
global MAX
MAX = 10000.0
#used for end condition
global Epsilon
#Epsilon = 0.00000001
Epsilon = 0.01
def print_matrix(list):
    """
    Prints the matrix in a more reqdable way
    """
    for i in range(0,len(list)):
        print list[i]

def end_conditon(U,U_old):
    """
    This is the end conditions, it happens when the U matrix stops chaning too much with successive iterations.
    """
    global Epsilon
    for i in range(0,len(U)):
        for j in range(0,len(U[0])):
            if abs(U[i][j] - U_old[i][j]) > Epsilon :
                return False
    return True

def initialise_U(data, cluster_number):
    """
    This function would randomis U such that the rows add up to 1. it requires a global MAX.
    """
    global MAX
    U = []
    for i in range(0,len(data)):
        current = []
        rand_sum = 0.0
        for j in range(0,cluster_number):
            dummy = random.randint(1,int(MAX))
            current.append(dummy)
            rand_sum += dummy
        for j in range(0,cluster_number):
            current[j] = current[j] / rand_sum
        U.append(current)
    return U

def distance(point, center):
    """
    This function calculates the distance between 2 points (taken as a list). We are refering to Eucledian Distance.
    """
    if len(point) != len(center):
        return -1
    dummy = 0.0
    for i in range(0,len(point)):
        dummy += abs(point[i] - center[i]) ** 2
    return math.sqrt(dummy)

def normalise_U(U):
    """
    This de-fuzzifies the U, at the end of the clustering. It would assume that the point is a member of the cluster whoes membership is maximum.
    """
    for i in range(0,len(U)):
        maximum = max(U[i])
        for j in range(0,len(U[0])):
            if U[i][j] != maximum:
                U[i][j] = 0
            else:
                U[i][j] = 1
    return U


def FCM(data, cluster_number, m = 2):
    """
    This is the main function, it would calculate the required center, and return the final normalised membership matrix U.
    It's paramaters are the : cluster number and the fuzzifier "m".
    """
    ## initialise the U matrix:
    U = initialise_U(data, cluster_number)
    #print_matrix(U)
    #initilise the loop
    iteration_num = 0

    while (True):
        #create a copy of it, to check the end conditions
        U_old = copy.deepcopy(U)
        # cluster center vector
        C = []
        for j in range(0,cluster_number):
            current_cluster_center = []
            for i in range(0,len(data[0])): #this is the number of dimensions
                dummy_sum_num = 0.0
                dummy_sum_dum = 0.0
                for k in range(0,len(data)):
                    print k
                    dummy_sum_num += (U[k][j] ** m) * data[k][i]
                    dummy_sum_dum += (U[k][j] ** m)
                current_cluster_center.append(dummy_sum_num/dummy_sum_dum)
            C.append(current_cluster_center)

        #creating a distance vector, useful in calculating the U matrix.

        distance_matrix =[]
        for i in range(0,len(data)):
            current = []
            for j in range(0,cluster_number):
                current.append(distance(data[i], C[j]))
            distance_matrix.append(current)

        # update U vector
        for j in range(0, cluster_number):
            for i in range(0, len(data)):
                dummy = 0.0
                for k in range(0,cluster_number):
                    dummy += (distance_matrix[i][j]/ distance_matrix[i][k]) ** (2/(m-1))
                U[i][j] = 1 / dummy

        if end_conditon(U,U_old):
            print "finished clustering"
            break
        i+=1
        print 'Iteration:', i
    U = normalise_U(U)
    print "normalised U"
    return U
