'''
Created on Feb 14, 2019

@author: Christian Ransom
'''
import random

class Matrix():
    '''This is a class to help handle network weight operations 
    '''

    def __init__(self, data):
        '''The outer list holds a list for each row. Each row list contains the elements
        in that row
        :param data: a list of lists where each inner list is a row in the matrix
        '''
        #TODO error checks
        
        if isinstance(data[0], list):
            self.data = data #the actual lists of lists storing the numbers
        else: 
            self.data = [data] #a single row matrix
        

    @staticmethod
    def make_matrix(rows, cols):
        result = []
        counter = 0
        for row in range(rows):
            result.append([])
            for col in range(cols):
                result[row].append(random.random()) #randomizes values
                #result[row].append(counter) #defaults all values to 0
                #result[row].append(1) #increments each value
                counter = counter + 1
        return Matrix(result)

    def __repr__(self):
        result = ""
        for row in range(len(self.data)):
            result = result + "[ "
            for col in range(len(self.data[0])):
                result = result + str(self.data[row][col]) + " "
            result = result + "]"
        return result
    
def add(a, b):
    '''Assumes that the dimensions of a and b are the same'''
    result = []
    for row in range(len(a.data)):
        result.append([])
        for col in range(len(a.data[0])):
            result[row].append(a.data[row][col] + b.data[row][col])
    return Matrix(result)

def subtract(a, b):
    '''Assumes that the dimensions of a and b are the same'''
    result = []
    for row in range(len(a.data)):
        result.append([])
        for col in range(len(a.data[0])):
            result[row].append(a.data[row][col] - b.data[row][col])
    return Matrix(result)

def multiply(a, b):
    '''Uses the dot products to calculate the result
    Assumes number of cols in a equals the number of rows in b'''
    result = []
    # iterate through rows of a
    for i in range(len(a.data)):
        result.append([])
        # iterate through columns of b
        for j in range(len(b.data[0])):
            result[i].append(0)
            # iterate through rows of b
            for k in range(len(b.data)):
                result[i][j] = result[i][j] + a.data[i][k] * b.data[k][j]
    return Matrix(result)

def hadamard(a, b):
    '''Hadamard product AKA elementwise product multiplies every element in matrix 
    a by the corresponding element in matrix b
    The dimensions of both a and b need to be equal. The result is a matrix of the
    same dimensions'''
    result = []
    # iterate through rows of a
    for i in range(len(a.data)):
        result.append([]) #creates a new row 
        # iterate through columns of a
        for j in range(len(a.data[0])):
            result[i].append(a.data[i][j] * b.data[i][j])
    return Matrix(result)

def transpose(a):
    '''Transpose swaps the row column position of every element. element[2][3] 
    is moved to [3][2]
    This method is static and does not change the original matrix. '''
    result = []

    for row in range(len(a.data)): # iterate through rows of a
        for col in range(len(a.data[0])): # iterate through elements in row a
            if row == 0:
                result.append([]) #creates a new row 
            result[col].append(a.data[row][col])
    return Matrix(result)
    
def set_one(a):
    result = []
    for row in range(len(a.data)):
        result.append([])
        for col in range(len(a.data[0])):
            result[row].append(1)
    return Matrix(result)

def sum(a):
    result = 0
    for row in range(len(a.data)):
        for col in range(len(a.data[0])):
            result = result + a.data[row][col]
    return result

def scalar(a, scalar):
    result = []
    for row in range(len(a.data)):
        result.append([])
        for col in range(len(a.data[0])):
            result[row].append(a.data[row][col] * scalar)
    return Matrix(result)

#TODO Methods that'll return a requested row or column in a list