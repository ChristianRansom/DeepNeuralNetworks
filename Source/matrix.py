'''
Created on Feb 14, 2019

@author: Christian Ransom
'''
import random

class Matrix():
    '''This is a class to help handle network weight operations 
    '''

    def __init__(self, data):
        
        self.data = data #the actual lists of lists storing the numbers


    @staticmethod
    def add(a, b):
        '''Assumes that the dimensions of a and b are the same'''
        result = []
        for row in range(len(a.data)):
            result.append([])
            for col in range(len(a.data[0])):
                result[row].append(a.data[row][col] + b.data[row][col])
        return Matrix(result)

    @staticmethod
    def subtract(a, b):
        '''Assumes that the dimensions of a and b are the same'''
        result = []
        for row in range(len(a.data)):
            result.append([])
            for col in range(len(a.data[0])):
                result[row].append(a.data[row][col] - b.data[row][col])
        return Matrix(result)
    
    @staticmethod
    def multiply(a, b):
        '''Assumes number of cols in a equals the number of rows in b'''
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
            result = result + "]\n"
        return result