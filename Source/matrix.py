'''
Created on Feb 14, 2019

@author: Christian Ransom
'''

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

    def __repr__(self):
        result = ""
        for row in range(len(self.data)):
            result = result + "[ "
            for col in range(len(self.data[0])):
                result = result + str(self.data[row][col]) + " "
            result = result + "]\n"
        return result