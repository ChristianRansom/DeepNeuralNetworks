'''
Created on Feb 7, 2019

@author: Christian Ransom
'''
import math

class Perceptron(object):
    '''
    classdocs
    '''


    def __init__(self, inputs, correct_data, threshold):
        '''
        Constructor
        '''
        self.inputs = inputs
        self.correct_data = correct_data
        self.threshold = threshold
        self.output = None
        
        #how much of a factor will it alter the weights when they are wrong... 
        #it should actually start large, and slowly narrow as it learns...
        self.learning_rate = 2 #more like recalculation factor... 
        
    def train(self, inputs, correct_data, iterations):
        '''iterations are how many cycles of training we want to do''' 
        #for each input i its weight will become 
        #W(i) = W(i) + learning rate*(correct_output - actual_output)*input_value, 
        #a = learning rate
        
        test_matrix = self.test_values(inputs)
        #print(test_matrix)
        
        self.inputs = inputs
        self.correct_data = correct_data
        for temp in correct_data:
            self.display_weight(temp)
        
        
        for i in range(iterations): #how many times to iterate through all possible inputs
            for row in range(len(test_matrix)): #all the possible combinations of input
                print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                print("Testing with input values: " + str(test_matrix[row]))
                counter = 0
                for input in self.inputs: #updates input values to next set of test values
                    input.value = test_matrix[row][counter]
                    counter = counter + 1  
                counter = 0
                for input in self.correct_data: #updates correct_data input values to next set of test values
                    input.value = test_matrix[row][counter]
                    counter = counter + 1   
                self.output = self.calc_output(self.inputs) #calculates the output of the whole neuron 
                correct_output = self.calc_output(self.correct_data)
                #calculates weight adjustment based on current output and correct output difference
                for input in self.inputs: 
                    w = input.weight
                    new_weight = w + self.learning_rate * (correct_output - self.output) * input.value
                    input.weight = new_weight
                    
                    self.display_weight(input)
                    print("weight: " + str(w))
                    print("learning rate: " + str(self.learning_rate))
                    print("correct output: " + str(correct_output))
                    print("output: " + str(self.output))
                    print("input value: " + str(input.value))
                    print("new adjusted weight = " + str(new_weight))
                    print("--------------------------------------------")
                
    def calc_output(self, inputs):
        return self.activation_function(self.input_sum(inputs))
    
    @staticmethod
    def input_sum(inputs):
        ''''multiplies the inputs by their weights and then adds them together
        ie linear combination '''
        result = 0
        for input in inputs:
            print(str(input.value) + " * " + str(input.weight))
            result = result + input.value * input.weight
        print("input sum result: " + str(result))
        return result
         
    @staticmethod
    def sigmoid(x):
        # 1 / (1 + 3^(x-1))
        return 1 /(1 + math.pow(math.e, (x * -1)))

    def step_function(self, input_sum):
        bias = self.threshold * -1
        if input_sum + bias > 0:
            return 1
        else:
            return 0
        
    def activation_function(self, input_sum):
        '''This function calculates the proper output for the perecptron
        based on the inputs'''
        x = input_sum - self.threshold
        return self.sigmoid(x) #returns a float between 1 or 0
        #return self.step_function(input_sum) #returns 1 or 0

    def add_input(self, input):
        self.inputs.append(input)

    def test_values(self, inputs):
        '''Generates a matrix of all possible test values'''
        result_matrix = []
        #Counts in binary numbers and adds them then splits the digits to make the matrix
        for i in range(len(inputs) * len(inputs)):
            row = []
            #Used for how many digits the binaries should have
            format_string = "0" + str(len(inputs)) + "b"
            binary_string = format(i, format_string)
            for ch in str(binary_string):
                row.append(int(ch))
            result_matrix.append(row)
        return result_matrix
            
    def display_weight(self, input):
        print("Input - Name: " + str(input.name) + " Weight: " + str(input.weight))
    
class Input(object):
    '''
    classdocs
    '''

    def __init__(self, name, weight):
        '''
        Constructor
        '''
        self.name = name
        self.weight = weight
        self.value = 0