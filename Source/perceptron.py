'''
Created on Feb 7, 2019

@author: Christian Ransom
'''
import math

class Perceptron(object):
    '''
    A perceptron is a single neuron with multiple inputs and a single output. 
    Each input has a value and a weight for the value. If the sum of the values
    multiplied by their weights is greater than the threshold, then the perceptron
    will output.
    Outputs will always be 1 or 0
    '''

    def __init__(self, inputs, correct_data, threshold):
        '''
        Constructor
        '''
        self.inputs = inputs
        self.correct_data = correct_data
        self.threshold = threshold
        self.output = []
        
        #how much of a factor will it adjust the weights when they are wrong... 
        self.learning_rate = 2 #more like recalculation factor... 
        
    def train(self, iterations):
        '''
        Iterations are how many cycles of training we want to do
        A training iteration will test all possible input values
        and adjust the weights with each test''' 
        
        #generates all possible test values in a matrix
        test_matrix = self.test_values(self.inputs)
        
        for data in self.correct_data:
            self.display_weight(data)
        
        for _ in range(iterations): #how many times to iterate through all possible inputs
            for row in range(len(test_matrix)): #all the possible combinations of input
                print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                print("Testing with input values: " + str(test_matrix[row]))
                
                #updates input values to next set of test values from the test matrix
                self.update_input_values(self.inputs, test_matrix, row)
                self.update_input_values(self.correct_data, test_matrix, row)
                    
                #calculates the output of the whole neuron 
                self.output = self.calc_output(self.inputs) 
                
                #calculates the output of a neuron with the correct weights
                correct_output = self.calc_output(self.correct_data) 
                self.adjust_weights(correct_output)
                
    def update_input_values(self, inputs, matrix, row):
        counter = 0 #keeps track of which test value input we're inputing
        #updates input values to next set of test values from the test matrix
        for input in inputs: 
            input.value = matrix[row][counter]
            counter = counter + 1              
                
                    
    def adjust_weights(self, correct_output):        
        #calculates weight adjustment based on current output and correct output difference
        for input in self.inputs: 
            w = input.weight
            
            #New Weight = Old Weight + learning rate * (correct_output - actual_output) * input_value, 
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
        '''multiplies the inputs by their weights and then adds them together'''
        result = 0
        for input in inputs:
            result = result + input.value * input.weight
        return result
    
    def activation_function(self, input_sum):
        '''
        This function calculates the proper output for the perecptron
        based on the inputs
        '''
        x = input_sum - self.threshold
        return self.sigmoid(x) 
        #return self.step_function(input_sum) #returns 1 or 0
         
    @staticmethod
    def sigmoid(x):
        '''Returns a float between 0 and 1'''
        # 1 / (1 + 3^(x-1))
        return 1 /(1 + math.pow(math.e, (x * -1)))

    def step_function(self, input_sum):
        '''Returns 1 or 0'''
        bias = self.threshold * -1
        if input_sum + bias > 0:
            return 1
        else:
            return 0
        
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
    
    @staticmethod
    def display_weight(input):
        print("Input - Name: " + str(input.name) + " Weight: " + str(input.weight))
    
class Input(object):
    '''
    Weights will always be a positive float
    '''

    def __init__(self, name, weight):
        '''
        Constructor
        '''
        self.name = name
        self.weight = weight
        self.value = 0