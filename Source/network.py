'''
Created on Feb 11, 2019

@author: Christian Ransom
'''
from abc import ABC, abstractmethod
from _collections import deque
import matrix
import copy
from pylint.checkers.variables import overridden_method
import math
from tkinter import font
from time import sleep
from pylint.test.input.func_noerror_access_attr_before_def_false_positive import Derived

class Network():
    '''
    A neural network is a system used for machine learning that has
    been inspired by the brain. Many neurons 
    '''

    def __init__(self, layout, canvas):
        '''
        @param layout: Determines how many layers and how many neurons in each layer.
        Each element in the list indicates how many neurons are in that layer.
        The first element in layout[] represents how man inputs. The last
        element in layout[-1] represents how many outputs. The other elements
        in layout represent how many neurons in the respective hidden layer
        '''
        
        #stores the actual neurons output in a nested list. Its a list of lists of each layer 
        self.layers = [] 
        #weights will store a list of Matrix objects
        self.weights = [] #len(weights) will be len(layers) - 1
        self.canvas = canvas #used for drawing. Expects a tk.Canvas()
        self.labels = []
        self.weight_displays = []
        self.output_displays = []
        self.biases = []
        self.default_bais = 1 
        self.build_network(layout)

    def build_network(self, layout):
        '''Builds the network according to the layout specifications
        This method takes care of creating and connecting all the neurons and
        inputs in the network
        This method builds the network from the bottom up, starting with the
        input layer'''

        for i in range(len(layout)): #the lengths of layout should be how many layers
            
            self.layers.append([]) #initialize the next empty layer
            self.biases.append([])
            for j in range(layout[i]):
                self.layers[i].append(0)
                self.biases[i].append(self.default_bais)
            if 0 < i < len(self.layers):
                weight_matrix = matrix.Matrix.make_matrix(len(self.layers[i]), len(self.layers[i-1]))
                self.weights.append(weight_matrix)
                #Matrix size = how many neurons in prev layer x neurons in current layer
        self.draw_network()   
                    
    def train(self, iterations):
        '''
        Eventually I want to be able to start and pause training with events 
        
        1. Get state #initialize inputs 
        2. Get output recursively from the root
        3. Calculate the error
        4. Update weights based on the output
        '''
        self.get_state() #Updates the current input values of the network 
        
        for i in range(iterations):
            self.recursive_train(self.root)
    
    
    @abstractmethod
    def get_state(self):
        pass
        
    def draw_network(self):
        self.canvas.update()
        
        self.canvas.delete("all") #it'd be better to just store each canvas circle object...
        #self.canvas.itemconfigure(self.canvas_frame, width=width, weight_text_height=event.weight_text_height)
        layer_width = self.canvas.winfo_width() / len(self.layers)
        #this should depend on the max number of nodes in a layer
        node_size = self.canvas.winfo_height() / 10 
        prev_layer = []
        weight_matrix_counter = 0
        self.weight_displays = []
        self.output_displays = []
        for i in range(len(self.layers)): #the number of layers
            w = layer_width * (i) + layer_width / 2
            counter = 0
            current_layer = []
            self.output_displays.append([])
            for _ in self.layers[i]: #loop through the neurons in this layer
                current_layer_size = len(self.layers[i])
                layer_size = self.canvas.winfo_height() / current_layer_size
                h = layer_size * (counter) + layer_size / 2
                counter = counter + 1

                current_layer.append(self.canvas.create_oval(w - node_size / 2, h - node_size / 2, 
                                w + node_size / 2, h + node_size / 2, 
                                outline="black", 
                                fill="white", width=2))
                a_font = font.Font(family='Helvetica', size=14, weight='bold')
                neuron_text = self.canvas.create_text(w, h, font=a_font, fill="blue", text="0")
                self.output_displays[i].append(neuron_text)

            #Draws the lines between the neurons
            if 0 < i < len(self.layers): 
                self.weight_displays.append([])
                for next_neuron in current_layer:
                    for prev_neuron in prev_layer:
                        line_start = self.canvas.coords(prev_neuron) 
                        line_finish = self.canvas.coords(next_neuron)
                        self.canvas.create_line(line_start[0] + node_size / 2, line_start[1] + node_size / 2,
                                                line_finish[0] + node_size / 2, line_finish[1] + node_size / 2)
                        #Adds weight labels on the lines 
                        weight_text_height = line_start[1] + 3 * (line_finish[1] - line_start[1]) / 4 + node_size / 2
                        weight_text = self.canvas.create_text(w - layer_width / 4, weight_text_height, text="0")
                        self.weight_displays[weight_matrix_counter].append(weight_text) #store all weight_text objects
                weight_matrix_counter = weight_matrix_counter + 1
                        
            prev_layer = copy.copy(current_layer)
        self.draw_weights()
            
    def draw_weights(self):
        #print(self.weights)
        i = 0
        for weight_matrix in self.weights: #which layer of weights
            j = 0
            for row in range(len(weight_matrix.data)):
                for col in range(len(weight_matrix.data[row])):
                    #print("item id: " + str(self.weight_displays[i][j]))
                    self.canvas.itemconfig(self.weight_displays[i][j], text = '%.2f' % weight_matrix.data[row][col])
                    j = j + 1
            i = i + 1
            
    def draw_outputs(self):
        #print("layers: " + str(self.layers))
        #print("outputs displays: " + str(self.output_displays))
        for i in range(len(self.output_displays)): #loop each layer
            for j in range(len(self.output_displays[i])): #loop neurons in this layer
                #print("TEST: " + str(self.layers[i][j]))
                temp = '%.2f' % self.layers[i][j]
                self.canvas.itemconfig(self.output_displays[i][j], text = temp)
    
    def feed_forward(self, inputs):
        '''Loop through layers calculating their outputs and storing the outputs in the neurons
        @param inputs: a matrix object with a single column and a row for each input'''
        
        current_outputs = matrix.transpose(matrix.Matrix([inputs]))
        #print(len(self.weights))
        for i in range(len(self.weights)): #How many weight matrices we have
            #print("weights: " + str(self.weights[i]))
            current_outputs = matrix.multiply(self.weights[i], current_outputs) #Gets the sum of the inputs
            
            #convert baises into a matrix so we can do math
            bias_matrix = matrix.transpose(matrix.Matrix([self.biases[i + 1]])) 
            current_outputs = matrix.add(current_outputs, bias_matrix)
            current_outputs = self.activation_function(current_outputs) 
            #print("current outputs: " + str(matrix.transpose(current_outputs).data))
            #print("self layers: " + str(self.layers))
            self.layers[i+1] = matrix.transpose(current_outputs).data[0]
            
            
        return current_outputs
                    
    def activation_function(self, inputs):
        '''
        This function calculates the proper output for the neuron
        based on the inputs
        @param inputs: a matrix object with only 1 column and a row for each input
        '''
        for i in range(len(inputs.data)):
            inputs.data[i][0] = self.sigmoid(inputs.data[i][0])
        return inputs
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
        
    def print_network(self):
        for i in range(len(self.layers)):
            for j in range(len(self.layers[i])):
                print("Layer " + str(i) + ": "+ str(self.layers[i][j]))


class Supervised_Network(Network):
    '''
    A supervised network is one where the correct outputs are given
    to train with. The weights are initially randomized and the network
    trains on different inputs and compares its output against the
    provided correct outputs. Every time the network produces a
    wrong output, the network does some weight adjustment to produce 
    more accurate predictions.This is a kind of regression learning.
    This kind of learning is useful for when there is always one and 
    only one correct output for every possible set of input. 
    '''    
    
    def __init__(self, layout, test_input, test_output, canvas):
        '''
        :param layout: a list whose length is the number of layers in the network and each 
        value in the list is how many neurons in that layer
        :param test_input: contains a list of lists of inputs to test
        :param test_output: contains a list of correct outputs that corresponds to the inputs
        :param canvas: used for drawing
        '''
        #TODO Error check to make sure test data format matches layout 
        #TODO move test data out to only be passed in as a paramater in the train method
        
        self.test_input = test_input #This is the correct output that the network should eventually learn after enough training
        self.test_output = test_output
        self.learning_rate = 1
        self.test_iterator = 0
        self.targets = []
        
        super().__init__(layout, canvas)
        
    def train(self, iterations = 1):
        '''
        @param correct_data type Matrix: A matrix that maps inputs to outputs with the correct data 
        @param iterations type integer: how many iterations of training and back propagation 
        
        Eventually I want to be able to start and pause training with events 
        '''
        self.draw_network()
        for i in range(iterations):
            '''
            1. Get state and initialize inputs 
            2. Feed Forward - get output 
            3. Back propagate - Update weights based on the output
            '''
            print("\n\n +++++++++++++++++++++Traning Round " + str(i + 1) + " ++++++++++++++++++++++")
            #print(self.layers)
            self.get_state() #Updates the current input values of the network
            outputs = self.feed_forward(self.layers[0])
            self.draw_outputs()
            #print("outputs: " + str(outputs))
            self.back_propagate()
            
            #If we've tested all inputs in the test matrix, start over from beginning 
            if(self.test_iterator >= len(self.test_input) - 1):
                self.test_iterator = 0
            else: 
                self.test_iterator = self.test_iterator + 1     
            self.draw_weights()
            self.draw_outputs()
            #sleep(.05)
    
    def back_propagate(self):
        '''Uses the error_matrix to adjust the weights throughout the network. 
        
        Each weight is adjusted proportionately to how much of an effect it had producing the 
        error_matrix. Or how much the error_matrix changes with respect to changing the weight
        
        https://www.youtube.com/watch?v=zpykfC4VnpM <- explains the backprop formulas used here
        1. Calculate the error_matrix of each Neuron in the network
            1. For Output Layer Neurons: (output-target)
            2. For all other Hidden Layers: SUM  (each (weights from self to next layer * Errors of next layer)           
        
        2. Use the errors to adjust the weights and biases 
            weight change: learning rate * error * output(1 - output) * input
        '''        
        print("\n------------- Begin Back Propagation -------------------")
        print("Inputs: " + str(self.layers[0]))
        print("Targets: " + str(self.targets))
        print("Outputs: " + str(self.layers[-1]))
        
        errors = copy.copy(self.layers) #initialize a matrix the same size as the network
        
        #Calculate the errors for the output layer
        final_outputs = self.layers[-1]
        final_outputs = matrix.Matrix([final_outputs])
        
        #print("targets matrix: " + str(matrix.Matrix([self.targets])))
        error_portions = matrix.subtract(final_outputs, matrix.Matrix([self.targets]))
        #print("error portion " + str(error_portions))
        #print("error portion " + str(error_portions))
        #print("derived_output " + str(derived_output))
        #error_matrix = matrix.hadamard(derived_output, error_portions)
        #print("Start Errors: \n" + str(errors))
        errors[-1] = matrix.transpose(error_portions).data[0]
        #print("Errors: \n" + str(errors))

        print("weights " + str(self.weights))
        
        
        #--------Calculate the errors for the rest of the hidden layers---------        
        #Move backwards through the list starting at last hidden layer
        for i in range(len(self.layers) - 2, -1, -1): 
            
            #This layer error_matrix depends on its output layers error_matrix multiplied by the connecting weights
            #print("i : " + str(i))
            #print(" weight length " + str(len(self.weights)))
            error_portions = matrix.multiply(matrix.transpose(self.weights[i]), error_portions)
            #print("error portion " + str(error_portions))
            errors[i] = matrix.transpose(error_portions).data[0]

        #---------Calculate the weight changes needed for each weight matrix------
        #Move backwards through the layers starting at the output layer and the weights attached to it
        for i in range(len(self.layers) - 1, 0, -1): 
            '''weight change: learning rate * error * output(1-output) * input to this layer'''
            
            #learning rate * layer errors
            error_vector = matrix.transpose(matrix.Matrix([errors[i]]))
            weight_changes = matrix.scalar(error_vector, self.learning_rate)
            
            #Convert layer output into a matrix
            layer_output = matrix.transpose(matrix.Matrix([self.layers[i]]))
            #creates a matrix of ones to help calculate the derivative 
            one = matrix.set_one(copy.copy(layer_output))
            #print("one " + str(one))
            temp =  matrix.subtract(one, layer_output)
            #print("one - next layer " + str(temp))
            #Gets the derivative of the output of this layer
            derived_output = matrix.hadamard(layer_output, temp)
            
            print("derived_output " + str(derived_output))

            #Calculates how much each weight contributed to the  next layers error_matrix portions
            weight_changes = matrix.hadamard(weight_changes, derived_output)
            print("biases " + str(self.biases))
            print("weight_changes " + str(weight_changes.data))
            bias_changes = matrix.subtract(matrix.Matrix([self.biases[i]]), matrix.transpose(weight_changes)) 
            self.biases[i] = bias_changes.data[0]
            print("bias changes " + str(weight_changes.data))
            print("Layer input " + str(self.layers[i-1]))
            weight_changes = matrix.multiply(weight_changes, matrix.Matrix([self.layers[i - 1]]))
            print("Errors: \n" + str(errors))
            print("weight_changes " + str(weight_changes.data))
            
            self.weights[i - 1] = matrix.subtract(self.weights[i - 1], weight_changes)

    def calc_error(self, actual, target):
        ''' Calculates how far off the output of the network is from the correct output 
        @param actual type Matrix: the calculated output of the network 
        @param target type Matrix: the correct expected output of the network based on the inputs 
        '''
        errors = matrix.subtract(target, actual)
        return errors
    
    def get_state(self):
        '''Updates the values of the network inputs to match the next testing state'''
        #state is a list of the inputs
        state = self.test_input[self.test_iterator] #get current state inputs in a list
        self.targets = self.test_output[self.test_iterator]
        
        for i in range(len(self.layers[0])): #lengths of input layer
            self.layers[0][i] = state[i]

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
    
class Single_Neuron_Network(Supervised_Network):
    '''
    This is a simple form of a supervised network that doesn't use any back
    propagation
    '''
    
    def __init__(self, canvas):
        layout = [5, 1] #5 inputs 1 output
        self.test_matrix = []
        self.correct_data = []
        super().__init__(layout, canvas)
    
    def test(self):
        '''This is a theoretical situation where we can use a single Neuron to learn to 
        recognize which fighters in a game will be strong enough to win a fight. There are
        many different factors which affect the chance a fighter will win. After lots of 
        labeled training, the Network should be able to accurately predict whether or not
        a fighter will win or lose'''
        
        '''
        self.correct_data.append(neuron.Input("fast", 5))
        self.correct_data.append(neuron.Input("strong", 7))
        self.correct_data.append(neuron.Input("skilled", 4))
        self.correct_data.append(neuron.Input("tall", 2))
        self.correct_data.append(neuron.Input("intelligent", 10))
        
        #the thresholds/biases should actually be implemented randomly as well. 
        #correct data needs to only be a matrix of inputs and a correct output. 
        #we don't need to create a whole network just for the inputs
        
        #I'll manually put in incorrect weights for now. Usually its randomly generated 
        input_data = []
        input_data.append(neuron.Input("fast", 0))
        input_data.append(neuron.Input("strong", 0))
        input_data.append(neuron.Input("skilled", 0))
        input_data.append(neuron.Input("tall", 0))
        input_data.append(neuron.Input("intelligent", 0))
        #a_neuron = neuron.Neuron(input_data, threshold)
        self.test_matrix = self.test_values(input_data)   
        self.train(100) '''
        
    def train(self, iterations):
        '''A training iteration will test all possible input values
        and adjust the weights with each test
        
        @param iterations: how many cycles of training we want to do
        '''
        
        #generates all possible test values in a matrix
        test_matrix = self.test_values(self.layers[0])
        
        self.draw_weights()
        
        for _ in range(iterations): #how many times to iterate through all possible inputs
            for row in range(len(test_matrix)): #all the possible combinations of input
                print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                print("Testing with input values: " + str(test_matrix[row]))
                0
                #updates input values to next set of test values from the test matrix
                self.update_input_values(self.layers[0], test_matrix, row)
                
                self.update_input_values(self.correct_data, test_matrix, row)
                    
                #calculates the output of the whole neuron 
                self.output = self.calc_output(self.layers[1][0]) #send in output neuron
                
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
                
    def calc_output(self):
        '''This method assumes that the inputs of this neuron are already updated'''
        return self.activation_function(self.input_sum(self.layers[0]))
        
    @staticmethod
    def input_sum(inputs):
        '''multiplies the inputs by their weights and then adds them together'''
        result = 0
        for input in inputs:
            result = result + input.output * input.weight
        return result
    
    def activation_function(self, input_sum):
        '''
        This function calculates the proper output for the neuron
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
                
               
class Reinforcement_Network(Network):
    '''
    A reinforcement network doesn't require sending in all the correct 
    outputs to the network for training. Instead, it requires some form
    of goals for the network to achieve which require a series of multiple
    decisions. When the network does succeed, it uses back-propagation 
    to adjust the weights of the decisions it made that led up to its 
    reward. The same process will happen with punishments. 
    
    This kind of learning is more useful in situations where there isn't 
    always a single correct output for every input state, 
    e.g. a video game like snake. 
    '''        
    def __init__(self, canvas):
        layout = [3, 4, 4, 2] #We'll only have one neuron for this simple network
        super().__init__(layout, canvas)
        
        
    def get_state(self):
        pass