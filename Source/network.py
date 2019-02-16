'''
Created on Feb 11, 2019

@author: Christian Ransom
'''
import neuron
from abc import ABC, abstractmethod
from _collections import deque
import matrix
import copy
from pylint.checkers.variables import overridden_method
import math

class Network():
    '''
    A neural network is a system used for machine learning that has
    been inspired by the brain. Many neurons 
    '''

    def __init__(self, layout, threshold, canvas):
        '''layout: Determines how many layers and how many neurons in each layer.
        Each element in the list indicates how many neurons are in that layer.
        The first element in layout[] represents how man inputs. The last
        element in layout[-1] represents how many outputs. The other elements
        in layout represent how many neurons in the respective hidden layer
        '''
        
        #stores the actual neurons in a nested list. Its a list of lists of each layer 
        self.layers = [] 
        #weights will store a list of Matrix objects
        self.weights = [] #len(weights) will be len(layers) - 1. It will 
        self.canvas = canvas #used for drawing. Expects a tk.Canvas()
        self.threshold = threshold
        self.labels = []
        self.weight_displays = []
        self.build_network(layout)

    def build_network(self, layout):
        '''Builds the network according to the layout specifications
        This method takes care of creating and connecting all the neurons and
        inputs in the network
        This method builds the network from the bottom up, starting with the
        input layer'''


        for i in range(len(layout)): #the lengths of layout should be how many layers
            
            self.layers.append([]) #initialize the next empty layer
            layer = []
            for _ in range(layout[i]): #how many neurons in this layer
                layer.append(neuron.Neuron(self.threshold))

            self.layers[i] = layer 
            if 0 < i < len(layout):
                weight_matrix = matrix.Matrix.make_matrix(len(self.layers[i]), len(self.layers[i-1]))
                self.weights.append(weight_matrix)
                #Matrix size = how many neurons in prev layer x neurons in current layer
                
        self.draw_network()
    
    def train(self, iterations):
        '''
        Eventually I want to be able to start and pause training with events 
        
        1. Get state #initialize inputs 
        2. Get output recursively from the root
        3. Update weights based on the output
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
        #self.canvas.itemconfigure(self.canvas_frame, width=width, text_height=event.text_height)
        layer_width = self.canvas.winfo_width() / len(self.layers)
        #this should depend on the max number of nodes in a layer
        node_size = self.canvas.winfo_height() / 10 
        prev_layer = []
        weight_matrix_counter = 0
        self.weight_displays = []
        for i in range(len(self.layers)): #the number of layers
            w = layer_width * (i) + layer_width / 2
            counter = 0
            current_layer = []
            for _ in self.layers[i]: #loop through the neurons in this layer
                current_layer_size = len(self.layers[i])
                layer_size = self.canvas.winfo_height() / current_layer_size
                h = layer_size * (counter) + layer_size / 2
                counter = counter + 1
                
                current_layer.append(self.canvas.create_oval(w - node_size / 2, h - node_size / 2, 
                                w + node_size / 2, h + node_size / 2, 
                                outline="black", 
                                fill="blue", width=2))
                
            #Draws the lines between the neurons
            if 0 < i < len(self.layers): 
                self.weight_displays.append([])
                for next_neuron in current_layer:
                    for prev_neuron in prev_layer:
                        line_start = self.canvas.coords(prev_neuron) 
                        line_finish = self.canvas.coords(next_neuron)
                        self.canvas.create_line(line_start[0] + node_size / 2, line_start[1] + node_size / 2,
                                                line_finish[0] + node_size / 2, line_finish[1] + node_size / 2)
                        
                        #Adds weight lables on the lines 
                        text_height = line_start[1] + 3 * (line_finish[1] - line_start[1]) / 4 + node_size / 2
                        text = self.canvas.create_text(w - layer_width / 4, text_height, text="0")
                        self.weight_displays[weight_matrix_counter].append(text) #store all text objects
                weight_matrix_counter = weight_matrix_counter + 1
                        
            prev_layer = copy.copy(current_layer)
        self.draw_weights()
            
    def draw_weights(self):
        print(self.weights)
        i = 0
        for weight_matrix in self.weights: #which layer of weights
            j = 0
            for row in range(len(weight_matrix.data)):
                for col in range(len(weight_matrix.data[row])):
                    #print("item id: " + str(self.weight_displays[i][j]))
                    self.canvas.itemconfig(self.weight_displays[i][j], text = '%.2f' % weight_matrix.data[row][col])
                    j = j + 1
            i = i + 1
            
    def print_network(self):
        for i in range(len(self.layers)):
            for a_neuron in self.layers[i]: 
                print("Layer " + str(i) + " " + str(a_neuron))


class Supervised_Network(Network):
    '''
    A supervised network is one where the correct outputs are given
    to train with. The weights are initially randomized and the network
    trains on different inputs and compares its output against the
    provided correct outputs. Every time the network produces a
    wrong output, the network does some weight adjustment to produce 
    more accurate predictions.This is a kind of regression learning.
    This kind of learning is useful for when there is always one and 
    only one correct output for every possible set of inputs. 
    '''    
    
    def __init__(self, layout, canvas):
        '''This is a theoretical situation where we can use a single Neuron to learn to 
        recognize which fighters in a game will be strong enough to win a fight. There are
        many different factors which affect the chance a fighter will win. After lots of 
        labeled training, the Network should be able to accurately predict whether or not
        a fighter will win or lose'''
        self.correct_data = [] #This is the correct output that the network should eventually learn after enough training
        self.test_iterator = 0
        threshold = 20 #this is like the total power they will need to win
        super().__init__(layout, threshold, canvas)
        
        
        
    
        
    def recursive_train(self, root):
        '''
        This method will recurse through the network tree using a depth 
        first search to do a single training iteration using the current 
        input state. The weights will only be adjusted once through this 
        single pass. Input values should already be updated to the current
        state. 
        1. Get state #initialize inputs 
        2. Get output recursively from the root
        3. Update weights based on the output        
        ''' 
        
        #base case: 
        network_output = self.calc_output()
        
        #calculates the output of a neuron with the correct weights
        correct_output = self.calc_output(self.correct_data) 
        
        
        self.adjust_weights(correct_output)
    
    def calc_output(self):
        '''this is implemented iteratively instead of recursively so that the 
        stack size wont limit the size of the networks. This method uses a 
        depth first search post order
        '''
        stack1 = deque()
        stack2 = deque() #stack 2 will be filled postorder and processed after the loop
        
        stack1.append(self.root)
        
        while len(stack1) != 0:
            temp = stack1.pop()
            stack2.append(temp)
            for input in temp.inputs:
                if type(input) != neuron.Input: #Only add if we haven't hit the bottom of the stack
                    stack1.append(input)
        
        while len(stack2) != 0:
            temp = stack2.pop() #DFS order is used to calc the output up through the network
            temp.calc_output()
    
    def get_state(self):
        '''Updates the values of the network inputs to match the next testing state'''
        state = self.test_matrix[self.test_iterator] #get current state inputs in a list
        
        for i in range(len(state)):
            self.inputs[i].output = state[i]
        
        #If we've tested all inputs in the test matrix, start over from beginning 
        if(self.test_iterator >= len(self.test_matrix)):
            self.test_iterator = 0
        else: 
            self.test_iterator = self.test_iterator + 1
    
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
        self.train(100) 
        
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