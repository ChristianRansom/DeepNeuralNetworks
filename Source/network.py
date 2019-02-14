'''
Created on Feb 11, 2019

@author: Christian Ransom
'''
import neuron
from abc import ABC, abstractmethod
from _collections import deque
import matrix

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
            if i > 0 and i < len(layout):
                weight_matrix = matrix.Matrix.make_matrix(len(self.layers[i-1]), len(self.layers[i]))
                self.weights.append(weight_matrix)
                #Matrix size = how many neurons in prev layer x neurons in current layer
                
        self.render()
    
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
        
    def render(self):
        self.canvas.update()
        self.canvas.delete("all") #it'd be better to just store each canvas circle object...
        #self.canvas.itemconfigure(self.canvas_frame, width=width, height=event.height)
        print(str(self.canvas.winfo_height()))
        print(str(self.canvas.winfo_width()))
        
        layer_count = self.canvas.winfo_width() / len(self.layers)
        #this should depend on the max number of nodes in a layer
        node_size = self.canvas.winfo_height() / 10 

        for i in range(len(self.layers)): #the number of layers
            w = layer_count * (i) + layer_count / 2
            counter = 0
            for _ in self.layers[i]: #loop through the neurons in this layer
                current_layer_size = len(self.layers[i])
                layer_size = self.canvas.winfo_height() / current_layer_size
                h = layer_size * (counter) + layer_size / 2
                #h = h + (layer_size / 2) * i
                counter = counter + 1
                
                self.canvas.create_oval(w - node_size / 2, h - node_size / 2, 
                                w + node_size / 2, h + node_size / 2, 
                                outline="black", 
                                fill="blue", width=2)     
                
                
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
    
    def __init__(self, canvas):
        '''This is a theoretical situation where we can use a single Neuron to learn to 
        recognize which fighters in a game will be strong enough to win a fight. There are
        many different factors which affect the chance a fighter will win. After lots of 
        labeled training, the Network should be able to accurately predict whether or not
        a fighter will win or lose'''
        self.correct_data = [] #This is the correct output that the network should eventually learn after enough training
        self.correct_data.append(neuron.Input("fast", 5))
        self.correct_data.append(neuron.Input("strong", 7))
        self.correct_data.append(neuron.Input("skilled", 4))
        self.correct_data.append(neuron.Input("tall", 2))
        self.correct_data.append(neuron.Input("intelligent", 10))
        
        
        #the thresholds/biases should actually be implemented randomly as well. 
        #correct data needs to only be a matrix of inputs and a correct output. 
        #we don't need to create a whole network just for the inputs
        threshold = 20 #this is like the totaly power they will need to win
        

        #I'll manually put in incorrect weights for now. Usually its randomly generated 
        input_data = []
        input_data.append(neuron.Input("fast", 0))
        input_data.append(neuron.Input("strong", 0))
        input_data.append(neuron.Input("skilled", 0))
        input_data.append(neuron.Input("tall", 0))
        input_data.append(neuron.Input("intelligent", 0))
        
        #a_neuron = neuron.Neuron(input_data, threshold)
        self.test_matrix = self.test_values(input_data)
        self.test_iterator = 0
        layout = [3, 4, 3, 2] #We'll only have one neuron for this simple network
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
        pass
        
        
    def get_state(self):
        pass