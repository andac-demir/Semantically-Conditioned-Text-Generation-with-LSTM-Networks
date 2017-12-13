'''
LSTM CLASS
'''

import numpy as np

class LSTM:
    def __init__(self, input_size):
        self.hiddenlayer_size = 100 
        self.time_steps = 25 
        self.learn_rate = 0.15 
        self.std_weight = 0.1 
        self.input_size = input_size
        self.smooth_loss = -np.log(1.0 / input_size) * self.time_steps

        self.output_size = self.hiddenlayer_size + input_size 
        self.fg_weights = np.random.randn(self.hiddenlayer_size, 
                            self.output_size) * self.std_weight + 0.5
        self.fg_bias = np.zeros((self.hiddenlayer_size, 1))
        self.ig_weights = np.random.randn(self.hiddenlayer_size, 
                            self.output_size) * self.std_weight + 0.5
        self.ig_bias = np.zeros((self.hiddenlayer_size, 1))
        self.cellstate_weights = np.random.randn(self.hiddenlayer_size, 
                            self.output_size) * self.std_weight
        self.cellstate_bias = np.zeros((self.hiddenlayer_size, 1))
        self.og_weights = np.random.randn(self.hiddenlayer_size,
                            self.output_size) * self.std_weight + 0.5
        self.og_bias = np.zeros((self.hiddenlayer_size, 1))
        self.output_weights = np.random.randn(input_size, 
                            self.hiddenlayer_size) * self.std_weight
        self.output_bias = np.zeros((input_size, 1))

        self.fg_weights_der = np.zeros(self.fg_weights.shape)
        self.ig_weights_der = np.zeros(self.ig_weights.shape)
        self.cellstate_weights_der = np.zeros(self.cellstate_weights.shape)
        self.og_weights_der = np.zeros(self.og_weights.shape)
        self.ou_weights_der = np.zeros(self.output_weights.shape)
        self.fg_bias_der = np.zeros(self.fg_bias.shape)
        self.ig_bias_der = np.zeros(self.ig_bias.shape)
        self.cellstate_bias_der = np.zeros(self.cellstate_bias.shape)
        self.og_bias_der = np.zeros(self.og_bias.shape)
        self.ou_bias_der = np.zeros(self.output_bias.shape)

        self.fg_weight_mem = np.zeros(self.fg_weights.shape)
        self.ig_weight_mem = np.zeros(self.ig_weights.shape)
        self.cellstate_weight_mem = np.zeros(self.cellstate_weights.shape)
        self.og_weight_mem = np.zeros(self.og_weights.shape)
        self.ou__weight_mem = np.zeros(self.output_weights.shape)
        self.fg_bias_mem = np.zeros(self.fg_bias.shape)
        self.ig_bias_mem = np.zeros(self.ig_bias.shape)
        self.cellstate_bias_mem = np.zeros(self.cellstate_bias.shape)
        self.og_bias_mem = np.zeros(self.og_bias.shape)
        self.ou_bias_mem = np.zeros(self.output_bias.shape)

        self.gate_weights = [self.fg_weights, self.ig_weights, 
            self.cellstate_weights, self.og_weights, self.output_weights, 
            self.fg_bias, self.ig_bias, self.cellstate_bias, self.og_bias, 
            self.output_bias]

        self.derivative_weights = [self.fg_weights_der, self.ig_weights_der, 
            self.cellstate_weights_der, self.og_weights_der, 
            self.ou_weights_der, self.fg_bias_der, self.ig_bias_der, 
            self.cellstate_bias_der, self.og_bias_der, self.ou_bias_der]

        self.memory_weights = [self.fg_weight_mem, self.ig_weight_mem, 
            self.cellstate_weight_mem, self.og_weight_mem, self.ou__weight_mem,
            self.fg_bias_mem, self.ig_bias_mem, self.cellstate_bias_mem, 
            self.og_bias_mem, self.ou_bias_mem]

        # Stores the values in each time step
        self.input_store = {}
        self.output_store = {}
        self.fg_store = {}
        self.ig_store = {}
        self.candidatecell_store = {}
        self.cellstate_store = {}
        self.og_store = {}
        self.hiddenstate_store = {}
        self.ou_s = {}
        self.softmaxoutput_store = {}


    # Define the utility functions:
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-1 * x))


    def der_sigmoid(self, x):
        return x * (1 - x)


    def tanh(self, x):
        return np.tanh(x)


    def der_tanh(self, x):
        return 1 - x * x


    def forward_propagation(self, x, prev_hiddenstate, prev_cellstate):
        inp = np.row_stack((prev_hiddenstate, x))
        fg = self.sigmoid(np.dot(self.fg_weights, inp) + self.fg_bias)
        ig = self.sigmoid(np.dot(self.ig_weights, inp) + self.ig_bias)
        candidatecell = self.tanh(np.dot(self.cellstate_weights, inp) + 
                                  self.cellstate_bias)
        cellstate = fg * prev_cellstate + ig * candidatecell
        og = self.sigmoid(np.dot(self.og_weights, inp) + self.og_bias)
        hiddenstate = og * self.tanh(cellstate)
        output = np.dot(self.output_weights, hiddenstate) + self.output_bias
        softmaxoutput = np.exp(output) / np.sum(np.exp(output))
        
        return inp, fg, ig, candidatecell, cellstate, og, hiddenstate, output, \
               softmaxoutput


    def backward_propagation(self, target, der_hiddenstate_next, 
                             der_cellstate_next, prev_cellstate, inp, fg, ig,
                             candidatecell, cellstate, og, hiddenstate, output, 
                             softmaxoutput):
        der_output = np.copy(softmaxoutput)
        der_output[target] -= 1
        self.ou_weights_der += np.dot(der_output, hiddenstate.T)
        self.ou_bias_der += der_output
        
        der_hiddenstate = np.dot(self.output_weights.T, der_output)
        der_hiddenstate += der_hiddenstate_next
        der_og = der_hiddenstate * self.tanh(cellstate)
        der_og = self.der_sigmoid(og) * der_og
        self.og_weights_der += np.dot(der_og, inp.T)
        self.og_bias_der += der_og

        der_cellstate = np.copy(der_cellstate_next)
        der_cellstate += der_hiddenstate * og * \
                         self.der_tanh(self.tanh(cellstate))
        der_candidatecell = der_cellstate * ig
        der_candidatecell = der_candidatecell * self.der_tanh(candidatecell)
        self.cellstate_weights_der += np.dot(der_candidatecell, inp.T)
        self.cellstate_bias_der += der_candidatecell

        der_ig = der_cellstate * candidatecell
        der_ig = self.der_sigmoid(ig) * der_ig
        self.ig_weights_der += np.dot(der_ig, inp.T)
        self.ig_bias_der += der_ig

        der_fg = der_cellstate * prev_cellstate
        der_fg = self.der_sigmoid(fg) * der_fg
        self.fg_weights_der += np.dot(der_fg, inp.T)
        self.fg_bias_der += der_fg

        der_input = np.dot(self.fg_weights.T, der_fg) + \
                    np.dot(self.ig_weights.T, der_ig) + \
                    np.dot(self.cellstate_weights.T, der_candidatecell) + \
                    np.dot(self.og_weights.T, der_og)
        der_prev_hiddenstate = der_input[:self.hiddenlayer_size, :]
        der_prev_cellstate = fg * der_cellstate
        
        return der_prev_hiddenstate, der_prev_cellstate


    def forward_backward_propagation(self, inputs, targets, 
                                     prev_hiddenstate, prev_cellstate):
        self.hiddenstate_store[-1] = np.copy(prev_hiddenstate)
        self.cellstate_store[-1] = np.copy(prev_cellstate)

        error = 0
        for t in range(len(inputs)):
            self.input_store[t] = np.zeros((self.input_size, 1))
            self.input_store[t][inputs[t]] = 1 # Input character

            self.output_store[t], self.fg_store[t], self.ig_store[t], \
            self.candidatecell_store[t], self.cellstate_store[t], \
            self.og_store[t], self.hiddenstate_store[t], self.ou_s[t], \
            self.softmaxoutput_store[t] = self.forward_propagation(
                self.input_store[t], self.hiddenstate_store[t - 1], 
                self.cellstate_store[t - 1]) # Forward pass

            error += -np.log(self.softmaxoutput_store[t][targets[t], 0]) 

        for der_weight in [self.fg_weights_der, self.ig_weights_der, 
                       self.cellstate_weights_der, self.og_weights_der,
                       self.ou_weights_der, self.fg_bias_der, 
                       self.ig_bias_der, self.cellstate_bias_der, 
                       self.og_bias_der, self.ou_bias_der]:
            der_weight.fill(0)

        der_hiddenstate_next = np.zeros_like(self.hiddenstate_store[0]) 
        der_cellstate_next = np.zeros_like(self.cellstate_store[0])

        for t in reversed(range(len(inputs))):
            der_hiddenstate_next, der_cellstate_next = self.backward_propagation(
                               target = targets[t], 
                               der_hiddenstate_next = der_hiddenstate_next,
                               der_cellstate_next = der_cellstate_next, 
                               prev_cellstate = self.cellstate_store[t-1], 
                               inp = self.output_store[t], 
                               fg = self.fg_store[t], 
                               ig = self.ig_store[t], 
                               candidatecell = self.candidatecell_store[t], 
                               cellstate = self.cellstate_store[t], 
                               og = self.og_store[t], 
                               hiddenstate = self.hiddenstate_store[t], 
                               output = self.ou_s[t], 
                               softmaxoutput = self.softmaxoutput_store[t])

        for der_weight in [self.fg_weights_der, self.ig_weights_der, 
                           self.cellstate_weights_der, self.og_weights_der,
                           self.ou_weights_der, self.fg_bias_der, 
                           self.ig_bias_der, self.cellstate_bias_der, 
                           self.og_bias_der, self.ou_bias_der]:
            np.clip(der_weight, -1, 1, out=der_weight)

        return error, self.hiddenstate_store[len(inputs) - 1], \
               self.cellstate_store[len(inputs) - 1]


    def generate(self, prev_hiddenstate, prev_cellstate, first_char_idx, 
                 text_length):
        x = np.zeros((self.input_size, 1))
        x[first_char_idx] = 1
        hidden_state = prev_hiddenstate
        cell_state = prev_cellstate
        indexes = []
        
        for t in range(text_length):
            _, _, _, _, cell_state, _, hidden_state, _, softmax_output = \
                        self.forward_propagation(x, hidden_state, cell_state)
            num = np.random.choice(range(self.input_size), 
                                   p=softmax_output.ravel())
            x = np.zeros((self.input_size, 1))
            x[num] = 1
            indexes.append(num)
        
        return indexes
