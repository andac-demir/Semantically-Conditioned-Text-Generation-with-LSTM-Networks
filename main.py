import numpy as np
from LSTM.lstm import LSTM
import matplotlib.pyplot as plt


def load_text():
    with open("Hemingway.txt", "r") as text_file:
        dat = text_file.read()
    # Remove non-ascii chars
    dat = ''.join([x for x in dat if ord(x) < 128])
    chars = list(set(dat))
    data_size, input_size = len(dat), len(chars)
    char_to_num = {ch:i for i,ch in enumerate(chars)}
    num_to_char = {i:ch for i,ch in enumerate(chars)}
    return dat, data_size, input_size, char_to_num, num_to_char


dat, data_size, input_size, char_to_num, num_to_char = load_text()
lstm = LSTM(input_size)
text_length = input("Choose the length of the text to be generated: ")


def update_params(inps, prev_hiddenstate, prev_cellstate):
    nums = lstm.generate(prev_hiddenstate, prev_cellstate,
                               inps[0], text_length)
    txt = ''.join(num_to_char[num] for num in nums)
    print("----\n %s" % (txt, ))
    print("iter %d, loss %f" % (i, lstm.smooth_loss))


i, j = 0, 0
eps = 1e-8
errors = []

try:
    while True:
        if j + lstm.time_steps >= data_size or i == 0:
            prev_hiddenstate_grad = np.zeros((lstm.hiddenlayer_size, 1))
            prev_cellstate_grad = np.zeros((lstm.hiddenlayer_size, 1))
            j = 0
    
        inps = [char_to_num[k] for k in dat[j: j + lstm.time_steps]]
        targets = [char_to_num[k] for k in dat[j + 1: j + lstm.time_steps + 1]]
        loss, prev_hiddenstate_grad, prev_cellstate_grad =  \
            lstm.forward_backward_propagation(inps, targets, prev_hiddenstate_grad, 
                                            prev_cellstate_grad)
        lstm.smooth_loss = lstm.smooth_loss * 0.999 + loss * 0.001
        errors.append(lstm.smooth_loss)
        
        if i % 50 == 0:
            update_params(inps, prev_hiddenstate_grad, prev_cellstate_grad)
    
        # Update weights
        for weight, der_weight, mem in zip(lstm.gate_weights, 
                                        lstm.derivative_weights,
                                        lstm.memory_weights):
            mem += der_weight * der_weight # Calculate sum of gradients
            weight += -(lstm.learn_rate * der_weight / np.sqrt(mem + eps))

        j += lstm.time_steps
        i += 1

        if i > 30000 or lstm.smooth_loss < 5:
            print('Iteration Complete!')
            plt.plot(list(xrange(i)), errors)
            plt.set_ylabel('Error')
            plt.set_xlabel('Iteration')   
            plt.title('Error vs Iteration Number')
            plt.show()
            break 

except KeyboardInterrupt:
    print('Interrupted!')
    plt.plot(list(xrange(i)), errors)
    plt.ylabel('Error')
    plt.xlabel('Iteration')
    plt.title('Error vs Iteration Number')   
    plt.show() 

    









