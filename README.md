
Neural networks with automatic differentiation and backprop in java. 

## Test cases:

### Flip a sign

Easy test case with only one neuron.

### XOR problem

The XOR function is a simple problem which cannot be solved just by linear separation. 

Exclusive OR (XOR) has the following truth table:

| Input 1  | Input 2 | Output |
| -------- | ------- | ------ |
| 0  | 0    |   0     |
| 0 | 1     |   1    |
| 1    | 0    |  1     |
| 1    | 1    |  0     |

It looks like the the code learns the XOR function with the following structure: Neural network of 2 input nodes, 2 neurons in the hidden layer and 1 output node.

The MSE error is plotted here.

![Alt text](./readme_img/xor_error.png "error chart")

## TODO:

Try:

-MNIST
-Vectorization
