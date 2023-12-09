
Neural networks with automatic differentiation and backprop in Java from scratch. 

Code has two main parts:

- Automatic differentiation which uses chain rule to calculate gradients for a computation graph. 
- Neural network structure which uses the automatic differentiation (autodiff) to learn.

## Test cases:

In addition to unit tests it's good to have some full functionality test cases of varying difficulty.

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

The following diagram depicts the structure of the neural network which learns the XOR function. The network has 2 input nodes, 2 neurons in the hidden layer and 1 output node.

```mermaid
graph LR;
Input1 --> L1_N1[Hidden layer neuron 1]
Input2 --> L1_N1[Hidden layer neuron 1]
Input1 --> L1_N2[Hidden layer neuron 2]
Input2 --> L1_N2[Hidden layer neuron 2]

L1_N1[Hidden layer neuron 1] --> L2_N1[Output neuron 1]
L1_N2[Hidden layer neuron 2] --> L2_N1[Output neuron 1]
```


The MSE error is plotted here.

![Alt text](./readme_img/xor_error.png "error chart")

## TODO:

- Pluggable loss functions
- Pluggable weight initialization tactics
- Try MNIST. Would be interesting to see if it would be possible to learn recognize the MNIST hand written digits.
- Vectorization. Maybe some simple CPU based (AVX) vectorization at first.
