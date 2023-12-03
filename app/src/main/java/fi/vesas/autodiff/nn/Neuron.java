package fi.vesas.autodiff.nn;

import java.util.Arrays;

import fi.vesas.autodiff.util.Util;

public class Neuron {
    
    public double [] weights;
    public double bias;
    public double output;

    public double [] inputs;
    public double [] weightGrads;
    public double biasGrad;
    private String label;

    public Neuron(int inputsize, String label) {

        this.label = label;

        this.weights = new double[inputsize];
        this.inputs = new double[inputsize];

        this.weightGrads = new double[inputsize];

        for(int i = 0; i < inputsize; i++) {
            this.weights[i] = Util.rangeRand(-1.0, 1.0);

        }
        this.bias = Util.rangeRand(-1.0, 1.0);
    }

    public double forward(double [] x) {
        // multiply x with weights, add bias, return 

        double activation = bias;
        for (int i = 0; i < x.length; i++) {
            this.inputs[i] = x[i]; // save inputs for backprop  
            activation += x[i] * weights[i];
        }

        output = Tanh.value(activation);

        // System.out.println(String.format(" > activation %.4f output: %.4f", activation, output));
        return output;
    }

    // learningrate is learning rate
    // grad is gradient from previous layer
    public double [] backward(double outputgrad) {

        // tanh gradient

        double [] outdata = new double[weights.length];

        double tanhGrad = 1.0 - output*output; // Math.tanh(output) * Math.tanh(output);
        tanhGrad = tanhGrad * outputgrad;
        // System.out.println("--" + label);
        // System.out.println(String.format(" > output: %.4f", output));
        // System.out.println(String.format("tanh grad: %.4f", tanhGrad));

        // bias gradient
        double bgrad = tanhGrad;
        this.biasGrad = bgrad;

        // System.out.println(String.format("bias grad: %.4f", biasGrad));

        for (int i = 0; i < weights.length; i++) {
            double weightGrad = tanhGrad * weights[i];
            weightGrads[i] = tanhGrad * inputs[i];
            outdata[i] = weightGrad;
            
            // System.out.println(String.format("weightgrad[%d]: %.4f outdata[%d]: %.4f", i, weightGrad, i, inputs[i]));
        }

        return outdata;
    }

    public void updateWeights(double learningRate) {

        System.out.print("weights before: " + Arrays.toString(weights));
        for (int i = 0; i < weights.length; i++) {
            this.weights[i] += (weightGrads[i] * -learningRate);
        }
        System.out.println(" >> weights after: " + Arrays.toString(weights));

        System.out.print("bias before: " + this.bias);
        this.bias += (this.biasGrad * -learningRate);
        System.out.println(" >> bias after: " + this.bias);
    }

    @Override
    public String toString() {
        return "Neuron [weights=" + Arrays.toString(weights) + ", bias=" + bias + ", output=" + output + ", inputs="
                + Arrays.toString(inputs) + ", weightGrads=" + Arrays.toString(weightGrads) + ", biasGrad=" + biasGrad
                + "]";
    }

}
