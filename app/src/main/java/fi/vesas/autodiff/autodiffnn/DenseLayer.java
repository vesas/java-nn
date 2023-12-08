package fi.vesas.autodiff.autodiffnn;

import java.util.Arrays;

import fi.vesas.autodiff.grad.GradNode;

public class DenseLayer {
    
    public Neuron [] neurons;
    private String label;

    public DenseLayer(GradNode [] inputs, int size, String label) {

        this.label = label;
        this.neurons = new Neuron[size];

        for(int i = 0; i < size; i++) {

            Neuron n = new Neuron( inputs, label + "n" + i);
            neurons[i] = n;
        }
    }

    public GradNode[] getOutputs() {
        GradNode [] outputs = new GradNode[neurons.length];

        for(int i = 0; i < neurons.length; i++) {
            outputs[i] = neurons[i].tanh;
        }

        return outputs;
    }

    public double [] forward() {

        double[] y = new double[neurons.length];

        for(int i = 0; i < neurons.length; i++) {
            y[i] = neurons[i].forward();
        }

        return y;
    }

    public void zeroGrads() {
        for(int i = 0; i < neurons.length; i++) {
            neurons[i].zeroGrads();
        }
	}

    public void backward() {

        for(int i = 0; i < neurons.length; i++) {
            neurons[i].backward();

        }
    }

    public void recordWeights() {
        for(int i = 0; i < neurons.length; i++) {
            neurons[i].recordWeights();

        }
    }

    public void updateWeights(double learningRate) {
        for(Neuron neuron : neurons) {
            neuron.updateWeights(learningRate);
        }
    }

    public void printWeights() {
        for(int i = 0; i < neurons.length; i++) {
            System.out.println("  neuron " + i);
            System.out.println("   " + Arrays.toString(neurons[i].weights));
            System.out.println("    b:" + neurons[i].bias);
        }
    }

    // toString
    @Override
    public String toString() {
        return "DenseLayer " + label + " [neurons=" + Arrays.toString(neurons) + "]";
    }

}
