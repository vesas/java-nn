package fi.vesas.autodiff.autodiffnn;

import java.util.Arrays;

import fi.vesas.autodiff.autodiffnn.WeightInitializers.WeightInitializerInterface;
import fi.vesas.autodiff.grad.GradNode;

public class DenseLayer {
    
    private int size = 0;
    public Neuron [] neurons;
    private String label;
    private WeightInitializerInterface weightInitializer;

    public DenseLayer(int size) {
        this.size = size;
    }

    public DenseLayer(GradNode [] inputs, int size, String label) {
        this.size = size;
        this.label = label;
        initialize(inputs);
    }

    public void setLabel(String label) {
        this.label = label;
    }

    public void setWeightInitializer(WeightInitializerInterface weightInitializer) {
        this.weightInitializer = weightInitializer;
    }

    public void initialize(GradNode [] inputs) {

        this.neurons = new Neuron[size];

        for(int i = 0; i < size; i++) {

            Neuron n = new Neuron( inputs, label + "n" + i);
            neurons[i] = n;
        }

        if(weightInitializer != null) {
            weightInitializer.initializeWeights(neurons, inputs.length);
        }
    }

    public void setActivation(Activation activationProto) {
        for(Neuron neuron : neurons) {

            GradNode adds = neuron.adds;
            Activation activation = activationProto.createInstance();
            activation.setInput(adds);
            neuron.activation = (GradNode)activation;
        }
    }

    public GradNode[] getOutputs() {
        GradNode [] outputs = new GradNode[neurons.length];

        for(int i = 0; i < neurons.length; i++) {
            outputs[i] = neurons[i].getOutput();
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

    public void backward() {
        for (Neuron neuron : neurons) {
            neuron.backward();
        }
    }

    public void recordWeights() {
        for (Neuron neuron : neurons) {
            neuron.recordWeights();
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
