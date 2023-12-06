package fi.vesas.autodiff.nn;

import java.util.Arrays;

public class DenseLayer {
    
    public Neuron [] neurons;
    private String label;

    public DenseLayer(int inputs, int outputs, String label) {

        this.label = label;

        neurons = new Neuron[outputs];

        for(int i = 0; i < outputs; i++) {
            Neuron neuron = new Neuron(inputs, label + "-n" + i);
            neurons[i] = neuron;
        }
    }

    public void printWeights() {
        for(int i = 0; i < neurons.length; i++) {
            System.out.println("neuron " + i);
            System.out.println(Arrays.toString(neurons[i].weights));
            System.out.println(neurons[i].bias);
        }
    }

    public double [] forward(double [] x) {

        double[] y = new double[neurons.length];

        for(int i = 0; i < neurons.length; i++) {
            y[i] = neurons[i].forward(x);
        }

        return y;
    }

    public double [] backward(double [] outputgrads) {

        // sum outputgrads  
        // double outputgrad = 0.0;
        // for(double d : outputgrads) {
            // outputgrad += d;
        // }

        double [] outgrads = new double[neurons[0].weights.length];

        for(int i = 0; i < outgrads.length; i++) {
            outgrads[i] = 0.0;
        }
        
        for(int i = 0; i < neurons.length; i++) {
            double [] neuronsgrads = neurons[i].backward(outputgrads[i]);

            for(int j = 0; j < neuronsgrads.length; j++) {
                outgrads[j] += neuronsgrads[j];
            }
        }

        return outgrads;
    }

    public void updateWeights(double learningRate) {
        for(Neuron neuron : neurons) {
            neuron.updateWeights(learningRate);
        }
    }

    
}
