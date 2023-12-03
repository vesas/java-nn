package fi.vesas.autodiff.nn;

import org.junit.jupiter.api.Test;

import fi.vesas.autodiff.nn.DenseLayer;
import fi.vesas.autodiff.nn.Neuron;

public class NeuronTest {
    
    @Test
    public void test1() {

        Neuron n = new Neuron(2, "n1");

        double[] x = {2.0, 0.0};
        double y = 1.0;

        System.out.println("x1: " + x[0]);
        System.out.println("x2: " + x[1]);
        System.out.println("--");

        n.weights[0] = -3.0;
        n.weights[1] = 1.0;
        n.bias = 6.8813735870195432;

        System.out.println("bias: " + n.bias);
        System.out.println("w[0]: " + n.weights[0]);
        System.out.println("w[1]: " + n.weights[1]);

        double prediction = n.forward(x);
        double loss = (prediction - y)*(prediction -y);
        double lossGrad = 2.0 * (prediction - y);

        System.out.println("y: " + y);
        System.out.println("prediction: " + prediction);
        System.out.println("loss: " + loss);
        System.out.println("lossGrad: " + lossGrad);

        // y: 1.0
        // prediction: 0.7071067811865477
        // loss: 0.2928932188134523

        System.out.println("-- backpropagation");
        double learningRate = 0.001;
        // n.backward(prediction);

        if((y-prediction) < 0) {
            n.updateWeights(learningRate);
        }
        else {
            n.updateWeights(-learningRate);
        }

        System.out.println("-- backpropagation");

        System.out.println("bias: " + n.bias);
        System.out.println("w[0]: " + n.weights[0]);
        System.out.println("w[1]: " + n.weights[1]);

        prediction = n.forward(x);
        loss = (prediction - y)*(prediction -y);
        lossGrad = 2.0 * (prediction - y);
        
        System.out.println("y: " + y);
        System.out.println("prediction: " + prediction);
        System.out.println("loss: " + loss);
        System.out.println("lossGrad: " + lossGrad);
        
    }

    @Test
    public void test2() {

        DenseLayer layer = new DenseLayer(2, 3, "l1");

        double[] x = {2.0, 3.0};
        double [] y = layer.forward(x);

        for (double d : y) {
            System.out.println(d);    
        }
        
    }
}
