package fi.vesas.autodiff.autodiffnn;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import fi.vesas.autodiff.autodiffnn.WeightInitializers.HeInitializer;
import fi.vesas.autodiff.grad.GradNode;
import fi.vesas.autodiff.grad.Value;

public class WeightInitializersTest {

    private static Neuron[] buildNeurons(int neuronCount, int inputSize) {
        GradNode[] inputs = new GradNode[inputSize];
        for (int i = 0; i < inputSize; i++) {
            inputs[i] = new Value(0.0, "in" + i);
        }

        Neuron[] neurons = new Neuron[neuronCount];
        for (int i = 0; i < neuronCount; i++) {
            neurons[i] = new Neuron(inputs, "n" + i);
        }
        return neurons;
    }

    @Test
    public void testEveryWeightIsOverwritten() {
        int inputSize = 8;
        Neuron[] neurons = buildNeurons(4, inputSize);

        // baseline: zero out weights so we can confirm the initializer wrote to them
        for (Neuron n : neurons) {
            for (int i = 0; i < n.weights.length; i++) {
                n.weights[i].value = 0.0;
            }
        }

        new HeInitializer().initializeWeights(neurons, inputSize);

        for (Neuron n : neurons) {
            assertEquals(inputSize, n.weights.length);
            for (Value w : n.weights) {
                // Gaussian draws being exactly 0.0 has probability 0
                assertNotEquals(0.0, w.value);
            }
        }
    }

    @Test
    public void testStddevApproximatesHeFormula() {
        int inputSize = 100;
        // Many neurons so we get ~10_000 samples — sample stddev will be
        // extremely close to the population value, making this robust.
        int neuronCount = 100;
        double expectedStddev = Math.sqrt(2.0 / inputSize);

        Neuron[] neurons = buildNeurons(neuronCount, inputSize);
        new HeInitializer().initializeWeights(neurons, inputSize);

        int n = 0;
        double sum = 0.0;
        for (Neuron neuron : neurons) {
            for (Value w : neuron.weights) {
                sum += w.value;
                n++;
            }
        }
        double mean = sum / n;

        double sqSum = 0.0;
        for (Neuron neuron : neurons) {
            for (Value w : neuron.weights) {
                double d = w.value - mean;
                sqSum += d * d;
            }
        }
        double stddev = Math.sqrt(sqSum / n);

        // Mean should be near 0; stddev within 20% of sqrt(2/inputSize).
        assertTrue(Math.abs(mean) < 0.05, "mean=" + mean + " should be near 0");
        assertTrue(
                Math.abs(stddev - expectedStddev) / expectedStddev < 0.2,
                "stddev=" + stddev + " expected≈" + expectedStddev);
    }
}
