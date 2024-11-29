package fi.vesas.autodiff.autodiffnn;

import java.util.Random;

public class WeightInitializers {

    public static interface WeightInitializerInterface {
        void initializeWeights(Neuron [] neurons, int inputSize);
    }

    public static class HeInitializer implements WeightInitializerInterface {
        private Random random = new Random();

        public void initializeWeights(Neuron [] neurons, int inputSize) {
            double stddev = Math.sqrt(2.0 / inputSize);  // He initialization

            for(Neuron neuron : neurons) {
                for (int i = 0; i < neuron.weights.length; i++) {
                    neuron.weights[i].value = random.nextGaussian() * stddev;
                }
            }
        }
    }
}
