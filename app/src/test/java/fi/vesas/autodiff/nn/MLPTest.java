package fi.vesas.autodiff.nn;

import org.junit.jupiter.api.Test;

import fi.vesas.autodiff.nn.MLP;

public class MLPTest {
    
    @Test
    public void testSimple1() {
        double [][] xs = {{-2.0},{3.0}};
        double [][] ys = {{1.0},{-1.0}};
        double learningRate = 0.1;

        MLP mlp = new MLP(1, new int[] {1});

        for(int q = 0; q  < 10; q++) {
            System.out.println(">> round: " + q);
            // training examples
            // have to calculate the error based on all examples, otherwise learning does not work

            double errorGrad = 0.0;
            double rmsError = 0.0;
            for(int i = 0; i < 2; i++) {

                double [] preds = mlp.forward(xs[i]);
                
                double rms = (preds[0] - ys[i][0]) * (preds[0] - ys[i][0]);
                System.out.println("preds[0]: " + preds[0] + " ys[i][0]: " + ys[i][0] + " error: " + rms);
                errorGrad += (preds[0] - ys[i][0]);
                rmsError += rms;

                double [] outputgrads = new double[1];
            
                System.out.println(">> rmsError: " + rmsError + " errorGrad: " + errorGrad);
                outputgrads[0] = errorGrad;

                double [] grads = mlp.backward(outputgrads);
                mlp.updateWeights(learningRate);

                // System.out.println(">> i: " + i + " preds " + preds[0]);
            }
        }
    }

    @Test
    public void testSimple() {
        double [][] xs = {{-2.0},{3.0}};
        double [][] ys = {{1.0},{-1.0}};
        double learningRate = 0.01;

        MLP mlp = new MLP(1, new int[] {1, 1});

        for(int q = 0; q  < 10; q++) {
            System.out.println(">> round: " + q);
            // training examples
            // have to calculate the error based on all examples, otherwise learning does not work

            double errorGrad = 0.0;
            double rmsError = 0.0;
            for(int i = 0; i < 2; i++) {

                double [] preds = mlp.forward(xs[i]);
                
                double rms = (preds[0] - ys[i][0]) * (preds[0] - ys[i][0]);
                System.out.println("preds[0]: " + preds[0] + " ys[i][0]: " + ys[i][0] + " error: " + rms);
                errorGrad += (preds[0] - ys[i][0]);
                rmsError += rms;

                double [] outputgrads = new double[1];
            
                System.out.println(">> rmsError: " + rmsError + " errorGrad: " + errorGrad);
                outputgrads[0] = errorGrad;

                double [] grads = mlp.backward(outputgrads);
                mlp.updateWeights(learningRate);

                // System.out.println(">> i: " + i + " preds " + preds[0]);
            }
        }
    }

    @Test
    public void testXOR() {
        // we have four examples
        double [][] xs = {{0.0,0.0},
                        {0.0,1.0},
                        {1.0,0.0},
                        {1.0,1.0}};
        double [][] ys = {{0.0},
                        {1.0},
                        {1.0},
                        {0.0}};

        double learningRate = 0.1;

        MLP mlp = new MLP(2, new int[] {2, 2, 1});

        for(int q = 0; q  < 30; q++) {
            System.out.println(">> round: " + q);
            // training examples
            // have to calculate the error based on all examples, otherwise learning does not work

            double errorGrad = 1.0;
            double rmsError = 0.0;
            for(int i = 0; i < 4; i++) {

                double [] preds = mlp.forward(xs[i]);
                
                double rms = (preds[0] - ys[i][0]) * (preds[0] - ys[i][0]);
                // System.out.println("preds[0]: " + preds[0] + " ys[i][0]: " + ys[i][0] + " error: " + rms);
                errorGrad += (preds[0] - ys[i][0]);
                rmsError += rms;

                double [] outputgrads = new double[1];
            
                // System.out.println(">> rmsError: " + rmsError + " errorGrad: " + errorGrad);
                outputgrads[0] = errorGrad;

                double [] grads = mlp.backward(outputgrads);
                mlp.updateWeights(learningRate);

                // System.out.println(">> i: " + i + " preds " + preds[0]);
            }

            for(int i = 0; i < 4; i++) {

                double [] preds = mlp.forward(xs[i]);
                System.out.println("%% pred: " + preds[0] + " y: " + ys[i][0]);
            }

        }
        
        int qwe = 0;
    }

    @Test
    public void test1() {

        double [] xs = {2.0,3.0};
        double [] ys = {0.0};

        MLP mlp = new MLP(2, new int[] {2, 1});

        mlp.denseLayers[0].neurons[0].weights[0] = 1.0;
        mlp.denseLayers[0].neurons[0].weights[1] = 2.0;
        mlp.denseLayers[0].neurons[0].bias = 0.5;

        mlp.denseLayers[0].neurons[1].weights[0] = 1.5;
        mlp.denseLayers[0].neurons[1].weights[1] = -3.0;
        mlp.denseLayers[0].neurons[1].bias = 1.5;

        mlp.denseLayers[1].neurons[0].weights[0] = 0.5;
        mlp.denseLayers[1].neurons[0].weights[1] = 1.5;
        mlp.denseLayers[1].neurons[0].bias = 0.5;


        double [] preds = mlp.forward(xs);
        double [] outputgrads = new double[] {1.0};

        mlp.backward(outputgrads);

        int qwe = 0;

        /*
        int i = 0;
        double loss = 0.0;
        for(double [] x : xs) {
            double [] preds = mlp.forward(x);

            for (double d : preds) {
                System.out.println(d);
            }

            double singleloss = preds[0] - ys[i];
            singleloss = singleloss * singleloss;
            loss += singleloss;
            

            i++;
        }

        System.out.println("loss: " + loss);
        */

    }

    @Test
    public void test2() {
        double [] xs = {2.0,3.0};
        double [] ys = {0.0};
        double h = 0.0000001;

        MLP mlp = new MLP(2, new int[] {2, 1});

        double [] preds = mlp.forward(xs);
        double [] outputgrads = new double[] {1.0};

        System.out.println(">> forward preds: " + preds[0]);

        // these are the input data gradients
        double [] grads = mlp.backward(outputgrads);

        mlp.updateWeights(0.001);

        preds = mlp.forward(xs);
        System.out.println(">> forward preds: " + preds[0]);

        for(double d : grads) {
            
            System.out.print(">> >>>>>>> data grads");
            System.out.print(" " + d);
        }

        mlp.printWeights();
        mlp.denseLayers[0].neurons[0].weights[0] += h;
        double [] preds2 = mlp.forward(xs);
        
        double derivative = (preds2[0] - preds[0]) / h;
        System.out.println(">> l0-n0-w0 derivative: " + derivative);
        System.out.println(">> forward preds: " + preds2[0]);

        int qwe = 0;
    }
}
