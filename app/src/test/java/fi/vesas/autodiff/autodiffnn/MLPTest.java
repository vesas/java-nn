package fi.vesas.autodiff.autodiffnn;

import org.junit.jupiter.api.Test;

public class MLPTest {
    
    @Test
    public void testSimple1() {
        double [][] xs = {{-2.0},{3.0},{-0.5},{5.0}};
        double [][] ys = {{1.0},{-1.0},{1.0},{-1.0}};
        double learningRate = 0.01;

        MLP mlp = new MLP(new int[] {1, 1});

        for(int q = 0; q  < 10; q++) {
            System.out.println(">> round: " + q);
            // training examples
            // have to calculate the error based on all examples, otherwise learning does not work

            double errorGrad = 0.0;
            double rmsError = 0.0;
            for(int i = 0; i < xs.length; i++) {
                
                double [] preds = mlp.forward(xs[i]);
                
                double rms = (preds[0] - ys[i][0]) * (preds[0] - ys[i][0]);
                System.out.println(">> preds[0]: " + preds[0] + " ys[i][0]: " + ys[i][0] + " error: " + rms);
                errorGrad += (preds[0] - ys[i][0]);
                rmsError += rms;

                double [] outputgrads = new double[1];
            
                System.out.println(">> rmsError: " + rmsError + " errorGrad: " + errorGrad);
                outputgrads[0] = errorGrad;

                mlp.backward();
                mlp.updateWeights(learningRate);

                // System.out.println(">> i: " + i + " preds " + preds[0]);
            }
        }
    }

}
