package fi.vesas.autodiff.autodiffnn;

import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;

import org.junit.jupiter.api.Test;

public class MLPTest {
    
    /**
     * Learn to flip the input
     */
    @Test
    public void testSimple1() {
        double [][] xs = {{-2.0},{3.0},{-0.5},{5.0}, {-1.0}, {0.3}, {1.0}, {2.0}, {3.0}, {4.0}, {-2.5}};
        double [][] ys = {{1.0},{-1.0},{1.0},{-1.0}, {1.0}, {-1.0}, {-1.0}, {-1.0}, {-1.0}, {-1.0}, {1.0}};
        double learningRate = 0.05;

        MLP mlp = new MLP(new int[] {1, 1});

        for(int q = 0; q  < 28; q++) {
            System.out.println(">> round: " + q);
        
            System.out.println("BEFORE: " + mlp.toString());
            for(int i = 0; i < xs.length; i++) {
                
                
                double [] preds = mlp.forward(xs[i]);
                
                mlp.backward(ys[i]);

                StringBuffer sb = new StringBuffer();

                sb.append("preds: ");
                for (double d : preds) {
                    sb.append(d + " ");
                }

                sb.append(" ys: ");
                for (double d : ys[i]) {
                    sb.append(d + " ");
                }

                System.out.println(sb.toString());

            }

            mlp.updateWeights(learningRate);
            mlp.zeroGrads();
            double errorAfter = mlp.loss.forward();
            StringBuffer sb = new StringBuffer();
            sb.append("error: " + errorAfter + " ");
            System.out.println(sb.toString());
        }
    }

    

    @Test
    public void testLong1() {
        double [][] xs = {{-2.0},{3.0},{-0.5},{5.0}};
        double [][] ys = {{1.0},{-1.0},{1.0},{-1.0}};
        double learningRate = 0.05;

        MLP mlp = new MLP(new int[] {1, 1, 1, 1, 1, 1, 1, 1});

        for(int q = 0; q  < 18; q++) {
            System.out.println(">> round: " + q);
            // training examples
            // have to calculate the error based on all examples, otherwise learning does not work

            for(int i = 0; i < xs.length; i++) {
                
                System.out.println("BEFORE: " + mlp.toString());
                double [] preds = mlp.forward(xs[i]);
                
                mlp.backward(ys[i]);

                StringBuffer sb = new StringBuffer();

                sb.append("preds: ");
                for (double d : preds) {
                    sb.append(d + " ");
                }

                sb.append(" ys: ");
                for (double d : ys[i]) {
                    sb.append(d + " ");
                }

                System.out.println(sb.toString());

            }

            mlp.updateWeights(learningRate);
            mlp.zeroGrads();
            double errorAfter = mlp.loss.forward();
            StringBuffer sb = new StringBuffer();
            sb.append("error: " + errorAfter + " ");
            System.out.println(sb.toString());
        }
    }

}
