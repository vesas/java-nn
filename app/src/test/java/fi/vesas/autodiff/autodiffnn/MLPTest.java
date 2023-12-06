package fi.vesas.autodiff.autodiffnn;

import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;

import org.junit.jupiter.api.Test;

public class MLPTest {
    
    @Test
    public void testSimple1() {
        double [][] xs = {{-2.0},{3.0},{-0.5},{5.0}};
        double [][] ys = {{1.0},{-1.0},{1.0},{-1.0}};
        double learningRate = 0.05;

        MLP mlp = new MLP(new int[] {1, 1});

        for(int q = 0; q  < 18; q++) {
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
            double errorAfter = mlp.error.exitNode.forward();
            StringBuffer sb = new StringBuffer();
            sb.append("error: " + errorAfter + " ");
            System.out.println(sb.toString());
        }
    }

    @Test
    public void testXOR() {

        Random rand = new Random(2);

        // we have four cases
        double [][] xscases = {{0.0,0.0},
                        {0.0,1.0},
                        {1.0,0.0},
                        {1.0,1.0}};
        double [][] yscases = {{0.0},
                        {1.0},
                        {1.0},
                        {0.0}};

        int exampleCount = 4;
        int trainSize = 35;

        double [][] xs = new double[trainSize][];
        double [][] ys = new double[trainSize][];

        for(int i = 0; i < trainSize; i++) {
            int rnd = rand.nextInt(exampleCount);
            
            xs[i] = xscases[rnd];
            ys[i] = yscases[rnd];
        }

        double learningRate = 0.1;

        MLP mlp = new MLP(new int[] {2, 4, 1});

        for(int q = 0; q  < 7; q++) {
            System.out.println(">> round: " + q);

            

            // training examples
            // have to calculate the error based on all examples, otherwise learning does not work

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

                // mlp.error.debug();
                
            }

            // mlp.printWeights();
            mlp.updateWeights(learningRate);
            // mlp.printWeights();
            mlp.zeroGrads();
            double error = mlp.error.exitNode.forward();
            
            StringBuffer sb = new StringBuffer();
            sb.append("error: " + error + " ");
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
            double errorAfter = mlp.error.exitNode.forward();
            StringBuffer sb = new StringBuffer();
            sb.append("error: " + errorAfter + " ");
            System.out.println(sb.toString());
        }
    }

}
