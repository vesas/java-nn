package fi.vesas.autodiff.autodiffnn;

import java.util.Random;

import org.junit.jupiter.api.Test;

import fi.vesas.autodiff.grad.GradNode;
import fi.vesas.autodiff.util.DotFile;
import fi.vesas.autodiff.util.Util;

/*
 * Tries to solve the XOR problem
 */
public class XORTest {

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
        int validationSize = 25;

        double [][] trainXs = new double[trainSize][];
        double [][] trainYs = new double[trainSize][];

        double [][] validationXs = new double[validationSize][];
        double [][] validationYs = new double[validationSize][];

        for(int i = 0; i < trainSize; i++) {
            int rnd = rand.nextInt(exampleCount);
            
            trainXs[i] = xscases[rnd];
            trainYs[i] = yscases[rnd];
        }

        for(int i = 0; i < validationSize; i++) {
            int rnd = rand.nextInt(exampleCount);
            
            validationXs[i] = xscases[rnd];
            validationYs[i] = yscases[rnd];
        }

        trainXs = Util.normalize(trainXs, -1.0, 1.0);
        trainYs = Util.normalize(trainYs, -1.0, 1.0);

        validationXs = Util.normalize(validationXs, -1.0, 1.0);
        validationYs = Util.normalize(validationYs, -1.0, 1.0);

        double learningRate = 0.1;

        MLP mlp = new MLP(new int[] {2, 4, 1});

        for(int q = 0; q  < 82; q++) {
            System.out.print(">> round: " + q + " ");


            for(int i = 0; i < trainXs.length; i++) {
                
                double [] preds = mlp.forward(trainXs[i]);
                mlp.backward(trainYs[i]);

                StringBuffer sb = new StringBuffer();

                sb.append("preds: ");
                for (double d : preds) {
                    sb.append(d + " ");
                }

                sb.append(" ys: ");
                for (double d : trainYs[i]) {
                    sb.append(d + " ");
                }

                // System.out.println(sb.toString());

                // mlp.error.debug();
                
            }

            // mlp.printWeights();
            mlp.updateWeights(learningRate);
            // mlp.printWeights();
            mlp.zeroGrads();

            // calculate error
            double error = 0.0;
            for(int i = 0; i < validationXs.length; i++) {

                double [] preds = mlp.forward(validationXs[i]);

                mlp.error.setTruth(validationYs[i][0]);

                GradNode [] gradNodes = mlp.denseLayers[mlp.denseLayers.length -1].getOutputs();

                double res = gradNodes[0].forward();
                
                double err = mlp.error.exitNode.forward();
                // System.out.println("error: " + err);
                error += err;

                
            }
            
            System.out.println("Total error: " + error);

            // mlp.backward(trainYs[0]);

            // DotFile.gen(mlp.error.exitNode, "xor" + q + ".dot");
        }

        for(int i = 0; i < validationXs.length; i++) {
                
                double [] preds = mlp.forward(validationXs[i]);

                StringBuffer sb = new StringBuffer();

                sb.append("preds: ");
                for (double d : preds) {
                    sb.append(d + " ");
                }

                sb.append(" ys: ");
                for (double d : validationYs[i]) {
                    sb.append(d + " ");
                }

                System.out.println(sb.toString());

                // mlp.error.debug();
                
            }
    }
}
