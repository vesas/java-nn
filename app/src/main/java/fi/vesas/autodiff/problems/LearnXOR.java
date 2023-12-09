package fi.vesas.autodiff.problems;

import java.util.Locale;
import java.util.Random;

import org.knowm.xchart.QuickChart;
import org.knowm.xchart.SwingWrapper;
import org.knowm.xchart.XYChart;

import fi.vesas.autodiff.autodiffnn.MLP;
import fi.vesas.autodiff.grad.GradNode;
import fi.vesas.autodiff.util.Log;
import fi.vesas.autodiff.util.LogToCsvFile;
import fi.vesas.autodiff.util.Util;

public final class LearnXOR {

    private LearnXOR () {}

    private Random rand = new Random();

    private void randomizeOrder(double[][] x, double [][] y, double [][] resX, double [][] resY) {

        boolean [] used = new boolean[x.length];

        for(int i = 0; i < x.length; i++) {
            int index = rand.nextInt(x.length);
            while(used[index]) {
                index = rand.nextInt(x.length);
            }
            used[index] = true;

            resX[i] = x[index];
            resY[i] = y[index];
        }
    }

    public void run() {

        Log log = new Log();

        Random rand = new Random(3);

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
        int trainSize = 45;
        int validationSize = 16;
        int batchSize = 4;
        int epochCount = 100;

        double [][] trainXs = new double[trainSize][];
        double [][] trainYs = new double[trainSize][];

        double [][] randomizedTrainXs = new double[trainSize][];
        double [][] randomizedTrainYs = new double[trainSize][];

        double [][] origValidationXs = new double[validationSize][];
        double [][] origValidationYs = new double[validationSize][];
        double [][] validationXs = new double[validationSize][];
        double [][] validationYs = new double[validationSize][];

        // create some test data
        for(int i = 0; i < trainSize; i++) {
            int rnd = rand.nextInt(exampleCount);
            
            trainXs[i] = xscases[rnd];
            trainYs[i] = yscases[rnd];
        }

        for(int i = 0; i < validationSize; i++) {
            int rnd = rand.nextInt(exampleCount);
            
            validationXs[i]     = xscases[rnd];
            origValidationXs[i] = xscases[rnd];

            validationYs[i]     = yscases[rnd];
            origValidationYs[i] = yscases[rnd];
        }

        trainXs = Util.normalize(trainXs, -1.0, 1.0);
        trainYs = Util.normalize(trainYs, -1.0, 1.0);

        validationXs = Util.normalize(validationXs, -1.0, 1.0);
        validationYs = Util.normalize(validationYs, -1.0, 1.0);

        double learningRate = 0.02;

        MLP mlp = new MLP(new int[] {2, 2, 1});

        LogToCsvFile.logHeader("epoch", "error");
        for(int q = 0; q  < epochCount; q++) {
            System.out.print(">> epoch: " + q + " ");

            randomizeOrder(trainXs, trainYs, randomizedTrainXs, randomizedTrainYs);

            // run training in mini batches
            int batchCounter = 0;
            for(int i = 0; i < randomizedTrainXs.length; i++) {
                
                int index = rand.nextInt(trainXs.length);
                double [] preds = mlp.forward(trainXs[index]);
                mlp.backward(trainYs[index]);

                StringBuffer sb = new StringBuffer();

                sb.append("preds: ");
                for (double d : preds) {
                    sb.append(d + " ");
                }

                sb.append(" ys: ");
                for (double d : trainYs[index]) {
                    sb.append(d + " ");
                }

                // update weights at the end of each minibatch, or at the end of the whole dataset
                if(batchCounter == batchSize || i == randomizedTrainXs.length - 1) {
                    mlp.updateWeights(learningRate);
                    mlp.zeroGrads();
                    batchCounter = 0;
                }
                else {
                    batchCounter++;
                }
            }
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
            
            System.out.println("Error: " + error);
            log.log1(error);
        }

        for(int i = 0; i < origValidationXs.length; i++) {
                
            double [] preds = mlp.forward(origValidationXs[i]);

            StringBuffer sb = new StringBuffer();

            sb.append("Xs: ");
            for (double d : origValidationXs[i]) {
                sb.append(d + " ");
            }

            sb.append("preds: ");
            for (double d : preds) {
                sb.append(d + " ");
            }

            sb.append(" ys: ");
            for (double d : origValidationYs[i]) {
                sb.append(d + " ");
            }

            System.out.println(sb.toString());

            // mlp.error.debug();
            
        }

        mlp.error.debug();
        System.out.println("Error exitnode reports: " + mlp.error.exitNode.forward());

        XYChart chart = QuickChart.getChart("Learning curve", "Epoch", "MSE", "MSE", log.getIndexes1(), log.getValues1());
        
        new SwingWrapper(chart).displayChart();
    }

    public static void main(String[] args) {
        
        LearnXOR learnXOR = new LearnXOR();
        learnXOR.run();
        
    }
}
