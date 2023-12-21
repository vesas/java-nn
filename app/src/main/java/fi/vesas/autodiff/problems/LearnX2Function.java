package fi.vesas.autodiff.problems;

import java.util.Random;

import org.knowm.xchart.QuickChart;
import org.knowm.xchart.SwingWrapper;
import org.knowm.xchart.XYChart;

import fi.vesas.autodiff.autodiffnn.DenseLayer;
import fi.vesas.autodiff.autodiffnn.InputLayer;
import fi.vesas.autodiff.autodiffnn.Linear;
import fi.vesas.autodiff.autodiffnn.Model;
import fi.vesas.autodiff.autodiffnn.ModelBuilder;
import fi.vesas.autodiff.autodiffnn.Sigmoid;
import fi.vesas.autodiff.loss.MSELoss;
import fi.vesas.autodiff.util.Log;
import fi.vesas.autodiff.util.LogToCsvFile;
import fi.vesas.autodiff.util.Util;

public final class LearnX2Function {

    private LearnX2Function () {}

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

        int exampleCount = 4;
        int trainSize = 45;
        int validationSize = 16;
        int batchSize = 4;
        int epochCount = 200;
        double learningRate = 0.005;

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

            double x = Util.rangeRand(-5, 5);
            
            trainXs[i] = new double[]{x};
            trainYs[i] = new double[]{x*x};
        }

        for(int i = 0; i < validationSize; i++) {
            
            double x = Util.rangeRand(-5, 5);
            
            validationXs[i]     = new double[]{x};
            origValidationXs[i] = new double[]{x};

            validationYs[i]     = new double[]{x*x};
            origValidationYs[i] = new double[]{x*x};
        }

        // trainXs = Util.normalize(trainXs, -1.0, 1.0);
        // trainYs = Util.normalize(trainYs, -1.0, 1.0);

        // validationXs = Util.normalize(validationXs, -1.0, 1.0);
        // validationYs = Util.normalize(validationYs, -1.0, 1.0);

        
        Model model = new ModelBuilder()
            .add(new InputLayer(1))
            .add(new DenseLayer(8))
            .add(new Sigmoid())
            .add(new DenseLayer(1))
            .add(new Linear())
            .add(new MSELoss())
            .build();
        
        // MLP mlp = new MLP(new int[] {2, 4, 2});

        LogToCsvFile.logHeader("epoch", "error");
        for(int q = 0; q  < epochCount; q++) {
            System.out.print(">> epoch: " + q + " ");

            randomizeOrder(trainXs, trainYs, randomizedTrainXs, randomizedTrainYs);

            // run training in mini batches
            int batchCounter = 0;
            for(int i = 0; i < randomizedTrainXs.length; i++) {
                
                double [] preds = model.forward(randomizedTrainXs[i]);
                model.backward(randomizedTrainYs[i]);

                StringBuffer sb = new StringBuffer();

                sb.append("preds: ");
                for (double d : preds) {
                    sb.append(d + " ");
                }

                sb.append(" ys: ");
                for (double d : randomizedTrainYs[i]) {
                    sb.append(d + " ");
                }

                // update weights at the end of each minibatch, or at the end of the whole dataset
                if(batchCounter == batchSize || i == randomizedTrainXs.length - 1) {
                    model.updateWeights(learningRate);
                    model.zeroGrads();
                    batchCounter = 0;
                }
                else {
                    batchCounter++;
                }
            }
            // calculate error
            double error = 0.0;
            for(int i = 0; i < validationXs.length; i++) {

                double [] thexs = validationXs[i];
                double [] preds = model.forward(thexs);
                double truth = validationYs[i][0];

                model.getLoss().setTruth(truth);                
                double err = model.getLoss().forward();
                // System.out.println("error: " + err);
                error += err;

                
            }
            
            System.out.println("Error: " + error);
            log.log1(error);
        }

        for(int i = 0; i < validationXs.length; i++) {
                
            double [] preds = model.forward(validationXs[i]);

            StringBuffer sb = new StringBuffer();

            sb.append("Xs: ");
            for (double d : validationXs[i]) {
                sb.append(d + " ");
            }

            sb.append("preds: ");
            for (double d : preds) {
                sb.append(d + " ");
            }

            sb.append(" ys: ");
            for (double d : validationYs[i]) {
                sb.append(d + " ");
            }

            sb.append(" Error Exitnode reports: " + model.getLoss().forward());

            System.out.println(sb.toString());

            // mlp.error.debug();
            
        }

        chartResults(model, log);

    }

    private void chartResults(Model model, Log log) {
        XYChart chart = QuickChart.getChart("Learning curve", "Epoch", "MSE", "MSE", log.getIndexes1(), log.getValues1());
        
        new SwingWrapper(chart).displayChart();


        double [][] values = new double[2][40];
        double [] indexes = new double[40];

        double stride = 10.0 / 40.0;
        int counter = 0;
        for(double i = -5; i < 5; i+= stride) {

            double [] x = new double[]{i};
            double [] preds = model.forward(x);

            values[0][counter] = i * i;
            values[1][counter] = preds[0];

            indexes[counter] = i;
            counter++;
        }

        String [] labels = new String[2];
        labels[0] = "Real";
        labels[1] = "Predicted";
        XYChart chart2 = QuickChart.getChart("Predicted vs. Real values", "X", "Y", labels, indexes, values);
        
        new SwingWrapper(chart2).displayChart();

    }

    public static void main(String[] args) {
        
        LearnX2Function learnX2 = new LearnX2Function();
        learnX2.run();
        
    }
}
