package fi.vesas.autodiff.problems;

import java.util.Random;

import org.knowm.xchart.QuickChart;
import org.knowm.xchart.SwingWrapper;
import org.knowm.xchart.XYChart;

import fi.vesas.autodiff.autodiffnn.DenseLayer;
import fi.vesas.autodiff.autodiffnn.InputLayer;
import fi.vesas.autodiff.autodiffnn.Model;
import fi.vesas.autodiff.autodiffnn.ModelBuilder;
import fi.vesas.autodiff.autodiffnn.Sigmoid;
import fi.vesas.autodiff.autodiffnn.Tanh;
import fi.vesas.autodiff.loss.CrossEntropyLoss;
import fi.vesas.autodiff.loss.MSELoss;
import fi.vesas.autodiff.util.Log;
import fi.vesas.autodiff.util.LogToCsvFile;

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
        int batchSize = 1;
        int epochCount = 500;
        double learningRate = 0.001;

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

        // trainXs = Util.normalize(trainXs, -1.0, 1.0);
        // trainYs = Util.normalize(trainYs, -1.0, 1.0);

        // validationXs = Util.normalize(validationXs, -1.0, 1.0);
        // validationYs = Util.normalize(validationYs, -1.0, 1.0);

        
        Model model = new ModelBuilder()
            .add(new InputLayer(2))
            .add(new DenseLayer(4))
            .add(new Sigmoid())
            .add(new DenseLayer(1))
            .add(new Sigmoid())
            .add(new MSELoss())
            .build();
        
        // MLP model = new MLP(new int[] {2, 4, 2});

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

        XYChart chart = QuickChart.getChart("Learning curve", "Epoch", "MSE", "MSE", log.getIndexes1(), log.getValues1());
        
        new SwingWrapper(chart).displayChart();


        int correct = 0;
        // make predictions
        for(int i = 0; i < 100; i++) {

            int r = rand.nextInt(exampleCount);

            double [] preds = model.forward(xscases[r]);

            // double roundedPred = Math.round(preds[0]);
            if(preds[0] > 0.5 && yscases[r][0] == 1.0) {
                correct++;
            }
            else if(preds[0] < 0.5 && yscases[r][0] == 0.0) {
                correct++;
            }
        }

        System.out.println("Correct: " + correct + " out of 1000");  
    }

    public static void main(String[] args) {
        
        LearnXOR learnXOR = new LearnXOR();
        learnXOR.run();
        
    }
}
