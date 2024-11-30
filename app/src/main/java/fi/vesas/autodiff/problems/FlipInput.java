package fi.vesas.autodiff.problems;

import org.knowm.xchart.QuickChart;
import org.knowm.xchart.SwingWrapper;
import org.knowm.xchart.XYChart;

import fi.vesas.autodiff.autodiffnn.DenseLayer;
import fi.vesas.autodiff.autodiffnn.InputLayer;
import fi.vesas.autodiff.autodiffnn.Linear;
import fi.vesas.autodiff.autodiffnn.Model;
import fi.vesas.autodiff.autodiffnn.ModelBuilder;
import fi.vesas.autodiff.autodiffnn.Tanh;
import fi.vesas.autodiff.loss.MSELoss;
import fi.vesas.autodiff.util.Log;

public class FlipInput {
    

    public static void main(String[] args) {
        
        FlipInput flipInput = new FlipInput();
        flipInput.run();
        
    }

    private double[][] generateData(int numPoints) {
        double[][] data = new double[numPoints][2]; // [x, y] pairs
        for (int i = 0; i < numPoints; i++) {
            double x = -5 + (Math.random() * 10); // Random x between -5 and 5
            data[i][0] = x;
            data[i][1] = x < -1 || (x > 0 && x < 1) ? 1.0 : -1.0; // Target function
        }
        return data;
    }

    public void run() {
        Log log = new Log();
        
        // Generate larger dataset
        double[][] allData = generateData(200);
        
        // Split into train/validation (80/20)
        int trainSize = (int)(allData.length * 0.8);
        double[][] trainX = new double[trainSize][1];
        double[][] trainY = new double[trainSize][1];
        double[][] valX = new double[allData.length - trainSize][1];
        double[][] valY = new double[allData.length - trainSize][1];

        // Populate splits
        for (int i = 0; i < allData.length; i++) {
            if (i < trainSize) {
                trainX[i][0] = allData[i][0];
                trainY[i][0] = allData[i][1];
            } else {
                valX[i-trainSize][0] = allData[i][0];
                valY[i-trainSize][0] = allData[i][1];
            }
        }
        
        double learningRate = 0.001;

        Model model = new ModelBuilder()
            .add(new InputLayer(1))
            .add(new DenseLayer(4))
            .add(new Tanh())
            .add(new DenseLayer(1))
            .add(new Tanh())
            .add(new MSELoss())
            .build();

        // Training loop with validation
        double bestValError = Double.MAX_VALUE;
        int patience = 5;
        int noImprovement = 0;
        
        for(int epoch = 0; epoch < 100; epoch++) {
            // Training
            double trainError = trainEpoch(model, trainX, trainY, learningRate);
            
            // Validation
            double valError = validateEpoch(model, valX, valY);
            
            log.log1(trainError);
            log.log2(valError); // Add second series for validation error
            
            System.out.printf("Epoch %d - Train Error: %.4f, Val Error: %.4f%n", 
                            epoch, trainError, valError);
            
            // Early stopping
            if (valError < bestValError) {
                bestValError = valError;
                noImprovement = 0;
            } else {
                noImprovement++;
                if (noImprovement >= patience) {
                    System.out.println("Early stopping triggered");
                    break;
                }
            }
        }

        chartResults(model, log);
    }

    private double trainEpoch(Model model, double[][] x, double[][] y, double learningRate) {
        double epochError = 0;
        for(int i = 0; i < x.length; i++) {
            model.forward(x[i]);
            model.backward(y[i]);
            model.updateWeights(learningRate);
            model.zeroGrads();
            epochError += model.getLoss().forward();
        }
        return epochError / x.length;
    }

    private double validateEpoch(Model model, double[][] x, double[][] y) {
        double valError = 0;
        for(int i = 0; i < x.length; i++) {
            model.forward(x[i]);
            valError += model.getLoss().forward();
        }
        return valError / x.length;
    }

    private void chartResults(Model model, Log log) {
        XYChart chart = QuickChart.getChart("Learning curves", "Epoch", "MSE", 
            new String[] {"Training MSE", "Validation MSE"}, 
            log.getIndexes1(), 
            new double[][] {log.getValues1(), log.getValues2()});
        new SwingWrapper(chart).displayChart();
    }
}
