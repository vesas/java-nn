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

    public void run() {
        Log log = new Log();

        double [][] xs = {{-2.0},{3.0},{-0.5},{5.0}, {-1.0}, {0.3}, {1.0}, {2.0}, {3.0}, {4.0}, {-2.5}};
        double [][] ys = {{1.0},{-1.0},{1.0},{-1.0}, {1.0}, {-1.0}, {-1.0}, {-1.0}, {-1.0}, {-1.0}, {1.0}};
        double learningRate = 0.05;

        Model model = new ModelBuilder()
            .add(new InputLayer(1))
            .add(new DenseLayer(4))
            .add(new Tanh())
            .add(new DenseLayer(1))
            .add(new Linear())
            .add(new MSELoss())
            .build();

        for(int q = 0; q  < 28; q++) {
            System.out.println(">> round: " + q);
        
            for(int i = 0; i < xs.length; i++) {
                
                
                double [] preds = model.forward(xs[i]);
                
                model.backward(ys[i]);

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

            model.updateWeights(learningRate);
            model.zeroGrads();
            double errorAfter = model.getLoss().forward();
            StringBuffer sb = new StringBuffer();
            sb.append("error: " + errorAfter + " ");
            log.log1(errorAfter);
            System.out.println(sb.toString());
        }

        chartResults(model, log);
    }

    private void chartResults(Model model, Log log) {
            XYChart chart = QuickChart.getChart("Learning curve", "Epoch", "MSE", "MSE", log.getIndexes1(), log.getValues1());
            new SwingWrapper(chart).displayChart();
    }
}
