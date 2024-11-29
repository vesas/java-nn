package fi.vesas.autodiff.autodiffnn;

import org.junit.jupiter.api.Test;

import fi.vesas.autodiff.grad.GradNode;
import fi.vesas.autodiff.loss.MSELoss;


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
    
    @Test
    public void testConstruction() {
        Model model = new ModelBuilder()
            .add(new InputLayer(2))
            .add(new DenseLayer(4))
            .add(new Linear())
            .add(new DenseLayer(1))
            .add(new Linear())
            .add(new MSELoss())
            .build();

        MLP mlp = (MLP)model;

        mlp.inputLayer.setInputValues(new double[] {0.0, 0.0});
        
        DenseLayer dense0 = mlp.denseLayers[0];
        dense0.neurons[0].weights[0].value = 0.0;
        dense0.neurons[0].weights[1].value = 0.0;
        dense0.neurons[0].bias.value = 1.0;
        dense0.neurons[1].weights[0].value = 0.0;
        dense0.neurons[1].weights[1].value = 0.0;
        dense0.neurons[1].bias.value = 1.0;
        dense0.neurons[2].weights[0].value = 0.0;
        dense0.neurons[2].weights[1].value = 0.0;
        dense0.neurons[2].bias.value = 1.0;
        dense0.neurons[3].weights[0].value = 0.0;
        dense0.neurons[3].weights[1].value = 0.0;
        dense0.neurons[3].bias.value = 1.0;

        DenseLayer dense1 = mlp.denseLayers[1];
        dense1.neurons[0].weights[0].value = 1.0;
        dense1.neurons[0].weights[1].value = 1.0;
        dense1.neurons[0].weights[2].value = 0.0;
        dense1.neurons[0].weights[3].value = 0.0;
        dense1.neurons[0].bias.value = 0.0;

        GradNode[] predictionHead = dense1.getOutputs();
        double value = predictionHead[0].forward();

    }
}
