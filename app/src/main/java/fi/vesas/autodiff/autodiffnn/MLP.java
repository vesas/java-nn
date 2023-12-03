package fi.vesas.autodiff.autodiffnn;

import java.util.Arrays;

import fi.vesas.autodiff.grad.GradNode;
import fi.vesas.autodiff.grad.Value;

public class MLP {

	public InputLayer inputLayer;
    public DenseLayer[] denseLayers;

    public MLP(int[] sizes) {

		this.inputLayer = new InputLayer(sizes[0]);

		denseLayers = new DenseLayer[sizes.length - 1];

		Value [] inputs = this.inputLayer.getOutputs();

		for (int i = 1; i < sizes.length; i++) {
			denseLayers[i-1] = new DenseLayer(inputs, "l" + i);
		}
	}

	public void printWeights() {

		for (int i = 0; i < denseLayers.length; i++) {
			DenseLayer layer = denseLayers[i];
			System.out.println("layer " + i);
			layer.printWeights();
		}
	}

	public double[] forward(double[] x) {

		// first feed input values
		Value [] inputs = inputLayer.getOutputs();
		for (int i = 0; i < x.length; i++) {
			
			inputs[i].value = x[i];
		}

		// then feed forward
		GradNode [] gradNodes = denseLayers[denseLayers.length -1].getOutputs();

		double [] output = new double[gradNodes.length];
		for (int i = 0; i < gradNodes.length; i++) {
			double value = gradNodes[i].forward();
			output[i] = value;
		}

		

		// then return outputs
		return output;

	}

	public void backward() {
        denseLayers[(denseLayers.length -1)].backward();
	}
	
	public void updateWeights(double learningRate) {
		for(DenseLayer layer : denseLayers) {
			layer.updateWeights(learningRate);
		}
	}

	static public void main(String [] args) {

		MLP mlp = new MLP(new int [] {2, 3, 2});

		mlp.printWeights();

		double [] x = new double [] {1.0, 2.0};
		double [] y = mlp.forward(x);

		System.out.println("y: " + Arrays.toString(y));

		double [] outputgrads = new double [] {1.0, 1.0};

		mlp.backward();

		mlp.updateWeights(0.1);

		mlp.printWeights();
	}

	public void testSimple1() {
        double [][] xs = {{-2.0},{3.0}};
        double [][] ys = {{1.0},{-1.0}};
        double learningRate = 0.1;

        MLP mlp = new MLP(new int[] {1, 1});

        for(int q = 0; q  < 10; q++) {
            System.out.println(">> round: " + q);
            // training examples
            // have to calculate the error based on all examples, otherwise learning does not work

            double errorGrad = 0.0;
            double rmsError = 0.0;
            for(int i = 0; i < 2; i++) {
                
                double [] preds = mlp.forward(xs[i]);
                
                double rms = (preds[0] - ys[i][0]) * (preds[0] - ys[i][0]);
                System.out.println("preds[0]: " + preds[0] + " ys[i][0]: " + ys[i][0] + " error: " + rms);
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
