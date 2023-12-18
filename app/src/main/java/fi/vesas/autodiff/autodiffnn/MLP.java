package fi.vesas.autodiff.autodiffnn;

import java.util.Arrays;

import fi.vesas.autodiff.grad.GradNode;
import fi.vesas.autodiff.grad.Value;
import fi.vesas.autodiff.loss.CrossEntropyLoss;
import fi.vesas.autodiff.loss.LossInterface;
import fi.vesas.autodiff.loss.MSELoss;

public class MLP implements Model {

	public InputLayer inputLayer;
    public DenseLayer[] denseLayers;
	public LossInterface loss = null;

	public MLP() {

	}

	/*
	 * Init to some reasonable default values based on the layer sizes
	 */
    public MLP(int[] sizes) {

		this.inputLayer = new InputLayer(sizes[0]);

		denseLayers = new DenseLayer[sizes.length - 1];

		GradNode [] inputs = this.inputLayer.getOutputs();

		for (int i = 1; i < sizes.length; i++) {
			denseLayers[i-1] = new DenseLayer(inputs, sizes[i], "l" + i);
			inputs = denseLayers[i-1].getOutputs();
		}

		// loss = new CrossEntropyLoss(inputs);
		loss = new MSELoss(inputs);
	}

	public void setInputValues(double[] x) {

		Value [] inputs = inputLayer.getOutputs();
		for (int i = 0; i < x.length; i++) {
			
			inputs[i].value = x[i];
		}
	}

	public double[] forward(double[] x) {

		setInputValues(x);

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

	public void zeroGrads() {
		loss.zeroGrads();
	}

	public LossInterface getLoss() {
		return loss;
	}

	public void backward(double [] y) {

		for(DenseLayer layer : denseLayers) {
			layer.recordWeights();
		}

		loss.backward(y);
	}
	
	public void updateWeights(double learningRate) {
		for(DenseLayer layer : denseLayers) {
			layer.updateWeights(learningRate);
		}
	}

	static public void main(String [] args) {

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

                mlp.backward(ys[i]);
                mlp.updateWeights(learningRate);

                // System.out.println(">> i: " + i + " preds " + preds[0]);
            }
        }
    }

	public String toString() {
		return "MLP: " + Arrays.toString(denseLayers);
	}

	public void printWeights() {

		for (int i = 0; i < denseLayers.length; i++) {
			DenseLayer layer = denseLayers[i];
			System.out.println("layer " + i);
			layer.printWeights();
		}
	}
}
