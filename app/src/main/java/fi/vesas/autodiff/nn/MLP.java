package fi.vesas.autodiff.nn;

public class MLP {
    public DenseLayer[] denseLayers;

    public MLP(int inputSize, int[] sizes) {
		int[] layerSizes = new int[sizes.length + 1];
		layerSizes[0] = inputSize;
		for (int i = 0; i < sizes.length; i++) {
			layerSizes[i + 1] = sizes[i];
		}

		denseLayers = new DenseLayer[layerSizes.length - 1];

		for (int i = 0; i < denseLayers.length; i++) {
			denseLayers[i] = new DenseLayer(layerSizes[i], layerSizes[i + 1], "l" + i);
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

		for (int i = 0; i < denseLayers.length; i++) {
			x = denseLayers[i].forward(x);
		}
		return x;
	}

	public double [] backward(double [] outputgrads) {

        double grads [] = denseLayers[(denseLayers.length -1)].backward(outputgrads);

		for (int i = (denseLayers.length -1) -1; i >= 0; i--) {
			grads = denseLayers[i].backward(grads);
		}

		return grads;
	}
	
	public void updateWeights(double learningRate) {
		for(DenseLayer layer : denseLayers) {
			layer.updateWeights(learningRate);
		}
	}
}
