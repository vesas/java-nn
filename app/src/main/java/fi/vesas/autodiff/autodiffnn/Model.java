package fi.vesas.autodiff.autodiffnn;

import fi.vesas.autodiff.loss.LossInterface;

public interface Model {
    
    double[] forward(double[] x);

    void backward(double [] y);

    void updateWeights(double learningRate);

    void zeroGrads();

    LossInterface getLoss();
}
