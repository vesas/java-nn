package fi.vesas.autodiff.loss;

import fi.vesas.autodiff.grad.GradNode;

public interface LossInterface {
    
    void setTruth(double ... t);

    void setInputs(GradNode [] inputs);

    double forward();

    void zeroGrads();

    void backward(double [] y);
}
