package fi.vesas.autodiff.loss;

import fi.vesas.autodiff.grad.GradNode;

public interface LossInterface {
    
    public void setTruth(double ... t);

    public void setInputs(GradNode [] inputs);

    public double forward();

    public void zeroGrads();

    public void backward(double [] y);
}
