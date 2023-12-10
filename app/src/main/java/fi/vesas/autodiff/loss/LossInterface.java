package fi.vesas.autodiff.loss;

public interface LossInterface {
    
    public void setTruth(double ... t);

    public double forward();

    public void zeroGrads();

    public void backward(double [] y);
}
