package fi.vesas.autodiff.autodiffnn;

import fi.vesas.autodiff.grad.GradNode;

public class Tanh extends GradNode {
    public GradNode x;

    public Tanh(GradNode x) {
        this.x = x;
    }

    public static double value(double x) {
        return Math.tanh(x);
    }

    public static double derivative(double x) {
        double tanh = Math.tanh(x);
        return 1 - (tanh * tanh);
    }
    
    @Override
    public double forward() {
        double temp = x.forward();
        
        return value(temp);

        
    }

    @Override
    public void grad() {
        
        double temp = x.forward();
        double tanh = Math.tanh(temp);    
        temp = 1- (tanh * tanh);
    
        this.x.grad += this.grad * temp;
    
        this.x.grad();
    }

}
