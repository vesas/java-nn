package fi.vesas.autodiff.nn;

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

    // derivative
    // double temp = x.forward();
    // double tanh = Math.tanh(temp);    
    // 
    // return 1- (tanh * tanh);

    @Override
    public String toDotString() {
        return "Tanh()=" + this.forward() + "";
    }

    @Override
    public GradNode[] getChildren() {
        
        GradNode [] children = {this.x};
        return children;
    }

}
