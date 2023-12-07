package fi.vesas.autodiff.autodiffnn;

import fi.vesas.autodiff.grad.GradNode;

public class Tanh extends GradNode {
    public GradNode x;

    public Tanh(GradNode x) {
        this.x = x;
    }

    public Tanh(GradNode x, String label) {
        this.x = x;
        this.label = label;
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
    public void grad(double g) {
        
        double temp = this.forward();

        this.grad = g * (1.0 - (temp * temp));
    
        this.x.grad(this.grad);
    }
    
    // toString
    @Override
    public String toString() {
        return "Tanh_" + label + "()=" + String.format("%.05f", this.forward()) + ", grad=" + String.format("%.08f", this.grad) + ")";
    }

    @Override
    public String toDotString() {
        return "Tanh_" + label;
    }

    @Override
    public GradNode[] getChildren() {
        
        GradNode [] children = {this.x};
        return children;
    }

}
