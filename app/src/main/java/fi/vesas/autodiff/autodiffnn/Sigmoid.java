package fi.vesas.autodiff.autodiffnn;

import fi.vesas.autodiff.grad.GradNode;

public class Sigmoid extends GradNode {

    public GradNode x;

    public Sigmoid(GradNode x) {
        this.x = x;
    }

    public Sigmoid(GradNode x, String label) {
        this.x = x;
        this.label = label;
    }

    public static double value(double value) {
        return 1.0 / (1.0 + Math.exp(-value));
    }

    @Override
    public double forward() {
        double temp = x.forward();
        return value(temp);
    }

    @Override
    public void grad(double g) {

        this.grad = g;
        double temp = this.forward();
        this.x.grad((1 - temp *  temp ) * g);
    }

    // toString
    @Override
    public String toString() {
        return "Sigmoid_" + label + "()=" + String.format("%.05f", this.forward()) + ", grad=" + String.format("%.08f", this.grad) + ")";
    }

    @Override
    public String toDotString() {
        return "Sigmoid_" + label;
    }

    @Override
    public GradNode[] getChildren() {
        
        GradNode [] children = {this.x};
        return children;
    }
}
