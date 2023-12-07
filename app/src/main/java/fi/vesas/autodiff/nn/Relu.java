package fi.vesas.autodiff.nn;

import fi.vesas.autodiff.grad.GradNode;
import fi.vesas.autodiff.grad.Value;

public class Relu extends GradNode {

    private Value x;

    public Relu(Value x) {
        this.x = x;
    }

    private static double max(double x, double y) {
        return x > y ? x : y;
    }

    public static double value(double x) {
        return max(0, x);
    }

    public void grad(double g) {
        if (this.x.forward() > 0) {
            this.x.grad(g);
        }
        this.x.grad(0);
    }

    @Override
    public String toDotString() {
        return "Relu";
    }

    @Override
    public GradNode[] getChildren() {
        
        GradNode [] children = {this.x};
        return children;
    }

}
