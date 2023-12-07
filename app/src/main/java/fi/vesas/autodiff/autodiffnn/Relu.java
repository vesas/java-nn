package fi.vesas.autodiff.autodiffnn;

import fi.vesas.autodiff.grad.GradNode;

public class Relu extends GradNode {

    public GradNode x;

    public Relu(GradNode x) {
        this.x = x;
    }

    private static double max(double x, double y) {
        return x > y ? x : y;
    }

    public static double value(double x) {
        return max(0, x);
    }

    public double forward() {
        return value(this.x.forward());
    }

    @Override
    public void grad(double g) {
        if (this.x.forward() > 0) {
            this.x.grad(g);
        }
        else {
            this.x.grad(0);
        }
        
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
