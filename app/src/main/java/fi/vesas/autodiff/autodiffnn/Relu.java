package fi.vesas.autodiff.autodiffnn;

import fi.vesas.autodiff.grad.GradNode;

public class Relu extends GradNode implements Activation {

    public GradNode x;

    public Relu() {
    }

    public Relu(GradNode x) {
        this.x = x;
    }

    @Override
    public void setInput(GradNode x) {
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
    public void zeroGrads(){
        this.grad = 0.0;
        this.x.zeroGrads();
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

    @Override
    public Activation createInstance() {
        return new Relu();
    }
    

}
