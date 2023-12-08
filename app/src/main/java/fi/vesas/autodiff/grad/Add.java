package fi.vesas.autodiff.grad;

/*
 * Add two values
 */
public class Add extends GradNode {

    public GradNode x;
    public GradNode y;

    public Add(GradNode x, GradNode y) {
        this.x = x;
        this.y = y;
    }

    @Override
    public double forward() {
        return this.x.forward() + this.y.forward();
    }

    /*
     * Propagate the gradient
     */
    @Override
    public void grad(double g) {
        this.grad = g;

        this.x.grad(this.grad * 1.0);
        this.y.grad(this.grad * 1.0);
    }

    @Override
    public void zeroGrads(){
        this.grad = 0.0;
        this.x.zeroGrads();
        this.y.zeroGrads();
    }

    // toString
    @Override
    public String toString() {
        return "Add() = " + String.format("%.05f", this.forward()) + ", grad=\"" + String.format("%.05f", this.grad) + ")";
    }

    @Override
    public String toDotString() {
        return "Add";
    }

    @Override
    public GradNode[] getChildren() {
        
        GradNode [] children = {this.x, this.y};
        return children;
    }
}
