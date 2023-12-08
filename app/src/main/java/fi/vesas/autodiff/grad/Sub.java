package fi.vesas.autodiff.grad;

/*
 * Subtract two values
 */
public class Sub extends GradNode {

    public GradNode x;
    public GradNode y;

    public Sub(GradNode x, GradNode y) {
        this.x = x;
        this.y = y;
    }

    public Sub(GradNode x, GradNode y, String label) {
        this.x = x;
        this.y = y;
        this.label = label;
    }

    @Override
    public double forward() {
        return this.x.forward() - this.y.forward();
    }

    @Override
    public void zeroGrads(){
        this.grad = 0.0;
        x.zeroGrads();
        y.zeroGrads();
    }

    /*
     * Propagate the gradient
     */
    @Override
    public void grad(double g) {

        this.grad = g;
        this.x.grad(g);
        this.y.grad(-g);
    }

    // toString
    @Override
    public String toString() {
        return "Sub_" + label + "() = " + String.format("%.05f", this.forward()) + ", grad=" + String.format("%.05f", this.grad) + ")";
    }

    @Override
    public String toDotString() {
        return "Sub_" + label;
    }

    @Override
    public GradNode[] getChildren() {
        
        GradNode [] children = {this.x, this.y};
        return children;
    }
    
}
