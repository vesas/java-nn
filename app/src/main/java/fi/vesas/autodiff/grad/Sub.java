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

    @Override
    public double forward() {
        return this.x.forward() - this.y.forward();
    }

    /*
     * Propagate the gradient
     */
    @Override
    public void grad(double g) {

        this.x.grad(1.0 * g);
        this.y.grad(-1.0 * g);
    }

    // toString
    @Override
    public String toString() {
        return "Sub(" + this.x.toString() + " - " + this.y.toString() + ") = " + this.forward() + "";
    }
    
}
