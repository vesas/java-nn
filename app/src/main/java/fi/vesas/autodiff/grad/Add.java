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
        this.grad = 1.0;

        this.x.grad(this.grad * g);
        this.y.grad(this.grad * g);
    }

    // toString
    @Override
    public String toString() {
        return "Add(" + this.x.toString() + " + " + this.y.toString() + ") = " + this.forward() + "";
    }
}
