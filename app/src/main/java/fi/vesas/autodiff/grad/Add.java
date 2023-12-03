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
    public void grad() {
        this.x.grad += this.grad;
        this.y.grad += this.grad;

        this.x.grad();
        this.y.grad();
    }

    
}
