package fi.vesas.autodiff.grad;

/*
 * Add multiple values
 */
public class AddMany extends GradNode {

    public GradNode [] nodes;

    public AddMany(GradNode [] nodes) {
        this.nodes = nodes;
    }

    @Override
    public double forward() {

        double sum = 0.0;
        for (int i = 0; i < nodes.length; i++) {
            sum += nodes[i].forward();
        }
        return sum;
    }

    /*
     * Propagate the gradient
     */
    @Override
    public void grad(double g) {

        this.grad = 1.0;

        for (int i = 0; i < nodes.length; i++) {
            nodes[i].grad(this.grad * g);
        }
    }

    // toString
    @Override
    public String toString() {
        String s = "AddMany(";
        for (int i = 0; i < nodes.length; i++) {
            s += nodes[i].toString();
            if (i < nodes.length - 1) {
                s += " + ";
            }
        }
        s += ") = " + this.forward() + "";
        return s;
    }
}
