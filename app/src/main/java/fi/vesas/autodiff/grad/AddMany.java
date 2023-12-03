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
    public void grad() {

        for (int i = 0; i < nodes.length; i++) {
            nodes[i].grad += this.grad;
            nodes[i].grad();
        }
    }
}
