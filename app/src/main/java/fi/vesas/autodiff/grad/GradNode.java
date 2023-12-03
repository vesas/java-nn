package fi.vesas.autodiff.grad;

public abstract class GradNode {

    public double grad = 0.0f;

    /*
     * The feed-forward function of this node.
     */
    public double forward() {
        return 0;
    }

    /**
     * Calculates gradients for this node and its parents.
     */
    public void grad() {
    }

    // starts a backward pass from this node.
    public void backward() {
        // when starting the backward pass, the gradient of the output node
        this.grad = 1.0;
        this.grad();
    }
}
