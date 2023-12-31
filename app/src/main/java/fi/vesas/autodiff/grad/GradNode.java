package fi.vesas.autodiff.grad;

public abstract class GradNode {

    public double grad = 0.0f;
    public String label = "";

    /*
     * The feed-forward function of this node.
     */
    public double forward() {
        return 0;
    }

    /**
     * Calculates gradients for this node and its parents.
     */
    public abstract void grad(double g);

    public abstract void zeroGrads();

    // starts a backward pass from this node.
    public void backward() {
        // when starting the backward pass, the gradient of the output node
        this.grad(1.0);
    }

    // for dotfile generation
    public abstract String toDotString();

    // for dotfile generation
    public abstract GradNode[] getChildren();
}
