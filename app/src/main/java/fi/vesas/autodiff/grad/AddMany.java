package fi.vesas.autodiff.grad;

/*
 * Add multiple values
 */
public class AddMany extends GradNode {

    public GradNode [] nodes;

    public AddMany(GradNode [] nodes) {
        this.nodes = nodes;
    }

    public AddMany(GradNode [] nodes, String label) {
        this.nodes = nodes;
        this.label = label;
    }

    @Override
    public double forward() {

        double sum = 0.0;
        for (int i = 0; i < nodes.length; i++) {
            double value = nodes[i].forward();
            sum = sum + value;
        }
        return sum;
    }

    /*
     * Propagate the gradient
     */
    @Override
    public void grad(double g) {

        this.grad = g;

        for (int i = 0; i < nodes.length; i++) {
            nodes[i].grad(this.grad * 1.0);
        }
    }

    @Override
    public void zeroGrads(){
        this.grad = 0.0;
        for (int i = 0; i < nodes.length; i++) {
            nodes[i].zeroGrads();
        }
    }

    // toString
    @Override
    public String toString() {
        String s = "AddMany_" + label + "() = " + String.format("%.05f", this.forward()) + ", grad=" + String.format("%.08f", this.grad) + ")";
        return s;
    }

    @Override
    public String toDotString() {
        return "AddMany_" + label;
    }

    @Override
    public GradNode[] getChildren() {
        
        return nodes;
    }
}
