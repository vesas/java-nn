package fi.vesas.autodiff.grad;

/*
 * Just a numeric value node
 */
public class Value extends GradNode {
    
    public double value = 0.0;

    public Value(double value) {
        this.value = value;
    }

    public Value(double value, String label) {
        this.value = value;
        this.label = label;
    }

    @Override
    public double forward() {
        return this.value;
    }

    @Override
    public void grad(double g) {
        this.grad += g;
    }

    @Override
    public void zeroGrads(){
        this.grad = 0.0;
    }

    // toString
    @Override
    public String toString() {
        return "Value_" + label + "(" + String.format("%.05f",this.value) + ", grad=" + String.format("%.05f", this.grad) + ")";
    }

    @Override
    public String toDotString() {
        return "Value_" + label;
    }

    @Override
    public GradNode[] getChildren() {
        return new GradNode[0];
    }

    
}
