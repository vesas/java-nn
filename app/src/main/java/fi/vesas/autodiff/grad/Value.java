package fi.vesas.autodiff.grad;

/*
 * Just a numeric value node
 */
public class Value extends GradNode {
    
    public double value = 0.0;

    public Value(double value) {
        this.value = value;
    }

    @Override
    public double forward() {
        return this.value;
    }

    @Override
    public void grad() {
    }
}
