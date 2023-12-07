package fi.vesas.autodiff.grad;

public class Mul extends GradNode {
    
    public GradNode x;
    public GradNode y;

    public Mul(GradNode x, GradNode y) {
        this.x = x;
        this.y = y;
    }

    public Mul(GradNode x, GradNode y, String label) {
        this.x = x;
        this.y = y;
        this.label = label;
    }

    @Override
    public double forward() {
        return this.x.forward() * this.y.forward();
    }

    @Override
    public void grad(double g) {

        this.grad = g;
        this.x.grad(this.y.forward() * g);
        this.y.grad(this.x.forward() * g);
    }

    // toString
    @Override
    public String toString() {
        return "Mul() = " + String.format("%.05f", this.forward()) + ", grad=" + String.format("%.08f", this.grad) + ")";
    }

    @Override
    public String toDotString() {
        return "Mul_" + label;
    }

    @Override
    public GradNode[] getChildren() {
        
        GradNode [] children = {this.x, this.y};
        return children;
    }

}
