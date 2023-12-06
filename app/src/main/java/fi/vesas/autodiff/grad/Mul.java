package fi.vesas.autodiff.grad;

public class Mul extends GradNode {
    
    public GradNode x;
    public GradNode y;

    public Mul(GradNode x, GradNode y) {
        this.x = x;
        this.y = y;
    }

    @Override
    public double forward() {
        return this.x.forward() * this.y.forward();
    }

    @Override
    public void grad(double g) {

        this.x.grad(this.y.forward() * g);
        this.y.grad(this.x.forward() * g);
    }

    // toString
    @Override
    public String toString() {
        return "Mul(" + this.x.toString() + " * " + this.y.toString() + ") = " + this.forward() + "";
    }
}
