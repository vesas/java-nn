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
    public void grad() {
        this.x.grad += this.y.forward() * this.grad;
        this.y.grad += this.x.forward() * this.grad;

        this.x.grad();
        this.y.grad();
    }
}
