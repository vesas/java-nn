package fi.vesas.autodiff.grad;

public class Square extends GradNode {
    
    public GradNode x;
    
    public Square(GradNode x) {
        this.x = x;
    }

    @Override
    public double forward() {
        double temp = this.x.forward();
        return temp * temp;
    }

    @Override
    public void grad(double g) {
        // double temp = this.x.forward() * this.grad * 2.0;
        // this.x.grad += temp;

        this.x.grad(g * 2.0 * this.x.forward());
    }

    // toString
    @Override
    public String toString() {
        return "Square(" + this.x.toString() + ") = " + this.forward() + "";
    }
}