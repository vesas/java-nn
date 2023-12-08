package fi.vesas.autodiff.grad;

public class Square extends GradNode {
    
    public GradNode x;
    
    public Square(GradNode x) {
        this.x = x;
    }

    public Square(GradNode x, String label) {
        this.x = x;
        this.label = label;
    }

    @Override
    public double forward() {
        double temp = this.x.forward();
        return temp * temp;
    }

    @Override
    public void grad(double g) {

        this.grad = g;

        double temp = this.x.forward() * g * 2.0;
        this.x.grad(temp);
    }

    @Override
    public void zeroGrads(){
        this.grad = 0.0;
        x.zeroGrads();
    }

    // toString
    @Override
    public String toString() {
        return "Square_" + label + "() = " + String.format("%.05f", this.forward()) + ", grad=" + String.format("%.05f", this.grad) + ")";
    }

    @Override
    public String toDotString() {
        return "Square_" + label;
    }

    @Override
    public GradNode[] getChildren() {
        
        GradNode [] children = {this.x};
        return children;
    }
}