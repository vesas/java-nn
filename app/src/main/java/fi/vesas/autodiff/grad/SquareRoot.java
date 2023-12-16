package fi.vesas.autodiff.grad;

public class SquareRoot extends GradNode {
    
    public GradNode x;
    
    public SquareRoot(GradNode x) {
        this.x = x;
    }

    @Override
    public double forward() {
        return Math.sqrt(this.x.forward());
    }

    @Override
    public void grad(double g) {

        this.grad = g;
        this.x.grad((1.0 / (2.0 * Math.sqrt(this.x.forward()))) * g);
    }

    @Override
    public void zeroGrads(){
        this.grad = 0.0;
        x.zeroGrads();
    }

    // toString
    @Override
    public String toString() {
        return "SquareRoot() = " + String.format("%.05f", this.forward()) + "";
    }

    @Override
    public String toDotString() {
        return "SquareRoot_" + label;
    }

    @Override
    public GradNode[] getChildren() {
        
        GradNode [] children = {this.x};
        return children;
    }
}