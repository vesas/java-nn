package fi.vesas.autodiff.grad;

public class Log extends GradNode {

    public GradNode x;

    public Log(GradNode x, String label) {
        this.x = x;
        this.label = label;
    }

    @Override
    public double forward() {

        double temp = this.x.forward();
        double result = Math.log(temp);
        if(Double.isInfinite(result) && result < 0.0) {
            return -Double.MAX_VALUE;
        }
        return result;
    }

    @Override
    public void grad(double g) {

        this.grad = g * (1.0 / this.x.forward());
        this.x.grad(this.grad);
    }

    @Override
    public void zeroGrads() {
        this.grad = 0.0;
        this.x.zeroGrads();
    }

    @Override
    public String toString() { 
        return "Log2() = " + String.format("%.05f", this.forward()) + ", grad=" + String.format("%.05f", this.grad) + ")";
    }

    @Override
    public String toDotString() {
        return "Log2_" + label;
    }

    @Override
    public GradNode[] getChildren() {
        
        GradNode [] children = {this.x};
        return children;
    }
}
