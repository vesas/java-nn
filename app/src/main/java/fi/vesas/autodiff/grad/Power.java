package fi.vesas.autodiff.grad;

public class Power extends GradNode {
    
    public GradNode x;
    public double power;

    public Power(GradNode x, double power) {
        this.x = x;
        this.power = power;
    }

    @Override
    public double forward() {
        return Math.pow(this.x.forward(), this.power);
    }

    @Override
    public void grad(double g) {

        this.grad = g;
        this.x.grad(power * Math.pow(this.x.forward(), power - 1) * g);
    }

    @Override
    public void zeroGrads(){
        this.grad = 0.0;
        x.zeroGrads();
    }

    @Override
    public String toString() {
        return "Power() = " + String.format("%.05f", this.forward()) + ", grad=" + String.format("%.05f", this.grad) + ")";
    }

    @Override
    public String toDotString() {
        return "Power_" + label;
    }

    @Override
    public GradNode[] getChildren() {
        
        GradNode [] children = {this.x};
        return children;
    }
}
