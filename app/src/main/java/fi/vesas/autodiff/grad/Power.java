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

        this.x.grad(power * Math.pow(this.x.forward(), power - 1) * g);
    }

    @Override
    public String toDotString() {
        return "Power()" + " = " + String.format("%.05f", this.forward()) + "";
    }

    @Override
    public GradNode[] getChildren() {
        
        GradNode [] children = {this.x};
        return children;
    }
}
