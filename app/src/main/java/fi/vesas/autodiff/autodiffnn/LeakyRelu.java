package fi.vesas.autodiff.autodiffnn;

import fi.vesas.autodiff.grad.GradNode;

public class LeakyRelu extends GradNode {

    public GradNode x;

    public LeakyRelu(GradNode x) {
        this.x = x;
    }

    private static final double NEGATIVE_SLOPE = 0.2;

    public static double value(double x) {
        if(x < 0) {
            return NEGATIVE_SLOPE * x;
        }
        else {
            return x;
        }
    }

    public double forward() {
        return value(this.x.forward());
    }

    @Override
    public void grad(double g) {
        if (this.x.forward() > 0) {
            this.x.grad(g);
        }
        else {
            this.x.grad(NEGATIVE_SLOPE * g);
        }
        
    }

}
