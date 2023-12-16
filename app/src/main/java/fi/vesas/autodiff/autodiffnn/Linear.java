package fi.vesas.autodiff.autodiffnn;

import fi.vesas.autodiff.grad.GradNode;

/*
 * Linear activation, can be used for example to debug
 */
public class Linear extends GradNode implements Activation {

    public GradNode x;

    public Linear() {
    }

    public Linear(GradNode x) {
        this.x = x;
    }

    @Override
    public void setInput(GradNode x) {
        this.x = x;
    }

    public double forward() {
        return this.x.forward();
    }

    @Override
    public void grad(double g) {
        this.grad = g;
        this.x.grad(g);
    }

    @Override
    public void zeroGrads(){
        this.grad = 0.0;
        this.x.zeroGrads();
    }

    @Override
    public String toDotString() {
        return "Linear";
    }

    @Override
    public GradNode[] getChildren() {
        
        GradNode [] children = {this.x};
        return children;
    }

    

}
