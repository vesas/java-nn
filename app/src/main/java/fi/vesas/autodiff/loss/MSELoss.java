package fi.vesas.autodiff.loss;

import fi.vesas.autodiff.grad.AddMany;
import fi.vesas.autodiff.grad.GradNode;
import fi.vesas.autodiff.grad.Mul;
import fi.vesas.autodiff.grad.Square;
import fi.vesas.autodiff.grad.Sub;
import fi.vesas.autodiff.grad.Value;

public class MSELoss implements LossInterface {

    public GradNode [] yhat;
    public Value [] truth;

    public GradNode exitNode;

    /*
     * Inputs are the network nodes for which we want to calculate the error, usually output of the neural network
     */
    public MSELoss(GradNode [] inputs) {
        this.yhat = inputs;

        this.truth = new Value[inputs.length];

        GradNode [] squares = new GradNode[inputs.length];

        for(int i = 0; i < this.truth.length; i++) {
            this.truth[i] = new Value(0.0, "_e" + i);

            squares[i] = new Square(new Sub(this.truth[i], inputs[i], "e" + i), "e" + i);
        }

        AddMany sum = new AddMany(squares);
        Mul mul = new Mul(sum, new Value( 1.0 / squares.length));

        this.exitNode = mul;
        // add this for mean squared root error
        // this.exitNode = new SquareRoot(mul);
    }

    public void setTruth(double ... t) {
        
        for(int i = 0; i < t.length; i++) {
            this.truth[i].value = t[i];
        }
    }

    public double forward() {
        return this.exitNode.forward();
    }

    public void zeroGrads() {
		this.exitNode.zeroGrads();
	}

    public void backward(double [] y) {

        // set the ground truth values to the loss function
        for (int i = 0; i < y.length; i++) {
            this.truth[i].value = y[i];
            this.truth[i].grad = 0.0;
        }

        // do the backward pass
        exitNode.backward();

    }



    public void debug() {

        System.out.println("Error debug");
        double sum = 0.0;
        for(int i = 0; i < this.truth.length; i++) {

            double temp = this.truth[i].value - this.yhat[i].forward();

            sum += temp * temp;
            System.out.println("yhat " + this.yhat[i].forward() + " truth " + i + " " + this.truth[i].value);
        }

        System.out.println("MSE: " + (sum / this.truth.length));
    }

}
