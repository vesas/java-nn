package fi.vesas.autodiff.loss;

import fi.vesas.autodiff.grad.Add;
import fi.vesas.autodiff.grad.AddMany;
import fi.vesas.autodiff.grad.GradNode;
import fi.vesas.autodiff.grad.Log;
import fi.vesas.autodiff.grad.Mul;
import fi.vesas.autodiff.grad.Sub;
import fi.vesas.autodiff.grad.Value;

public class CrossEntropyLoss implements LossInterface {
 
    public GradNode [] yhat;
    public Value [] truth;

    public GradNode exitNode;

    public CrossEntropyLoss() {}

    public CrossEntropyLoss(GradNode [] inputs) {
        initialize(inputs);
    }

    private void initialize(GradNode [] inputs) {

        this.yhat = inputs;

        this.truth = new Value[inputs.length];

        GradNode [] subs = new GradNode[inputs.length];

        for(int i = 0; i < this.truth.length; i++) {
            this.truth[i] = new Value(0.0, "_e" + i);
            Log log = new Log(inputs[i], "log" + i);
            Mul mul = new Mul(this.truth[i], log, "e" + i);

            Sub sub1 = new Sub(new Value(1.0), this.truth[i], "-sum");
            Sub sub2 = new Sub(new Value(1.0), inputs[i], "-sum");

            Log log2 = new Log(sub2, "log_2_" + i);

            Mul mul2 = new Mul(sub1, log2, "e" + i);

            Add add = new Add(mul, mul2);
            Sub sub = new Sub(new Value(0.0), add, "-sub");
            subs[i] = sub;
        }

        AddMany sum = new AddMany(subs);

        this.exitNode = sum;
    }

    public void setInputs(GradNode [] inputs) {
        initialize(inputs);
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
}
