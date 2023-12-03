package fi.vesas.autodiff.autodiffnn;

import fi.vesas.autodiff.grad.GradNode;
import fi.vesas.autodiff.grad.Mul;
import fi.vesas.autodiff.grad.AddMany;
import fi.vesas.autodiff.grad.Value;
import fi.vesas.autodiff.util.Util;

public class Neuron {
    
    public Value [] weights;
    public Mul [] muls;
    public AddMany adds;
    public Value bias;
    public Value output;

    public Tanh tanh = null;

    public GradNode [] inputs;
    public double biasGrad;
    private String label;

    public Neuron(GradNode [] inputs, String label) {

        this.label = label;

        this.weights = new Value[inputs.length];
        this.muls = new Mul[inputs.length];
        this.inputs = inputs;

        for(int i = 0; i < inputs.length; i++) {

            this.weights[i] = new Value(Util.rangeRand(-1.0, 1.0));
            this.muls[i] = new Mul(this.weights[i], inputs[i]);
        }
        this.bias = new Value(Util.rangeRand(-1.0, 1.0));

        GradNode[] result = new GradNode[this.muls.length + 1];
        System.arraycopy(this.muls, 0, result, 0, this.muls.length);
        result[result.length - 1] = this.bias;

        adds = new AddMany(result);

        this.tanh = new Tanh(adds);

    }

    public double forward() {
        return this.tanh.forward();
    }

    public void backward() {

        this.tanh.backward();
    }


    public void updateWeights(double learningRate) {

        for (int i = 0; i < weights.length; i++) {
            this.weights[i].value += (this.weights[i].grad * learningRate);
        }

        this.bias.value += (this.bias.grad * learningRate);
    }

    @Override
    public String toString() {
        return "";
    }

}
