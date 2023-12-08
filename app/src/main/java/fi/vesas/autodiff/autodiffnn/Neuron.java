package fi.vesas.autodiff.autodiffnn;

import fi.vesas.autodiff.grad.AddMany;
import fi.vesas.autodiff.grad.GradNode;
import fi.vesas.autodiff.grad.Mul;
import fi.vesas.autodiff.grad.Value;
import fi.vesas.autodiff.util.Stats;
import fi.vesas.autodiff.util.Util;

public class Neuron {
    
    public Value [] weights;
    public Mul [] muls;
    public AddMany adds;
    public Value bias;

    public Tanh tanh = null;

    public GradNode [] inputs;
    private String label;

    public Neuron(GradNode [] inputs, String label) {

        this.label = label;

        this.weights = new Value[inputs.length];
        this.muls = new Mul[inputs.length];
        this.inputs = inputs;

        for(int i = 0; i < inputs.length; i++) {

            this.weights[i] = new Value(Util.rangeRand(-0.9, 0.9), this.label + "w" + i);
            this.muls[i] = new Mul(this.weights[i], inputs[i], this.label + "m" + i);
        }
        this.bias = new Value(Util.rangeRand(-0.8, 0.8), this.label + "b");

        GradNode[] result = new GradNode[this.muls.length + 1];
        System.arraycopy(this.muls, 0, result, 0, this.muls.length);
        result[result.length - 1] = this.bias;

        adds = new AddMany(result, this.label + "am");

        this.tanh = new Tanh(adds, this.label + "t");

    }

    public double forward() {
        return this.tanh.forward();
    }

    public void zeroGrads() {

        for (int i = 0; i < inputs.length; i++ ) {
            weights[i].zeroGrads();
            muls[i].zeroGrads();
        }
        this.bias.zeroGrads();
        this.adds.zeroGrads();
        this.tanh.zeroGrads();
	}

    public void backward() {

        this.tanh.backward();
    }

    public void recordWeights() {
            
            for (int i = 0; i < inputs.length; i++ ) {
                Stats.recordNeuronWeights(label, weights[i].value);
            }
            Stats.recordNeuronBias(label, this.bias.value);
    }

    public void updateWeights(double learningRate) {

        for (int i = 0; i < weights.length; i++) {
            this.weights[i].value -= this.weights[i].grad * learningRate;
        }

        this.bias.value -= this.bias.grad * learningRate;
    }

    @Override
    public String toString() {
        StringBuffer buf = new StringBuffer();
        buf.append("Neuron ");
        buf.append(this.label);
        buf.append(" [weights=");

        for (int i = 0; i < weights.length; i++) {
            buf.append(" " + i + "=");
            buf.append(this.weights[i].value);
        }

        buf.append(", bias=");
        buf.append(bias.value);
        buf.append("]");

        buf.append(" [inputs=");
        for (int i = 0; i < inputs.length; i++) {
            buf.append(" " + i + "=");
            buf.append(this.inputs[i].forward());
        }
        buf.append("]");

        return buf.toString();
    }

}