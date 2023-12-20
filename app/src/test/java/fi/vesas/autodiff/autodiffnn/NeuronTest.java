package fi.vesas.autodiff.autodiffnn;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

import fi.vesas.autodiff.grad.Mul;
import fi.vesas.autodiff.grad.Value;

class NeuronTest {

    private static final double h = 0.0001;
    @Test
    void test1() {

        Value v1 = new Value(2.0);
        Value v2 = new Value(3.0);
        
        Neuron n = new Neuron( new Value[] { v1, v2}, "neuron1");
        n.setActivation(new Sigmoid());
        
        n.weights[0].value = 0.0;
        n.weights[1].value = 1.0;
        n.bias.value = 0.0;

        double val = n.forward();

        n.backward();
        double v1Grad = v1.grad;
        double v2Grad = v2.grad;

        v2.value += h;
        double val2 = n.forward();

        double diff = (val2 - val) / h;

        assertEquals(0.0, v1Grad, 0.0001);
        assertEquals(diff, v2Grad, 0.0001);
        
    }

    @Test
    void test2() {

        Value v1 = new Value(-2.0);
        Value v2 = new Value(-3.0);
        
        Neuron n = new Neuron( new Value[] { v1, v2}, "neuron1");
        n.setActivation(new Sigmoid());

        n.weights[0].value = 0.0;
        n.weights[1].value = 1.0;
        n.bias.value = 0.0;

        Value v3 = new Value(1.0);
        Mul m = new Mul(n.activation, v3);

        m.backward();
        n.updateWeights(1000.0);

        double w1 =  n.weights[0].value;
        double w2 =  n.weights[1].value;
        double b =  n.bias.value;

        double val3 = m.forward();

        int qwe = 0;
        
    }

    @Test
    void testBiasAffectsOutput() {

        Value v1 = new Value(0.0);
        Value v2 = new Value(0.0);
        
        Neuron n = new Neuron( new Value[] { v1, v2}, "neuron1");
        n.setActivation(new Linear());

        n.weights[0].value = 0.0;
        n.weights[1].value = 0.0;
        n.bias.value = 1.0;

        double val = n.forward();

        assertEquals(1.0, val, 0.0001);
        
    }

}
