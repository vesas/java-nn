package fi.vesas.autodiff.autodiffnn;


import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

import fi.vesas.autodiff.grad.AddMany;
import fi.vesas.autodiff.grad.GradNode;
import fi.vesas.autodiff.grad.Sub;
import fi.vesas.autodiff.grad.Util;
import fi.vesas.autodiff.grad.Value;

public class SigmoidTest {
    
    static final double h = 0.0001f;

    @Test
    public void test1() {
        Value a = new Value(3.0f);
        Sigmoid s = new Sigmoid(a);
        
        s.backward();

        double diff = Util.diff(s, a);

        // check that the backprogapaged gradient matches the forward pass estimate
        double grad = a.grad;
        assertEquals(diff, grad , 0.0001);
    }

    @Test
    public void test2() {
        Value a = new Value(0.0f);
        Sigmoid s = new Sigmoid(a);
        
        s.backward();

        double diff = Util.diff(s, a);

        // check that the backprogapaged gradient matches the forward pass estimate
        double grad = a.grad;
        assertEquals(diff, grad , 0.0001);
    }

    @Test
    public void test3() {
        Value a = new Value(-222.0f);
        Sigmoid s = new Sigmoid(a);
        
        s.backward();

        double diff = Util.diff(s, a);

        // check that the backprogapaged gradient matches the forward pass estimate
        double grad = a.grad;
        assertEquals(diff, grad , 0.0001);
    }

    @Test
    public void test4() {
        Value a = new Value(0.1);
        Value b = new Value(0.2);
        Value c = new Value(0.3);
        Value d = new Value(-0.3);
        AddMany add = new AddMany(new GradNode[] {a, b, c, d});
        Sigmoid s = new Sigmoid(add);
        Sub sub = new Sub(s, new Value(0.5));
        
        sub.backward();

        double diff = Util.diff(sub, a);

        // check that the backprogapaged gradient matches the forward pass estimate
        double grad = a.grad;
        assertEquals(diff, grad , 0.0001);
    }
}
