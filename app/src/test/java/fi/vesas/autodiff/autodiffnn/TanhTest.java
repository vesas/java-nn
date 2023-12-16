package fi.vesas.autodiff.autodiffnn;


import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

import fi.vesas.autodiff.grad.AddMany;
import fi.vesas.autodiff.grad.GradNode;
import fi.vesas.autodiff.grad.Sub;
import fi.vesas.autodiff.grad.Util;
import fi.vesas.autodiff.grad.Value;

public class TanhTest {
    
    static final double h = 0.0001f;

    @Test
    public void test1() {
        Value a = new Value(3.0f);
        Tanh t = new Tanh(a);
        
        t.backward();

        double diff = Util.diff(t, a);

        // check that the backpropagated gradient matches the forward pass estimate
        double grad = a.grad;
        assertEquals(diff, grad , 0.0001);
    }

    @Test
    public void test2() {
        Value a = new Value(0.0f);
        Tanh t = new Tanh(a);
        
        t.backward();

        double diff = Util.diff(t, a);

        // check that the backpropagated gradient matches the forward pass estimate
        double grad = a.grad;
        assertEquals(diff, grad , 0.0001);
    }

    @Test
    public void test3() {
        Value a = new Value(-222.0f);
        Tanh t = new Tanh(a);
        
        t.backward();

        double diff = Util.diff(t, a);

        // check that the backpropagated gradient matches the forward pass estimate
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
        Tanh t = new Tanh(add);
        Sub s = new Sub(t, new Value(0.5));
        
        s.backward();

        double diff = Util.diff(s, a);

        // check that the backpropagated gradient matches the forward pass estimate
        double grad = a.grad;
        assertEquals(diff, grad , 0.0001);
    }
}
