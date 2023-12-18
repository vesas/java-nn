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

        // check that the backpropagated gradient matches the forward pass estimate
        double grad = a.grad;
        assertEquals(diff, grad , 0.0001);
    }

    @Test
    public void test2() {
        Value a = new Value(0.0f);
        Sigmoid s = new Sigmoid(a);
        
        s.backward();

        double diff = Util.diff(s, a);

        // check that the backpropagated gradient matches the forward pass estimate
        double grad = a.grad;
        assertEquals(diff, grad , 0.0001);
    }

    @Test
    public void test3() {
        Value a = new Value(-222.0f);
        Sigmoid s = new Sigmoid(a);
        
        s.backward();

        double diff = Util.diff(s, a);

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
        Sigmoid s = new Sigmoid(add);
        Sub sub = new Sub(s, new Value(0.5));
        
        sub.backward();

        double diff = Util.diff(sub, a);

        // check that the backpropagated gradient matches the forward pass estimate
        double grad = a.grad;
        assertEquals(diff, grad , 0.0001);
    }

    @Test
    public void testGradientAtZero() {

        Value v = new Value(0.0f);
        Sigmoid s = new Sigmoid(v);
        s.backward();

        // sigmoid should have gradient 0.25 at 0.0
        assertEquals(0.25, v.grad , 0.0001);

    }

    @Test
    public void testGradientAtLargeNegative() {

        Value v = new Value(-100000.0f);
        Sigmoid s = new Sigmoid(v);
        s.backward();

        // sigmoid should have sent a gradient to value near 0 at a large negative input value
        assertEquals(0.0, v.grad , 0.001);
    }

    @Test
    public void testGradientAtLargePositive() {

        Value v = new Value(100000.0f);
        Sigmoid s = new Sigmoid(v);
        s.backward();

        // sigmoid should have sent a gradient to value near 0 at a large input value
        assertEquals(0.0, v.grad , 0.001);

    }
}
