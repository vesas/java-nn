package fi.vesas.autodiff.autodiffnn;


import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

import fi.vesas.autodiff.grad.Util;
import fi.vesas.autodiff.grad.Value;

public class ReluTest {
    
    static final double h = 0.0001f;

    @Test
    public void test1() {
        Value a = new Value(3.0f);
        Linear r = new Linear(a);
        
        r.backward();

        double diff = Util.diff(r, a);

        // check that the backpropagated gradient matches the forward pass estimate
        double grad = a.grad;
        assertEquals(diff, grad , 0.0001);
    }

    @Test
    public void test2() {
        Value a = new Value(-4.0f);
        Linear r = new Linear(a);
        
        r.backward();

        double diff = Util.diff(r, a);

        // check that the backpropagated gradient matches the forward pass estimate
        double grad = a.grad;
        assertEquals(diff, grad , 0.0001);
    }
}
