package fi.vesas.autodiff.grad;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

public class SubTest {
    
    @Test
    public void test1() {
        Value a = new Value(-3.0f);
        Value b = new Value(-2.0f);
        Sub c = new Sub(a, b);
        
        c.backward();

        double diff = Util.diff(c, b);

        // check that the backprogapaged gradient matches the forward pass estimate
        double grad = b.grad;
        assertEquals(diff, grad, 0.0001);
    }

    @Test
    public void test2() {
        Value a = new Value(2.0f);
        Value b = new Value(4.0f);
        Sub c = new Sub(a, b);
        
        c.backward();

        double diff = Util.diff(c, b);

        // check that the backprogapaged gradient matches the forward pass estimate
        double grad = b.grad;
        assertEquals(diff, grad, 0.0001);
    }
}