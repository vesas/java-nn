package fi.vesas.autodiff.grad;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

public class MulTest {
    
    static final double h = 0.0001f;

    @Test
    public void test1() {
        Value a = new Value(3.0f);
        Value b = new Value(2.0f);
        Mul c = new Mul(a, b);
        
        c.backward();

        double diff = Util.diff(c, a);

        // check that the backprogapaged gradient matches the forward pass estimate
        double grad = a.grad;
        assertEquals(diff, grad, 0.0001);
    }

    @Test
    public void test2() {

        Value a = new Value(-0.22512219684853296);
        Value b = new Value(1.0);

        Mul c = new Mul(a, b);
        
        c.backward();

        int qwe  = 0;

    }
}
