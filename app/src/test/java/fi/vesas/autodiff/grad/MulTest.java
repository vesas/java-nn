package fi.vesas.autodiff.grad;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

public class MulTest {
    
    static final double h = 0.0001f;

    @Test
    public void testGradientMatchesDiff() {
        Value a = new Value(3.0f);
        Value b = new Value(2.0f);

        Mul c = new Mul(a, b);
        c.backward();

        double diff = Util.diff(c, a);

        // check that the backpropagated gradient matches the forward pass estimate
        double grad = a.grad;
        assertEquals(diff, grad, 0.0001);
    }

    @Test
    public void testGradientsMatchParams() {
        Value a = new Value(3.0f);
        Value b = new Value(2.0f);
        new Mul(a, b).backward();
        
        // multiply should send the other value as gradient
        assertEquals(b.value, a.grad, 0.0001);
        assertEquals(a.value, b.grad, 0.0001);
    }

    @Test
    public void testGradientParams() {

        Value a = new Value(-0.22512219684853296);
        Value b = new Value(1.0);

        Mul c = new Mul(a, b);
        
        c.backward();

        // in multiplication, each inputs forward value is the gradient on the other
        assertEquals(1.0, a.grad, 0.0001);
        assertEquals(-0.22512219684853296,  b.grad, 0.0001);

    }
}
