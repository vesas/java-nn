package fi.vesas.autodiff.autodiffnn;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

import fi.vesas.autodiff.grad.Value;

public class LeakyReluTest {

    @Test
    public void testForwardAtPositive() {
        Value a = new Value(3.0);
        LeakyRelu lr = new LeakyRelu(a);

        assertEquals(3.0, lr.forward(), 0.0001);
    }

    @Test
    public void testForwardAtNegative() {
        Value a = new Value(-4.0);
        LeakyRelu lr = new LeakyRelu(a);

        // 0.2 * -4.0 = -0.8
        assertEquals(-0.8, lr.forward(), 0.0001);
    }

    @Test
    public void testGradientAtPositive() {
        Value a = new Value(3.0);
        new LeakyRelu(a).backward();

        assertEquals(1.0, a.grad, 0.0001);
    }

    @Test
    public void testGradientAtNegative() {
        Value a = new Value(-4.0);
        new LeakyRelu(a).backward();

        assertEquals(0.2, a.grad, 0.0001);
    }
}
