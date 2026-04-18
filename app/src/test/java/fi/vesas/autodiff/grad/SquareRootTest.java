package fi.vesas.autodiff.grad;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

public class SquareRootTest {

    @Test
    void testForwardMatchesMathSqrt() {
        Value a = new Value(9.0);
        SquareRoot sqrt = new SquareRoot(a);

        assertEquals(3.0, sqrt.forward(), 0.0001);
    }

    @Test
    void testGradientMatchesDiff() {
        Value a = new Value(4.0);
        SquareRoot sqrt = new SquareRoot(a);

        sqrt.backward();

        assertEquals(Util.diff(sqrt, a), a.grad, 0.0001);
    }

    @Test
    void testGradientMatchesDiffAtOne() {
        Value a = new Value(1.0);
        SquareRoot sqrt = new SquareRoot(a);

        sqrt.backward();

        assertEquals(Util.diff(sqrt, a), a.grad, 0.0001);
    }
}
