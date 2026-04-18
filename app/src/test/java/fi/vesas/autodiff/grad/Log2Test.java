package fi.vesas.autodiff.grad;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

public class Log2Test {

    @Test
    void testForwardMatchesLogBase2() {
        Value a = new Value(8.0);
        Log2 log = new Log2(a, "l");

        assertEquals(3.0, log.forward(), 0.0001);
    }

    @Test
    void testGradientMatchesDiffAtOne() {
        Value a = new Value(1.0);
        Log2 log = new Log2(a, "l");

        log.backward();

        assertEquals(Util.diff(log, a), a.grad, 0.0001);
    }

    @Test
    void testGradientMatchesDiffAtFour() {
        Value a = new Value(4.0);
        Log2 log = new Log2(a, "l");

        log.backward();

        assertEquals(Util.diff(log, a), a.grad, 0.0001);
    }
}
