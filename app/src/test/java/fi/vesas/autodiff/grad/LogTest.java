package fi.vesas.autodiff.grad;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

import org.junit.jupiter.api.Test;

public class LogTest {

    @Test
    void testForwardMatchesMathLog() {
        Value a = new Value(2.5);
        Log log = new Log(a, "l");

        assertEquals(Math.log(2.5), log.forward(), 0.0001);
    }

    @Test
    void testGradientMatchesDiff() {
        Value a = new Value(2.5);
        Log log = new Log(a, "l");

        log.backward();

        assertEquals(Util.diff(log, a), a.grad, 0.0001);
    }

    @Test
    void testForwardGuardAtZero() {
        Value a = new Value(0.0);
        Log log = new Log(a, "l");

        double result = log.forward();

        assertFalse(Double.isInfinite(result), "guard should replace -Infinity");
        assertFalse(Double.isNaN(result));
        assertEquals(-Double.MAX_VALUE, result);
    }
}
