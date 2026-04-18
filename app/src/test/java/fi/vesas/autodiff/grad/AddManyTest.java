package fi.vesas.autodiff.grad;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

public class AddManyTest {

    @Test
    void testForwardSumsAllChildren() {
        Value a = new Value(1.0);
        Value b = new Value(2.5);
        Value c = new Value(-0.5);
        AddMany sum = new AddMany(new GradNode[] { a, b, c });

        assertEquals(3.0, sum.forward(), 0.0001);
    }

    @Test
    void testGradientIsOneForEachChild() {
        Value a = new Value(1.0);
        Value b = new Value(2.0);
        Value c = new Value(3.0);
        AddMany sum = new AddMany(new GradNode[] { a, b, c });

        sum.backward();

        assertEquals(1.0, a.grad, 0.0001);
        assertEquals(1.0, b.grad, 0.0001);
        assertEquals(1.0, c.grad, 0.0001);
    }

    @Test
    void testGradientMatchesDiff() {
        Value a = new Value(1.5);
        Value b = new Value(-2.25);
        Value c = new Value(0.75);
        AddMany sum = new AddMany(new GradNode[] { a, b, c });

        sum.backward();

        assertEquals(Util.diff(sum, a), a.grad, 0.0001);
        assertEquals(Util.diff(sum, b), b.grad, 0.0001);
        assertEquals(Util.diff(sum, c), c.grad, 0.0001);
    }
}
