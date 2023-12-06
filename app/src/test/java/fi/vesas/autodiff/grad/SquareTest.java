package fi.vesas.autodiff.grad;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

public class SquareTest {
    
    static final double h = 0.0001f;

    @Test
    public void testForward1() {

        Value a = new Value(-9.0);
        Square sq = new Square(new Sub(a, new Value(2.0)));

        // (-9 - 2) * (-9 - 2) = 121
        double val1 = sq.forward();

        // tweak value a bit
        a.value += h;

        // do forward pass again
        double val2 = sq.forward();

        // how much did the value change
        double diff = (val2 - val1) / h;

        sq.backward();

        assertEquals(121, val1, 0.0001);

    }
    @Test
    public void test1() {
        Value a = new Value(3.0f);
        Square s = new Square(a);
        
        s.backward();

        double val1 = s.forward();

        // tweak value a bit
        a.value += h;

        // do forward pass again
        double val2 = s.forward();

        // how much did the value change
        double diff = (val2 - val1) / h;

        // check that the backprogapaged gradient matches the forward pass estimate
        double grad = a.grad;
        assertEquals(diff, grad , 0.0001);
    }

    @Test
    public void test2() {
        Value a = new Value(-9.0f);
        Square s = new Square(a);
        
        s.backward();

        double val1 = s.forward();

        // tweak value a bit
        a.value += h;

        // do forward pass again
        double val2 = s.forward();

        // how much did the value change
        double diff = (val2 - val1) / h;

        // check that the backprogapaged gradient matches the forward pass estimate
        double grad = a.grad;
        assertEquals(diff, grad , 0.0001);
    }
}
