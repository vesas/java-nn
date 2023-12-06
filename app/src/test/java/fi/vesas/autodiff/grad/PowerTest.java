package fi.vesas.autodiff.grad;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

public class PowerTest {
    
    static final double h = 0.0001f;

    @Test
    public void test1() {
        Value a = new Value(3.0f);
        Power s = new Power(a, -1.0f);
        
        s.backward();

        double diff = Util.diff(s, a);

        // check that the backprogapaged gradient matches the forward pass estimate
        double grad = a.grad;
        assertEquals(diff, grad , 0.0001);
    }

    @Test
    public void test2() {
        Value a = new Value(-9.0f);
        Square s = new Square(a);
        
        s.backward();

        double diff = Util.diff(s, a);

        // check that the backprogapaged gradient matches the forward pass estimate
        double grad = a.grad;
        assertEquals(diff, grad , 0.0001);
    }
}
