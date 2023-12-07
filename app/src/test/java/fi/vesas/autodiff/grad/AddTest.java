package fi.vesas.autodiff.grad;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

public class AddTest {
    
    @Test
    void test1() {
        Value a = new Value(3.0f);
        Value b = new Value(1.0f);
        Add c = new Add(a, b);
        
        c.backward();

        double diff = Util.diff(c, a);

        // check that the backprogapaged gradient matches the forward pass estimate
        double grad = a.grad;
        assertEquals(diff, grad, 0.0001);
    }

    @Test
    void test2() {

        // 15.5 + (-1.0) + (-3.0) = 11.5
        Add add = new Add(new Value(-3.0f), new Add(new Value(-1.0f), new Value(15.5f)));
        assertEquals(11.5, add.forward(), 0.0001);
    }}
