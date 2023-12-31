package fi.vesas.autodiff.grad;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

class GradTest {
    
    /*
     * Test if two same input values are calculated correctly, they refer to the same object.
     */
    @Test
    void testBug() {
        Value a = new Value(3.0f);
        Add b = new Add(a, a);
        
        b.backward();

        assertEquals(2.0, a.grad, 0.0001);
        assertEquals(6.0, b.forward(), 0.0001);

    }

    /*
     * Test two same input values as separate values
     */
    @Test
    void testBug2() {
        Value a = new Value(3.0f);
        Add b = new Add(a,  new Value(3.0f));
        
        b.backward();

        assertEquals(1.0, a.grad, 0.0001);
        assertEquals(6.0, b.forward(), 0.0001);

    }

    static final double h = 0.0001f;

    @Test
    public void test1() {
        // we are testing gradient calculation for value Mul e 
        Value a = new Value(3.0f);
        Value c = new Value(10.0f);
        Add d = new Add(a, a);
        Mul e = new Mul(d, c);
        
        e.backward();

        double diff = Util.diff(e, c);

        // check that the backpropagated gradient matches the forward pass estimate
        assertEquals(diff, c.grad, 0.0001);
    }

}
