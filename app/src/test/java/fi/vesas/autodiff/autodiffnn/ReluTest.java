package fi.vesas.autodiff.autodiffnn;


import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

import fi.vesas.autodiff.grad.Value;

public class ReluTest {
    
    @Test
    public void testGradientAtPositive() {
        Value a = new Value(3.0f);
        new Relu(a).backward();
        assertEquals(1.0, a.grad , 0.0001);
    }

    @Test
    public void testGradientAtNegative() {
        Value a = new Value(-4.0f);
        new Relu(a).backward();
        assertEquals(0.0, a.grad , 0.0001);
    }
}
