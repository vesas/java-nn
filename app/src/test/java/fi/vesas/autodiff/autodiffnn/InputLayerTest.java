package fi.vesas.autodiff.autodiffnn;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

import fi.vesas.autodiff.grad.Value;

public class InputLayerTest {

    @Test
    public void testOutputsLengthMatchesSize() {
        InputLayer layer = new InputLayer(5);

        assertEquals(5, layer.getOutputs().length);
    }

    @Test
    public void testSetInputValuesPopulatesOutputs() {
        InputLayer layer = new InputLayer(3);
        double[] inputs = { 1.5, -2.25, 0.75 };

        layer.setInputValues(inputs);

        Value[] outputs = layer.getOutputs();
        assertEquals(1.5, outputs[0].forward(), 1e-12);
        assertEquals(-2.25, outputs[1].forward(), 1e-12);
        assertEquals(0.75, outputs[2].forward(), 1e-12);
    }

    @Test
    public void testInitialValuesAreZero() {
        InputLayer layer = new InputLayer(2);
        Value[] outputs = layer.getOutputs();

        assertEquals(0.0, outputs[0].forward(), 1e-12);
        assertEquals(0.0, outputs[1].forward(), 1e-12);
    }
}
