package fi.vesas.autodiff.autodiffnn;

import static org.junit.jupiter.api.Assertions.assertEquals;
import org.junit.jupiter.api.Test;

import fi.vesas.autodiff.grad.GradNode;
import fi.vesas.autodiff.grad.Value;

public class DenseLayerTest {
    
    @Test
    public void testLayerBias() {

        GradNode [] inputs = new Value[2];
        inputs[0] = new Value(0.0);
        inputs[1] = new Value(0.0);
        DenseLayer l = new DenseLayer(1);
        l.initialize(inputs);

        l.neurons[0].setActivation(new Linear());

        l.neurons[0].weights[0].value = 0.0;
        l.neurons[0].weights[1].value = 0.0;
        l.neurons[0].bias.value = 1.0;

        double []val = l.forward();

        // just bias coming through
        assertEquals(1.0, val[0], 0.0001);
        

    }
}
