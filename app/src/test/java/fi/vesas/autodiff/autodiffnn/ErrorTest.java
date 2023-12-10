package fi.vesas.autodiff.autodiffnn;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

import fi.vesas.autodiff.grad.Util;
import fi.vesas.autodiff.grad.Value;
import fi.vesas.autodiff.loss.MSELoss;

public class ErrorTest {
    
     @Test
    public void testSimple1() {

         double t1 = 1.0; // truth
         double p1 = 0.4223782380931712; // prediction

         double t2 = -1.0; // truth
         double p2 = -0.9999999999669938; // prediction

         double t3 = 1.0; // truth
         double p3 = 0.59; // prediction

        //  MSE mean squared error
        final double expectedValue = 
            (Math.pow(t1 - p1, 2.0) + Math.pow(t2 - p2, 2.0) + Math.pow(t3 - p3, 2.0) ) / 3.0;

        Value [] inputs = new Value[3];
        inputs[0] = new Value(p1);
        inputs[1] = new Value(p2);
        inputs[2] = new Value(p3);

        System.out.println(String.format("inputs: %.9f %.9f %.9f", inputs[0].value, inputs[1].value, inputs[2].value));
        

        MSELoss e = new MSELoss(inputs);

        e.setTruth(t1, t2, t3);

        double res = e.forward();

        assertEquals(expectedValue, res, 0.0000000000000001);


        // 
        // check gradients
        // 

        e.exitNode.backward();
        double grad = inputs[0].grad;

        double diff = Util.diff(e.exitNode, inputs[0]);

        assertEquals(diff, grad, 0.000001);

        // 
        // learning
        // 

        inputs[0].value -= inputs[0].grad * 0.1;
        inputs[1].value -= inputs[1].grad * 0.1;
        inputs[2].value -= inputs[2].grad * 0.1;

        System.out.println(String.format("inputs: %.9f %.9f %.9f", inputs[0].value, inputs[1].value, inputs[2].value));

        double res2 = e.forward();

        e.exitNode.backward();

        inputs[0].value -= inputs[0].grad * 0.1;
        inputs[1].value -= inputs[1].grad * 0.1;
        inputs[2].value -= inputs[2].grad * 0.1;

        System.out.println(String.format("inputs: %.9f %.9f %.9f", inputs[0].value, inputs[1].value, inputs[2].value));

        double res3 = e.forward();
    }
}
