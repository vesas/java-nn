package fi.vesas.autodiff.loss;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import fi.vesas.autodiff.grad.GradNode;
import fi.vesas.autodiff.grad.Value;

public class CrossEntropyLossTest {
    
    @Test
    public void test1() {
        double [] preds = {0.0, 0.0, 0.0};
        double [] truth = {0.0, 0.0, 0.0};

        GradNode [] inputs = new GradNode[3];
        inputs[0] = new Value(truth[0]);
        inputs[1] = new Value(truth[1]);
        inputs[2] = new Value(truth[2]);

        CrossEntropyLoss e = new CrossEntropyLoss(inputs);

        e.setTruth(preds);

        double res = e.forward();

        assertEquals(0.00000000, res, 0.00000001);
    }

    @Test
    // TODO: check
    public void test2() {
        double [] preds = {0.0, 0.9};
        double [] truth = {0.0, 1.0};

        double result = Math.log(0.9) / Math.log(2);

        GradNode [] inputs = new GradNode[2];
        inputs[0] = new Value(preds[0]);
        inputs[1] = new Value(preds[1]);

        CrossEntropyLoss e = new CrossEntropyLoss(inputs);

        e.setTruth(truth);

        double res = e.forward();

        assertEquals(0.10536051565782628, res, 0.0001);
    }

    static public double h = 0.00001;
    
    @Test
    public void testBug() {
        double [] preds = {0.9917027567108665, 0.9915823892516565};
        double [] truth = {1.0, 0.0};

        GradNode [] inputs = new GradNode[2];
        inputs[0] = new Value(preds[0]);
        inputs[1] = new Value(preds[1]);

        CrossEntropyLoss e = new CrossEntropyLoss(inputs);

        e.backward(truth);
        double grad = inputs[0].grad;

        e.setTruth(truth);

        double res = e.forward();

        ((Value)inputs[0]).value = Math.max(-0.99999999999999, ((Value)inputs[0]).value - (grad * 0.01));
        ((Value)inputs[0]).value = Math.min(0.99999999999999, ((Value)inputs[0]).value - (grad * 0.01));

        double res2 = e.forward();

        double diff = (res2 - res) / h;
        
        // loss should have gone down
        assertTrue(res2 < res);
        // assertEquals(diff, grad, 0.001);
    }

}
