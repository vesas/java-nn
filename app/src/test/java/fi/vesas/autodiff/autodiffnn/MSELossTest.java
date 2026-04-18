package fi.vesas.autodiff.autodiffnn;

import static org.junit.jupiter.api.Assertions.assertEquals;

import fi.vesas.autodiff.grad.GradNode;
import fi.vesas.autodiff.grad.Value;
import fi.vesas.autodiff.loss.MSELoss;

import org.junit.jupiter.api.Test;

public class MSELossTest {

    @Test
    public void testForwardMatchesHandComputedMSE() {
        Value yhat1 = new Value(0.5);
        Value yhat2 = new Value(0.8);
        double[] truth = { 1.0, 0.0 };

        GradNode[] nodes = { yhat1, yhat2 };

        MSELoss loss = new MSELoss(nodes);
        loss.setTruth(truth);

        // ((1.0 - 0.5)^2 + (0.0 - 0.8)^2) / 2 = (0.25 + 0.64) / 2 = 0.445
        assertEquals(0.445, loss.forward(), 1e-9);
    }

    @Test
    public void testZeroLossAtPerfectPrediction() {
        Value yhat1 = new Value(1.0);
        Value yhat2 = new Value(-2.0);
        double[] truth = { 1.0, -2.0 };

        GradNode[] nodes = { yhat1, yhat2 };

        MSELoss loss = new MSELoss(nodes);
        loss.setTruth(truth);

        assertEquals(0.0, loss.forward(), 1e-12);
    }
}
