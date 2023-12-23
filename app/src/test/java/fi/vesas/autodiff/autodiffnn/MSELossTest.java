package fi.vesas.autodiff.autodiffnn;

import org.junit.jupiter.api.Test;

import fi.vesas.autodiff.grad.GradNode;
import fi.vesas.autodiff.grad.Value;
import fi.vesas.autodiff.loss.MSELoss;

public class MSELossTest {
    
    @Test
    public void test1() {

        double [] preds = {0.0, 0.0, 0.0};
        double [] ys = {1.0, 1.0};

        Value v1 = new Value(1.0);
        Value v2 = new Value(1.0);

        GradNode [] nodes =  {v1, v2};

        MSELoss e = new MSELoss(nodes);
        e.setTruth(ys);

        double res = e.forward();

        System.out.println("res: " + res);
    }
}
