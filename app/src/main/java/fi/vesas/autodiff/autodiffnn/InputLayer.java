package fi.vesas.autodiff.autodiffnn;

import fi.vesas.autodiff.grad.GradNode;
import fi.vesas.autodiff.grad.Value;

public class InputLayer {
    
    private Value [] vals;

    public InputLayer(int size) {
        vals = new Value[size];

        for (int i = 0; i < size; i++) {
            vals[i] = new Value(0.0);
        }
    }

    public Value [] getOutputs() {
        return vals;
    }

    public void setInputs(double [] inputs) {
        for (int i = 0; i < inputs.length; i++) {
            vals[i].value = inputs[i];
        }
    }

    public float forward() {
        return 0;
    }
}
