package fi.vesas.autodiff.nn;

import fi.vesas.autodiff.grad.Value;

public class InputLayer {
    
    private Value [] vals;

    private InputLayer() {
    }

    public InputLayer(int size) {
        vals = new Value[size];
    }

    public float forward() {
        return 0;
    }
}
