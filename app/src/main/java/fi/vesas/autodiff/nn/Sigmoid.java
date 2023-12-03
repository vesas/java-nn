package fi.vesas.autodiff.nn;

public class Sigmoid {

    public float value(float value) {
        return 1.0f / (1.0f + (float)Math.exp(-value));
    }
}
