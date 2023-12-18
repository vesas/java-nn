package fi.vesas.autodiff.grad;

public final class Util {
   
    private Util() {}
    
    private static final double h = 0.0000001f;
    
    // Estimate the gradient using finite differences
    // Can be used for testing
    public static double diff(GradNode node, Value val) {
        double val1 = node.forward();

        double originalValue = val.value;
        // tweak value a bit
        val.value += h;

        // do forward pass again
        double val2 = node.forward();

        // how much did the value change
        double diff = (val2 - val1) / h;

        // restore original value
        val.value = originalValue;
        return diff;
    }


    
}
