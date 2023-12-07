package fi.vesas.autodiff.grad;

public class Util {
   
    static final double h = 0.0000001f;
    
    public static double diff(GradNode node, Value val) {
        double val1 = node.forward();

        double originalValue = val.value;
        // tweak value a bit
        val.value += h;

        // do forward pass again
        double val2 = node.forward();

        // how much did the value change
        double diff = (val2 - val1) / h;

        val.value = originalValue;
        return diff;
    }


    
}
