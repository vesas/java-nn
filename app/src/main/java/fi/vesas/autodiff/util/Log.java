package fi.vesas.autodiff.util;

/**
 * Logs to memory for charting
 */
public class Log {
    
    private double [] values1 = new double[6000];
    private int index1 = 0;

    private double [] values2 = new double[6000];
    private int index2 = 0;

    public void init() {
        index1 = 0;
        index2 = 0;
    }

    // logs to series 1
    public final void log1(double value) {
        values1[index1++] = value;
    }

    // logs to series 2
    public final void log2(double value) {
        values2[index2++] = value;
    }

    // gets series 1 data
    public final double [] getValues1() {
        double ret [] = new double[index1];
        System.arraycopy(values1, 0, ret, 0, index1);
        return ret;
    }

    // gets series 2 data
    public final double [] getValues2() {
        double ret [] = new double[index2];
        System.arraycopy(values2, 0, ret, 0, index2);
        return ret;
    }

    public final double [] getIndexes1() {
        double ret [] = new double[index1];
        for(int i = 0; i < index1; i++) {
            ret[i] = i;
        }
        return ret;
    }

    public final double [] getIndexes2() {
        double ret [] = new double[index2];
        for(int i = 0; i < index2; i++) {
            ret[i] = i;
        }
        return ret;
    }

}
