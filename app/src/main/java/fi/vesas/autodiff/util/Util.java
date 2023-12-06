package fi.vesas.autodiff.util;

import java.util.Random;

public class Util {
    
    static Random rand = new Random(31);
    
    public static double rangeRand(double min, double max) {
        return min + (max - min) * rand.nextDouble();
    }

    public static double rangeGaussian(double range) {
        return range * rand.nextGaussian();
    }
}
