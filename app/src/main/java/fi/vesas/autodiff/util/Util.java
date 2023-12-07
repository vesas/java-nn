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

    /*
    * [a,b] 
    * (b -a) * ((x - minx) / (maxx - minx)) + a
     */
    public static double [][] normalize(double [][] vals, double a, double b) {

        double min = Double.MAX_VALUE;
        double max = Double.MIN_VALUE;

        for (int i = 0; i < vals.length; i++) {
            for (int j = 0; j < vals[i].length; j++) {
                if (vals[i][j] < min) {
                    min = vals[i][j];
                }
                if(vals[i][j] > max) {
                    max = vals[i][j];
                }
            }
        }

        double [][] normalized = new double[vals.length][];
        for (int i = 0; i < vals.length; i++) {
            normalized[i] = new double[vals[i].length];
            for (int j = 0; j < vals[i].length; j++) {
                normalized[i][j] = (b - a) * ((vals[i][j] - min) / (max - min)) + a;
            }
        }
        return normalized;
    }

     /*
    * [a,b] 
    * (b -a) * ((x - minx) / (maxx - minx)) + a
     */
    public static double [] normalize(double [] vals, double a, double b) {

        double min = Double.MAX_VALUE;
        double max = Double.MIN_VALUE;

        for (int i = 0; i < vals.length; i++) {
            if (vals[i] < min) {
                min = vals[i];
            }
            if(vals[i] > max) {
                max = vals[i];
            }
        }

        double [] normalized = new double[vals.length];
        for (int i = 0; i < vals.length; i++) {
            normalized[i] = 
                (b - a) * ((vals[i] - min) / (max - min)) + a;
        }
        return normalized;
    }
}
