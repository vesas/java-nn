package fi.vesas.autodiff.util;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Stats {
    

    public static Map<String, List<List<Double>>> neuronWeights = new HashMap<>();
    public static Map<String, List<Double>> recordNeuronBias = new HashMap<>();

    public static void recordNeuronWeights(String label, double value) {

        var vals = neuronWeights.get(label);

        if(vals == null) {
            vals = new ArrayList<List<Double>>();
            neuronWeights.put(label, vals);
        }


    }


    public static void recordNeuronBias(String label, double value) {

    }

}
