package fi.vesas.autodiff.autodiffnn;

import java.util.ArrayList;
import java.util.List;

import fi.vesas.autodiff.autodiffnn.WeightInitializers.WeightInitializerInterface;
import fi.vesas.autodiff.grad.GradNode;
import fi.vesas.autodiff.loss.LossInterface;

public class ModelBuilder {
    
    private GradNode [] lastOutputs;
    private InputLayer inputLayer;
    private List<DenseLayer> layers = new ArrayList<>();
    private LossInterface loss;

    private WeightInitializerInterface weightInitializer;

    public ModelBuilder setWeightInitializer(WeightInitializerInterface weightInitializer) {
        this.weightInitializer = weightInitializer;
        return this;
    }

    public ModelBuilder add(InputLayer inputLayer) {
        this.inputLayer = inputLayer;
        
        this.lastOutputs = inputLayer.getOutputs();
        return this;
    }
    
    public ModelBuilder add(DenseLayer layer) {
        layers.add(layer);
        layer.setLabel("l" + layers.size());
        layer.initialize(lastOutputs);
        this.lastOutputs = layer.getOutputs();
        return this;
    }

    public ModelBuilder add(Activation activation) {
        
        DenseLayer lastLayer = layers.get(layers.size() - 1);
        lastLayer.setActivation(activation);
        this.lastOutputs = lastLayer.getOutputs();
        return this;
    }

    public ModelBuilder add(LossInterface loss) {
        
		loss.setInputs(this.lastOutputs);
        this.loss = loss;
        return this;
    }

    public Model build() {
        MLP mlp = new MLP();
        mlp.inputLayer = inputLayer;
        
        mlp.denseLayers = new DenseLayer[layers.size()];
        for(int i = 0; i < layers.size(); i++) {
            DenseLayer layer = layers.get(i);
            if(weightInitializer != null) {
                layer.setWeightInitializer(weightInitializer);
            }
            mlp.denseLayers[i] = layer;
        }

        mlp.loss = loss;
        return mlp;
    }
}
