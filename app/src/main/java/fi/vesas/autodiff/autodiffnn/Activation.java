package fi.vesas.autodiff.autodiffnn;

import fi.vesas.autodiff.grad.GradNode;

public interface Activation {
    
    void setInput(GradNode x);

    Activation createInstance();
}
