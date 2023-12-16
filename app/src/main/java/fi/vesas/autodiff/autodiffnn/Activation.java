package fi.vesas.autodiff.autodiffnn;

import fi.vesas.autodiff.grad.GradNode;

public interface Activation {
    
    public void setInput(GradNode x);
}
