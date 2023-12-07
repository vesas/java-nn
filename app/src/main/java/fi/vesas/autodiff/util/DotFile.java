package fi.vesas.autodiff.util;

import java.io.FileWriter;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import fi.vesas.autodiff.autodiffnn.Tanh;
import fi.vesas.autodiff.grad.GradNode;
import fi.vesas.autodiff.grad.Mul;
import fi.vesas.autodiff.grad.Value;

public class DotFile {


    static Map<String, String> nodeValues = new HashMap<>();

    public static void gen(GradNode node, String fileName) {

        try {
            FileWriter fileWriter = new FileWriter(fileName, false);

            fileWriter.write("digraph G {\n");
            fileWriter.write("rankdir=RL;\n");
            generateNodeValues(node, fileWriter);
            writeGradNode(node, fileWriter);
            fileWriter.write("\n}\n");

            fileWriter.flush();
            fileWriter.close();
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }

    }

    public static void generateNodeValues(GradNode node, FileWriter fileWriter) throws IOException {

        String str = node.toDotString();
        String strLabel = node.toString();

        fileWriter.write(str );
        fileWriter.write(" [label=\"");
        fileWriter.write(strLabel);
        fileWriter.write("\"];");
        fileWriter.write("\n");

        GradNode [] children = node.getChildren();
        for (int i = 0; i < children.length; i++) {
            generateNodeValues(children[i], fileWriter);
        }
    }

    public static void writeGradNode(GradNode node, FileWriter fileWriter) throws IOException {

        String str = node.toDotString();

        GradNode [] children = node.getChildren();
        for (int i = 0; i < children.length; i++) {

            fileWriter.write(str );
            fileWriter.write(" -> ");
            fileWriter.write(children[i].toDotString());
            fileWriter.write(";\n");
        }

        for (int i = 0; i < children.length; i++) {
            writeGradNode(children[i], fileWriter);
        }

    }


    public static void main(String[] args) {
        
        Value a = new Value(0.5f, "input1");
        Value b = new Value(-2.0f, "input2");
        Mul c = new Mul(a, b);
        Tanh t = new Tanh(c);

        t.backward();

        
        String fileName = "test.dot";
        gen(t, fileName);
    }
}
