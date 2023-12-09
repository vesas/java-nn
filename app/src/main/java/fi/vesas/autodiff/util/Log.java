package fi.vesas.autodiff.util;

import java.io.FileWriter;
import java.io.IOException;

/*
 * Logs values to csv file
 */
public class Log {
    
    private static FileWriter writer = null;
    
    static {
        try {
            writer = new FileWriter("log.csv");
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
    }

    public static void logHeader(String ... cols) {

        String header = String.join(",", cols);
        try {
            writer.write(header);
            writer.write("\n");
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
    }

    public static void log(String ... values) {
        String header = String.join(",", values);
        try {
            writer.write(header);
            writer.write("\n");
            writer.flush();
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
    }
}
