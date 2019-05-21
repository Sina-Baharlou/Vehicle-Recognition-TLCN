/**
 * Vehicle Recognition Project
 * Associated with the paper "Transfer Learning Approach for Classification and Noise Reduction on Noisy Web data"
 * Created by Sina M.Baharlou (Sina.Baharlou@gmail.com) on 9/17/17.
 * Project page: https://www.sinabaharlou.com/VehicleRecognition
 */

// -- Package declaration -- 
package com.deepsoft.vehiclerecognition;

// -- Import required libraries --
import android.content.Context;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.Vector;

// -- Utils Class -- 
class Utils
{
    // -- Load json file --
    static String loadJson(Context context, String filename) throws IOException
    {
        // -- Get json file input stream --
        InputStream inputStream = context.getAssets().open(filename);

        // -- Get file size and allocate the buffer --
        int fileSize = inputStream.available();
        byte[] inputBuffer = new byte[fileSize];

        // -- Read the file and close it --
        inputStream.read(inputBuffer);
        inputStream.close();

        return new String(inputBuffer, "UTF-8");
    }

    // -- Load labels file --
    static Vector<String> loadLabels(Context context, String filename) throws IOException
    {
        // -- Get the file input stream --
        InputStream inputStream = context.getAssets().open(filename);
        BufferedReader reader=new BufferedReader(new InputStreamReader(inputStream));
        Vector<String> labels= new Vector<>();

        // -- Fetch labels line by line --
        String inputLine;
        while((inputLine=reader.readLine())!=null)
            labels.add(inputLine);

        // -- Close the file --
        reader.close();
        inputStream.close();

        return labels;
    }
}
