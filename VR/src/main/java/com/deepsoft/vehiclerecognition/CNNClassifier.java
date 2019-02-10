package com.deepsoft.vehiclerecognition;


import android.content.Context;
import android.util.Log;
import android.util.Pair;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.io.IOException;
import java.util.Collections;
import java.util.Comparator;
import java.util.Vector;

/**
 * Vehicle Recognition Project
 * Associated with the paper "Transfer Learning Approach for Classification and Noise Reduction on Noisy Web data"
 * Created by Sina M.Baharlou (Sina.Baharlou@gmail.com) on 9/17/17.
 * Project page: https://www.sinabaharlou.com/VehicleRecognition
 */

class ResultsComparator implements  Comparator<Pair<String,Float>>
{
    @Override
    public int compare(Pair<String, Float> o1, Pair<String, Float> o2) {
        return o2.second.compareTo(o1.second);
    }
}

class CNNClassifier
{

    // class Variables
    private TensorFlowInferenceInterface mTensorFlow;
    final private Context mAppContext;

    // network parameters
    private String mNetworkName;
    private String mInputLayer;
    private String mOutputLayer;
    private int mInputSize;
    private int mInputDim;
    private double[] mMeanArray;
    private String mNetworkPath;
    private String mLabelsPath;
    private double mThreshold;

    // other parameters
    private Scalar mMeanScalar;
    private long mOutputSize;
    private int mLabelsCount;
    private Vector<String> mLabelsVector;

    private Mat mFloatMat;
    private Size mDstSize;
    private float[] mFloatArray;
    private Mat mCroppedMat;


    // final variables
    private final String JSON_TAG_NAME ="Name";
    private final String JSON_TAG_INPUT_LAYER ="InputLayer";
    private final String JSON_TAG_OUTPUT_LAYER ="OutputLayer";
    private final String JSON_TAG_INPUT_SIZE ="InputSize";
    private final String JSON_TAG_INPUT_DIM ="InputDim";
    private final String JSON_TAG_MEAN ="Mean";
    private final String JSON_TAG_NETWORK_PATH ="NetworkPath";
    private final String JSON_TAG_LABELS_PATH ="LabelsPath";
    private final String JSON_TAG_THRESHOLD = "Threshold";
    private final boolean ENABLE_DEBUG = true;

    public CNNClassifier(Context context, String config) throws JSONException, IOException
    {
        mAppContext = context;

        // load and parse json parameters
        String jsonString = Utils.loadJson(mAppContext, config);
        JSONObject jsonConfigs = new JSONObject(jsonString);


        // assign network parameters
        mNetworkName = jsonConfigs.getString(JSON_TAG_NAME);
        mInputLayer = jsonConfigs.getString(JSON_TAG_INPUT_LAYER);
        mOutputLayer = jsonConfigs.getString(JSON_TAG_OUTPUT_LAYER);
        mInputSize = jsonConfigs.getInt(JSON_TAG_INPUT_SIZE);
        mInputDim = jsonConfigs.getInt(JSON_TAG_INPUT_DIM);
        mNetworkPath = jsonConfigs.getString(JSON_TAG_NETWORK_PATH);
        mLabelsPath = jsonConfigs.getString(JSON_TAG_LABELS_PATH);
        mThreshold = jsonConfigs.getDouble(JSON_TAG_THRESHOLD);


        // get network mean json array
        JSONArray jsonArray = jsonConfigs.getJSONArray(JSON_TAG_MEAN);

        // read json array values
        mMeanArray = new double[jsonArray.length()];
        for (int i = 0; i < jsonArray.length(); i++)
            mMeanArray[i] = jsonArray.getDouble(i);

        // create mean scalar
        mMeanScalar = new Scalar(mMeanArray);

        // initialize tensorflow class
        mTensorFlow = new TensorFlowInferenceInterface(mAppContext.getAssets(),
                String.format("file:///android_asset/%s", mNetworkPath));

        // determine network's output size
        mOutputSize = mTensorFlow.graphOperation(mOutputLayer).output(0).shape().size(1);

        // load classifier labels
        mLabelsVector = Utils.loadLabels(mAppContext, mLabelsPath);
        mLabelsCount = mLabelsVector.size();

        // create CV.Size variable
        mDstSize = new Size(mInputSize, mInputSize);

        // initialize buffers
        mFloatArray = new float[mInputSize * mInputSize * mInputDim];
        mFloatMat = new Mat(mDstSize, CvType.CV_32FC3);

        if (ENABLE_DEBUG)
        {
            Log.d("CNN-Classifier", "Network name: " + mNetworkName);
            Log.d("CNN-Classifier", "Input Layer : " + mInputLayer);
            Log.d("CNN-Classifier", "Output Layer : " + mOutputLayer);
            Log.d("CNN-Classifier", "Input Size  : " + mInputSize);
            Log.d("CNN-Classifier", "Input dimension : " + mInputDim);
            Log.d("CNN-Classifier", "Network path : " + mNetworkPath);
            Log.d("CNN-Classifier", "Labels path : " + mLabelsPath);
            Log.d("CNN-Classifier", "Output size : " + mOutputSize);
            Log.d("CNN-Classifier", "Labels : " + mLabelsVector.toString());
        }


    }

    static Rect getROI(Mat inputMat)
    {
        // get input size
        Size matSize = inputMat.size();

        // find the smaller dimension
        double minDim = Math.min(matSize.width, matSize.height);
        // determine the new frame size
        minDim = Math.floor(minDim / 2.0);
        // get input center
        Point frameCenter = new Point(inputMat.width() / 2.0, inputMat.height() / 2.0);
        // create new frame crop rectangle

        return new Rect((int) (frameCenter.x - minDim),
                (int) (frameCenter.y - minDim),
                (int) (minDim * 2), (int) (minDim * 2));
    }

    private Mat preProcess(Mat inputMat)
    {
        // get crop rect
        Rect cropRect=getROI(inputMat);

        // crop input matrix
        Mat subMat = inputMat.submat(cropRect);
        // discard alpha channel
        Imgproc.cvtColor(subMat, subMat, Imgproc.COLOR_RGBA2RGB);
        // resize the input
        Imgproc.resize(subMat, subMat, mDstSize);
        // convert to float mat
        subMat.convertTo(mFloatMat, CvType.CV_32FC3);
        // subtract the mean
        Core.subtract(mFloatMat, mMeanScalar, mFloatMat);
        // get float array
        mFloatMat.get(0, 0, mFloatArray);

        if (ENABLE_DEBUG)
        {
            Log.d("CNN-Classifier", "Input size :" + inputMat.size().toString());
            Log.d("CNN-Classifier", "Rect size :" + cropRect.toString());
        }
        return subMat;
    }


    private Vector<Pair<String,Float>> getResults(float [] accuracy)
    {
        Vector<Pair<String,Float>> resultVector=new Vector<>();
        int index=0;
        for (String label : mLabelsVector)
            resultVector.add(new Pair<>(label, accuracy[index++]));
        return resultVector;

    }


    Vector<Pair<String,Float>> performRecognition(Mat cameraMat)
    {

        // convert color,crop, resize, and subtract the mean
        mCroppedMat =preProcess(cameraMat);

        // Assign network's input data
        mTensorFlow.feed(mInputLayer, mFloatArray,
                1, mInputSize,
                mInputSize,
                mInputDim);

        // Feeding forward the network
        mTensorFlow.run(new String[]{mOutputLayer});

        float[] rawOutput = new float[(int) mOutputSize];
        float[] acctualOutput = new float[mLabelsCount];

        // fetch output values
        mTensorFlow.fetch(mOutputLayer, rawOutput);

        // get acctual output
        System.arraycopy(rawOutput, 0, acctualOutput, 0, mLabelsCount);

        // determine the accuracy
        Vector<Pair<String,Float>> results = getResults(acctualOutput);
        Collections.sort(results,new ResultsComparator());

        if (ENABLE_DEBUG)
        {
            for (Pair<String, Float> pair : results) {
                String output = String.format("%s:\t\t%.3f", pair.first, pair.second);
                Log.d("CNN-Classifier", output);
            }
        }
        return results;


    }

    double getThreshold()
    {
        return mThreshold;
    }

    Mat getCroppedMat()
    {
        return mCroppedMat;
    }
}
