/**
 * Vehicle Recognition Project
 * Associated with the paper "Transfer Learning Approach for Classification and Noise Reduction on Noisy Web data"
 * Created by Sina M.Baharlou (Sina.Baharlou@gmail.com) on 9/17/17.
 * Project page: https://www.sinabaharlou.com/VehicleRecognition
 */

// -- Package declaration -- 
package com.deepsoft.vehiclerecognition;

// -- Import required libraries --
import android.app.Activity;
import android.content.Context;
import android.content.DialogInterface;
import android.hardware.Camera;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.os.Looper;
import android.speech.tts.TextToSpeech;
import android.support.v7.app.AlertDialog;
import android.text.method.LinkMovementMethod;
import android.util.Pair;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.CompoundButton;
import android.widget.TextView;
import android.widget.Toast;
import android.widget.ToggleButton;
import org.json.JSONException;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;
import java.io.File;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;
import java.util.Vector;

// -- The main class --
public class VehicleRecognition extends Activity implements CameraBridgeViewBase.CvCameraViewListener2 {
    
    // -- Class variables--
    private Context mContext;
    private JavaCameraView mOpenCvCameraView;
    private CNNClassifier mClassifier;
    private TextToSpeech mToSpeech;
    private Mat mCameraMat;

    // -- UI elements --
    private Button mDetectBtn;
    private ToggleButton mVoiceBtn;
    private Button mFlashBtn;
    private Button mSelectBtn;
    private Button mInfoBtn;
    private Button mSaveBtn;

    // -- Flags --
    private boolean mObjDetect = false;
    private boolean mIsBusy = false;

    // -- Settings flags --
    private boolean mEnableVoice = true;
    private boolean mEnableFlash = false;
    private boolean mEnableSave = false;

    // -- Storing configuration --
    private final String APP_FOLDER = "/VR";
    private boolean mDirExists = false;
    private File mSaveFolder;

    // -- ROI frame properties --
    private Scalar mFrameColor = new Scalar(255, 255, 255);
    private double mFrameLength = 10;
    private int mFrameMargin = 10;
    private int mFrameWidth = 2;

    // -- Networks' titles
    String mNetworkNames[] = new String[]
            {"SqueezeNet FT, 9 classes (90% top-1)",
                    "SqueezeNet FT, 15 classes (80% top-1)",
                    "CarNet+SVM 9 classes reduced noise (96% top-1)",
                    "SqueezeNet+SVM 9 classes (90.53% top-1)"};
    
    // -- Networks' definition files
    String mNetworkFiles[] = new String[]
            {"tensorflow_squeezenet.json",
                    "tensorflow_squeezenet_beta.json",
                    "tensorflow_carnet_svm.json",
                    "tensorflow_squeezenet_svm.json"};
    
    // -- Constructor --
    public VehicleRecognition() {
    }

    // -- On UI create -- 
    @Override
    public void onCreate(Bundle savedInstanceState) {

        super.onCreate(savedInstanceState);
        try {
            // -- Keep the current context -- 
            mContext = this;
            
            // -- Set screen flags -- 
            getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
            setContentView(R.layout.activity_vehicle_recognition);

            // -- Get java camera surface --
            mOpenCvCameraView = (JavaCameraView) findViewById(R.id.java_surface_view);
            mDetectBtn = (Button) findViewById(R.id.detectBtn);
            mVoiceBtn = (ToggleButton) findViewById(R.id.voiceBtn);
            mFlashBtn = (Button) findViewById(R.id.flashBtn);
            mSaveBtn = (Button) findViewById(R.id.saveBtn);
            mSelectBtn = (Button) findViewById(R.id.selectBtn);
            mInfoBtn = (Button) findViewById(R.id.infoBtn);

            // -- Set camera view listener --
            mOpenCvCameraView.setCvCameraViewListener(this);

            // --init OpenCV --
            initOpenCV();

            // Create TensorFlow classifier --
            initClassifier(0);

            // -- Init text2speech --
            initSpeech();

            // -- Get app folder --
            mSaveFolder = new File(Environment.getExternalStorageDirectory() + APP_FOLDER);
            mDirExists = mSaveFolder.exists() || mSaveFolder.mkdir();

            // -- Set on save button click handler --
            mSaveBtn.setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View v) {
                    mEnableSave = !mEnableSave;
                    if (mEnableSave) {
                        String saveStr = String.format(getString(R.string.storage_on), mSaveFolder.toString());
                        Toast.makeText(mContext, saveStr, Toast.LENGTH_SHORT).show();

                    }
                }
            });

            // -- Set network-select button listener --
            mSelectBtn.setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View v) {
                    AlertDialog.Builder builder = new AlertDialog.Builder(mContext);
                    builder.setTitle(R.string.select_network);
                    builder.setIcon(R.drawable.select);
                    builder.setItems(mNetworkNames, new DialogInterface.OnClickListener() {
                        @Override
                        public void onClick(DialogInterface dialog, int which) {
                            initClassifier(which);
                        }
                    });
                    builder.show();
                }
            });

            // -- Set network-select button listener --
            mInfoBtn.setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View v) {
                    AlertDialog.Builder builder = new AlertDialog.Builder(mContext);
                    builder.setTitle(R.string.information_title);
                    builder.setIcon(R.mipmap.ic_launcher);
                    builder.setMessage(R.string.information);
                    AlertDialog infoDialog = builder.create();
                    infoDialog.show();
                    ((TextView) infoDialog.findViewById(android.R.id.message)).setMovementMethod(LinkMovementMethod.getInstance());
                }
            });

            // -- Set detect button listener --
            mDetectBtn.setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View v) {
                    mObjDetect = true;
                    mDetectBtn.setEnabled(false);
                }
            });

            // -- Set voice button listener --
            mVoiceBtn.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
                @Override
                public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
                    if (isChecked && mToSpeech != null)
                        mToSpeech.speak(getString(R.string.voice_on), TextToSpeech.QUEUE_FLUSH, null);
                    mEnableVoice = isChecked;
                }
            });

            // -- Set flash button listener
            mFlashBtn.setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View v) {
                    mEnableFlash = !mEnableFlash;
                    enableTorch(mEnableFlash);
                }
            });
        } catch (Exception e) {
            Toast.makeText(mContext, e.getMessage(), Toast.LENGTH_SHORT).show();
            finish();
        }
    }

    // -- Init the classifier -- 
    private void initClassifier(int netIndex) {
        try {
            // -- Instantiate the classifier -- 
            mClassifier = new CNNClassifier(mContext, mNetworkFiles[netIndex]);
            Toast.makeText(mContext, mNetworkNames[netIndex], Toast.LENGTH_SHORT).show();
        } catch (JSONException e) {
            Toast.makeText(mContext, R.string.json_error, Toast.LENGTH_SHORT).show();
            finish();
        } catch (IOException e) {
            Toast.makeText(mContext, R.string.file_missing, Toast.LENGTH_SHORT).show();
            finish();
        } catch (Exception e) {
            Toast.makeText(mContext, R.string.tensor_error, Toast.LENGTH_SHORT).show();
            finish();
        }
    }

    // -- Enable torch light --
    private void enableTorch(boolean enabled) {
        // -- Retreive camera parameters --
        Camera.Parameters params = mOpenCvCameraView.mCamera.getParameters();
        // -- Determine flash mode --
        String flashMode = (enabled)
                ? Camera.Parameters.FLASH_MODE_TORCH :
                Camera.Parameters.FLASH_MODE_OFF;
        // -- Set flash mode --
        params.setFlashMode(flashMode);
        // -- Set camera parameters --
        mOpenCvCameraView.mCamera.setParameters(params);
    }

    // -- Init OpenCV -- 
    private void initOpenCV() {
        if (OpenCVLoader.initDebug()) {
            mCameraMat = new Mat();
            mOpenCvCameraView.enableView();
        } else {
            Toast.makeText(mContext, R.string.opencv_failed, Toast.LENGTH_SHORT).show();
            finish();
        }
    }
        
    // -- Init Text-to-Speech class
    private void initSpeech() {
        try {
            // -- Instansiate the classs --
            mToSpeech = new TextToSpeech(this, new TextToSpeech.OnInitListener() {
                @Override
                public void onInit(int status) {
                    // -- Set language to UK if initialized correctly--
                    if (status != TextToSpeech.ERROR)
                        mToSpeech.setLanguage(Locale.UK);
                }
            });
        } catch (Exception e) {
            Toast.makeText(mContext, R.string.speech_failed, Toast.LENGTH_SHORT).show();
            mToSpeech = null;
        }
    }

    // -- On UI pause -- 
    @Override
    public void onPause() {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    // -- On UI resume -- 
    @Override
    public void onResume() {
        super.onResume();
        initOpenCV();
    }

    // -- On UI destroy -- 
    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    // -- Recognition takes place here -- 
    public void performRecognition(Mat cameraMat) {

        try {
            //  -- Perform the prediction --
            long startTime = System.currentTimeMillis();
            Vector<Pair<String, Float>> predict = mClassifier.performRecognition(cameraMat);
            long endTime = System.currentTimeMillis();
            long elapsedTime = endTime - startTime;
            final String toastString;
            double threshValue = mClassifier.getThreshold();

            // -- Get the top prediction --
            Pair<String, Float> topOne = predict.elementAt(0);

            // -- Notify if the accuracy is greater than threshold --
            if (topOne.second > threshValue) {
                
                // -- If prediction accuracy is available --
                if (threshValue>0.0)
                    toastString = String.format(getString(R.string.output_format),
                        topOne.first,
                        topOne.second * 100.0,
                        (float) elapsedTime / 1000.0);
                
                // -- Otherwise --
                else {
                    // -- Sample belongs to one SVM class --
                    if (topOne.second>0.0)
                        toastString = String.format(getString(R.string.output_format_svm_h),
                                topOne.first,
                                (float) elapsedTime / 1000.0);
                    // -- The sample doesn't belong to any class but it is close to one --
                    else
                        toastString = String.format(getString(R.string.output_format_svm_l),
                                topOne.first,
                                (float) elapsedTime / 1000.0);
                }
                // -- Say the detected label --
                if (mEnableVoice && mToSpeech != null)
                    mToSpeech.speak(topOne.first, TextToSpeech.QUEUE_FLUSH, null);

                // -- Save the image --
                if (mEnableSave && mDirExists)
                    saveFrame(mClassifier.getCroppedMat(), topOne);

            } else
                toastString = getString(R.string.nothing_found);


            // -- Show toast notification and enable the button --
            new Handler(Looper.getMainLooper()).post(new Runnable() {
                public void run() {
                    Toast.makeText(mContext, toastString, Toast.LENGTH_SHORT).show();
                    mDetectBtn.setEnabled(true);
                }
            });

            // -- Set the boolean flags --
            mIsBusy = false;
            mObjDetect = false;
            cameraMat.release();
        } catch (Exception e) {
            new Handler(Looper.getMainLooper()).post(new Runnable() {
                public void run() {
                    Toast.makeText(mContext, R.string.network_error, Toast.LENGTH_SHORT).show();
                }
            });

            // -- Set the boolean flags --
            mIsBusy = false;
            mObjDetect = false;
        }

    }

    // -- Save the frame -- 
    private boolean saveFrame(Mat inputMat, Pair<String, Float> topOne) {

        // -- Get current date string --
        String currentDateAndTime = new SimpleDateFormat(getString(R.string.date_format)).format(new Date());

        // -- Format output filename --
        String outFile = String.format(getString(R.string.save_format),
                mSaveFolder.toString(),
                currentDateAndTime,
                topOne.first,
                topOne.second);

        // -- Create temporary mat --
        Mat saveMat = new Mat();
        
        // -- Swap R and B --
        Imgproc.cvtColor(inputMat, saveMat, Imgproc.COLOR_RGBA2BGR);
        
        // -- Save the results 
        return Highgui.imwrite(outFile, saveMat);
    }

    // -- Draw the frame -- 
    private Mat drawFrame(Mat inputMat) {
        
        // -- Get region of interest --
        Rect roiRect = CNNClassifier.getROI(inputMat);

        // -- Set frame margins --
        roiRect.y = roiRect.y + mFrameMargin;
        roiRect.height = roiRect.height - mFrameMargin * 2;

        //  -- Draw lines (frame corners) --
        // -- Top left --
        Core.line(inputMat, new Point(roiRect.x, roiRect.y),
                new Point(roiRect.x + mFrameLength, roiRect.y), mFrameColor, mFrameWidth);

        Core.line(inputMat, new Point(roiRect.x, roiRect.y),
                new Point(roiRect.x, roiRect.y + mFrameLength), mFrameColor, mFrameWidth);

        // -- Top right --
        Core.line(inputMat, new Point(roiRect.x + roiRect.width, roiRect.y),
                new Point(roiRect.x + roiRect.width - mFrameLength, roiRect.y), mFrameColor, mFrameWidth);

        Core.line(inputMat, new Point(roiRect.x + roiRect.width, roiRect.y),
                new Point(roiRect.x + roiRect.width, roiRect.y + mFrameLength), mFrameColor, mFrameWidth);

        // -- Bottom left --
        Core.line(inputMat, new Point(roiRect.x, roiRect.y + roiRect.height),
                new Point(roiRect.x, roiRect.y + roiRect.height - mFrameLength), mFrameColor, mFrameWidth);

        Core.line(inputMat, new Point(roiRect.x, roiRect.y + roiRect.height),
                new Point(roiRect.x + mFrameLength, roiRect.y + roiRect.height), mFrameColor, mFrameWidth);

        // -- Bottom right --
        Core.line(inputMat, new Point(roiRect.x + roiRect.width, roiRect.y + roiRect.height),
                new Point(roiRect.x + roiRect.width - mFrameLength, roiRect.y + roiRect.height), mFrameColor, mFrameWidth);

        Core.line(inputMat, new Point(roiRect.x + roiRect.width, roiRect.y + roiRect.height),
                new Point(roiRect.x + roiRect.width, roiRect.y + roiRect.height - mFrameLength), mFrameColor, mFrameWidth);

        return inputMat;
    }


    // -- On receiving camrea frame --
    public Mat onCameraFrame(final CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        
        try {
            // -- Get input frame matrix --
            Mat canvas = inputFrame.rgba();

            // -- Classify object --
            if (mObjDetect && !mIsBusy) {
                
                // -- Copy input mat for further processing --
                canvas.copyTo(mCameraMat);

                // -- Set busy flag --
                mIsBusy = true;

                // -- Create recognition runnable --
                Runnable recRunnable = new Runnable() {
                    @Override
                    public void run() {
                        synchronized (mCameraMat) {
                            performRecognition(mCameraMat);
                        }
                    }
                };

                // -- Start recognition thread --
                Thread recThread = new Thread(recRunnable);
                recThread.start();
            }

            // -- Draw frame corners --
            return drawFrame(canvas);
        } catch (Exception e) {          
            // -- Show error toast --
            new Handler(Looper.getMainLooper()).post(new Runnable() {
                public void run() {
                    Toast.makeText(mContext, R.string.frame_error, Toast.LENGTH_SHORT).show();
                }
            });
            finish();
            return null;
        }


    }
    
    public void onCameraViewStarted(int width, int height) {
    }
    
    public void onCameraViewStopped() {
    }
}
