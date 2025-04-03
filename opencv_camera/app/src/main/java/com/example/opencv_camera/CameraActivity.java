package com.example.opencv_camera;


import android.Manifest;
import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.util.Log;
import android.view.SurfaceView;
import android.view.Window;
import android.view.WindowManager;

import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import org.json.JSONArray;
import org.json.JSONObject;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class CameraActivity extends Activity implements CameraBridgeViewBase.CvCameraViewListener2 {
    private static final String TAG = "CameraActivity";

    List<Mat> facesData = Collections.synchronizedList(new ArrayList<>());
    private final List<byte[]> facesBytesData = new ArrayList<>();
    private volatile boolean sendingInProgress = false;
    private boolean pauseFrameCapture = false;
    private Mat mRgba;
    private Mat mGray;
    private Mat transpose_gray,transpose_rgb;


    private CameraBridgeViewBase mOpenCvCameraView;

    private CascadeClassifier cascadeClassifier;

    private String username;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

         username = getIntent().getStringExtra("USERNAME");
        Log.d(TAG, "Username received: " + username);


        // Request Camera Permission
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_DENIED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, 0);
        }

        setContentView(R.layout.activity_camera);

        mOpenCvCameraView = findViewById(R.id.frame_Surface);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCameraPermissionGranted();
        mOpenCvCameraView.setCameraIndex(CameraBridgeViewBase.CAMERA_ID_FRONT);
        mOpenCvCameraView.setCvCameraViewListener(this);



        try {
            InputStream is = getResources().openRawResource(R.raw.haarcascade_frontalface_alt);
            File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
            File mCascadeFile = new File(cascadeDir,"haarcascade_frontalface_alt.xml");
            FileOutputStream os = new FileOutputStream(mCascadeFile);
            byte[] buffer = new byte[4096];
            int byteRead;
            while((byteRead =is.read(buffer))!=-1){
                os.write(buffer,0,byteRead);
            }
            is.close();
            os.close();

            cascadeClassifier = new CascadeClassifier(mCascadeFile.getAbsolutePath());

            if (cascadeClassifier.empty()) {
                Log.e(TAG, "Failed to load cascade classifier");
                cascadeClassifier = null;
            } else {
                Log.d(TAG, "Cascade classifier loaded successfully");
            }
        }
        catch(IOException e){
            Log.i(TAG,"Cascade file not found");
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (OpenCVLoader.initDebug()) {
            Log.d(TAG, "OpenCV initialized successfully");
            mOpenCvCameraView.enableView();
            mOpenCvCameraView.setScaleX(1.0f);
            mOpenCvCameraView.setScaleY(1.3f);
        } else {
            Log.e(TAG, "OpenCV initialization failed");
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (mOpenCvCameraView != null) {
            mOpenCvCameraView.disableView();
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null) {
            mOpenCvCameraView.disableView();
        }
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        mRgba = new Mat(height, width, CvType.CV_8UC4);
        mGray = new Mat(height, width, CvType.CV_8UC1);
       // rects = new MatOfRect();

        mOpenCvCameraView.setMaxFrameSize(1280, 720);
    }

    @Override
    public void onCameraViewStopped() {
        mRgba.release();
        mGray.release();

    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {

        mRgba = inputFrame.rgba();
        mGray=inputFrame.gray();

        if (pauseFrameCapture) {
            Imgproc.putText(mRgba, "Uploading...", new Point(50, 50),
                    Imgproc.FONT_HERSHEY_SIMPLEX, 1.0, new Scalar(0, 255, 0), 1);
            return mRgba;
        }

        MatOfRect rects = new MatOfRect();
        cascadeClassifier.detectMultiScale(mGray,rects,1.1,2);
        int flag = 0,j=1;
        for (Rect rect : rects.toList()){

            Mat submat = mRgba.submat(rect);
            Imgproc.rectangle(mRgba,rect,new Scalar(0,255,0),10);


            Mat croppedFace = new Mat(mGray, rect);

            
            Mat resizedFace = new Mat();
            Size size = new Size(50, 50);
            Imgproc.resize(croppedFace, resizedFace, size);

            
            byte[] face1DArray = matToGrayscaleByteArray(resizedFace);

            synchronized (facesData) {
                if (facesData.size() < 100) {
                    facesData.add(resizedFace.clone());  
                    facesBytesData.add(face1DArray);     
                    
                    Log.d(TAG, "Hold on Wait - Stored face: " + facesData.size());
                }

                if (facesData.size() == 100 && flag == 0 && !sendingInProgress) {
                    flag = 1;
                    Log.d(TAG, "Collected 100 frames, sending to server..." + j);
                    j++;
                    sendingInProgress = true;
                    pauseFrameCapture = true;
                    sendFramesToServer(username);
                }
            }

        }

       if(flag != 1) {
           
           Imgproc.putText(mRgba, "Total Frame: " + facesData.size(), new Point(50, 50),
                   Imgproc.FONT_HERSHEY_SIMPLEX, 1.0, new Scalar(255, 0, 0), 1);
       }else {
           Imgproc.putText(mRgba, "Uploading...", new Point(50, 50),
                   Imgproc.FONT_HERSHEY_SIMPLEX, 1.0, new Scalar(0, 255, 0), 1);
       }

        return mRgba;

    }


    private void sendFramesToServer(String username) {
        Log.d(TAG, "sendFramesToServer called for username: " + username);

        new Thread(() -> {
            try {
                Log.d(TAG, "Preparing to send frames for username: " + username);
                URL url = new URL("http://192.168.29.217:5000/send-file/"+username);
                Log.d(TAG, "Server URL: " + url.toString());
                HttpURLConnection connection = (HttpURLConnection) url.openConnection();
                connection.setRequestMethod("POST");
                connection.setDoOutput(true);
                connection.setRequestProperty("Content-Type", "application/json");

                List<byte[]> facesBytesCopy;
                synchronized (facesBytesData) {
                    facesBytesCopy = new ArrayList<>(facesBytesData);
                    facesBytesData.clear();
                }

                
                JSONArray framesArray = new JSONArray();
                for (byte[] faceBytes : facesBytesCopy) {
                    JSONArray frameArray = new JSONArray();
                    for (byte b : faceBytes) {
                        frameArray.put(b & 0xFF);  // Convert byte to unsigned int (0-255)
                    }
                    framesArray.put(frameArray);
                }


                JSONObject requestBody = new JSONObject();
                requestBody.put("faces_data", framesArray);

                Log.d(TAG, "Request Body (faces_data) prepared for username: " + username+"Frame Array"+framesArray.length());

                OutputStream os = connection.getOutputStream();
                os.write(requestBody.toString().getBytes());
                os.flush();
                os.close();

                int responseCode = connection.getResponseCode();
                Log.d(TAG, "Response Code for username " + username + ": " + responseCode);
                Log.d(TAG, "Response Code: " + responseCode);

                if (responseCode == HttpURLConnection.HTTP_OK) {
                    Log.d(TAG, "Frames uploaded successfully.");

                    facesData.clear();  

                    saveNameToLocalStorage(username);

                    
                    runOnUiThread(() -> {
                        Log.d(TAG, "Redirecting to MainActivity for username: " + username);
                        redirectToMainActivity();

                    });

                } else {
                    Log.e(TAG, "Failed to upload frames.");
                }

            } catch (Exception e) {
                Log.e(TAG, "Error sending frames: " + e.getMessage(), e);
            }
        }).start();
    }

    private double[] extractFeatureVector(Mat faceMat) {
        Log.d(TAG, "Extracting feature vector from frame - Original Size: "
                + faceMat.rows() + "x" + faceMat.cols() + ", Type: " + faceMat.type());

        Mat gray = new Mat();
        Imgproc.cvtColor(faceMat, gray, Imgproc.COLOR_RGBA2GRAY);

        Log.d(TAG, "Converted to grayscale - Size: "
                + gray.rows() + "x" + gray.cols() + ", Type: " + gray.type());

        gray = gray.reshape(1, 1);  
        Log.d(TAG, "Reshaped to single row - New Size: "
                + gray.rows() + "x" + gray.cols());

        if (gray.empty()) {
            Log.e(TAG, "Gray Mat is unexpectedly empty after processing.");
            return new double[0];
        }

        
        Mat floatGray = new Mat();
        gray.convertTo(floatGray, CvType.CV_32F);

        double[] faceVector = new double[(int) floatGray.total()];
        Log.d(TAG, "Total elements expected: " + floatGray.total());

        try {
            floatGray.get(0, 0, faceVector);

            if (faceVector == null || faceVector.length == 0) {
                Log.e(TAG, "faceVector is empty or null after floatGray.get()");
            } else {
                Log.d(TAG, "Extracted feature vector length (post-get): " + faceVector.length);
                Log.d(TAG, "Sample values from faceVector: "
                        + Arrays.toString(Arrays.copyOf(faceVector, Math.min(faceVector.length, 10))));
            }

        } catch (Exception e) {
            Log.e(TAG, "Exception extracting feature vector: " + e.getMessage(), e);
            return new double[0];  
        }

        return faceVector;
    }


    private byte[] matToGrayscaleByteArray(Mat mat) {
        byte[] data = new byte[(int) (mat.total() * mat.channels())];
        mat.get(0, 0, data);
        return data;
    }



    private byte[] matToByteArray(Mat mat) {
        Log.d(TAG, "Converting Mat to byte array - Size: " + mat.rows() + "x" + mat.cols() + ", Type: " + mat.type());

        MatOfByte matOfByte = new MatOfByte();
        boolean result = Imgcodecs.imencode(".jpg", mat, matOfByte);

        if (result) {
            Log.d(TAG, "Mat successfully encoded to byte array, size: " + matOfByte.size().toString());
        } else {
            Log.e(TAG, "Failed to encode Mat to byte array.");
        }

        byte[] byteArray = matOfByte.toArray();
        Log.d(TAG, "Byte array length: " + byteArray.length);

        return byteArray;
    }


    private void saveNameToLocalStorage(String username) {
        SharedPreferences sharedPreferences = getSharedPreferences("FaceDetectionApp", Context.MODE_PRIVATE);
        SharedPreferences.Editor editor = sharedPreferences.edit();
        editor.putString("detectedName", username);
        editor.apply();

        Log.d(TAG, "Saved username to local storage: " + username);
    }

    
    private void redirectToMainActivity() {
        Intent intent = new Intent(CameraActivity.this, MainActivity.class);
        startActivity(intent);
        finish();  
    }


}
