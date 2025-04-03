package com.example.opencv_camera;

import android.Manifest;
import android.app.Activity;
import android.content.Context;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.util.Log;
import android.view.SurfaceView;

import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import org.json.JSONArray;
import org.json.JSONObject;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.KNearest;
import org.opencv.ml.Ml;
import org.opencv.objdetect.CascadeClassifier;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.*;


public class CameraActivity2 extends Activity implements CameraBridgeViewBase.CvCameraViewListener2 {
    private static final String TAG = "CameraActivity2";

    private Mat mRgba;
    private Mat mGray;

    private KNearest knnModel;

    private ExecutorService executorService = Executors.newSingleThreadExecutor();
    private Future<Void> trainingFuture;
    private List<String> storedNames = new ArrayList<>();

    private CameraBridgeViewBase mOpenCvCameraView;

    private CascadeClassifier cascadeClassifier;

    private String username;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_camera2);

        username = getIntent().getStringExtra("USERNAME");
        Log.d(TAG, "Username received: " + username);
        trainKNNAsync();

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
        Log.d(TAG, "Calling .json file ");

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
        mGray = inputFrame.gray();

        MatOfRect rects = new MatOfRect();
        cascadeClassifier.detectMultiScale(mGray,rects,1.1,2);

        for (Rect rect : rects.toList()){
            
            Mat faceROI = new Mat(mGray, rect);

            Imgproc.resize(faceROI, faceROI, new Size(50, 50));

            float[] flattenedFace = flattenMat(faceROI);

            Log.d(TAG, "Flatten Size: " + flattenedFace.length);

            String recognizedName = recognizeFace(flattenedFace);

            if (!recognizedName.equals("Error")) {
                Imgproc.rectangle(mRgba, rect, new Scalar(0, 255, 0), 5); 
                Imgproc.putText(mRgba, recognizedName, new Point(rect.x, rect.y - 10),
                        Imgproc.FONT_HERSHEY_SIMPLEX, 0.8, new Scalar(255, 0, 0), 2);
            }
        }

        return mRgba;
    }

    private float[] flattenMat(Mat faceMat) {

        faceMat.convertTo(faceMat, CvType.CV_32F);

        float[] faceArray = new float[(int) faceMat.total()];

        faceMat.get(0, 0, faceArray);

        return faceArray;
    }

    private void trainKNNAsync() {
        trainingFuture = executorService.submit(() -> {
            trainKNN();
            return null;
        });
    }

    private void trainKNN() {
        Map<String, List<float[]>> userData = loadStoredData();
        if (userData.isEmpty()) return;

        List<float[]> storedFaces = new ArrayList<>();
        storedNames = new ArrayList<>();  
        for (Map.Entry<String, List<float[]>> entry : userData.entrySet()) {
            for (float[] face : entry.getValue()) {
                storedFaces.add(face);
                storedNames.add(entry.getKey());
            }
        }

        Mat trainingData = new Mat(storedFaces.size(), storedFaces.get(0).length, CvType.CV_32F);
        Mat labels = new Mat(storedFaces.size(), 1, CvType.CV_32S);

        for (int i = 0; i < storedFaces.size(); i++) {
            trainingData.put(i, 0, storedFaces.get(i));
            labels.put(i, 0, i);
        }

        knnModel = KNearest.create();
        knnModel.train(trainingData, Ml.ROW_SAMPLE, labels);
        Log.d(TAG, "KNN Model trained successfully");
    }

    private String recognizeFace(float[] newFace) {
        try {
            if (trainingFuture != null) {
                trainingFuture.get(); 
            }

            if (knnModel == null) {
                Log.e(TAG, "KNN Model is not trained!");
                return "Error";
            }

            Mat testFace = new Mat(1, newFace.length, CvType.CV_32F);
            testFace.put(0, 0, newFace);

            Mat results = new Mat();
            knnModel.findNearest(testFace, 5, results);

            int matchedIndex = (int) results.get(0, 0)[0];

            if (matchedIndex >= 0 && matchedIndex < storedNames.size()) {
                return storedNames.get(matchedIndex);
            } else {
                return "Unknown";
            }
        } catch (Exception e) {
            e.printStackTrace();
            return "Error";
        }
    }


    private Map<String, List<float[]>> loadStoredData() {
        Map<String, List<float[]>> userData = new HashMap<>();
        try {
            File file = new File(getFilesDir(), "user_data.json");
            if (!file.exists()) {
                Log.e(TAG, "User data file not found!");
                return userData;
            }

            String jsonContent = new String(Files.readAllBytes(file.toPath()), StandardCharsets.UTF_8);
            JSONObject jsonObject = new JSONObject(jsonContent);

            Iterator<String> keys = jsonObject.keys();
            while (keys.hasNext()) {
                String userName = keys.next();
                JSONArray faceArray = jsonObject.getJSONArray(userName);

                List<float[]> faceEmbeddings = new ArrayList<>();
                for (int i = 0; i < faceArray.length(); i++) {
                    JSONArray faceVector = faceArray.getJSONArray(i);
                    float[] faceData = new float[faceVector.length()];
                    for (int j = 0; j < faceVector.length(); j++) {
                        faceData[j] = (float) faceVector.getDouble(j);
                    }
                    faceEmbeddings.add(faceData);
                }
                userData.put(userName, faceEmbeddings);
            }

        } catch (Exception e) {
            Log.e(TAG, "Error loading user data", e);
        }
        return userData;
    }



    private List<float[]> loadStoredFaces() {
        try {
            File facesDataFile = new File(getFilesDir(), "faces_data.pkl");
            if (!facesDataFile.exists()) {
                Log.d(TAG, "No stored data found: faces_data.pkl missing.");
            }else{
                Log.d(TAG, "stored data faces_data.pkl  "+facesDataFile);
            }
            FileInputStream fis = new FileInputStream(facesDataFile);
            ObjectInputStream ois = new ObjectInputStream(fis);
            List<float[]> storedFaces = (List<float[]>) ois.readObject();
            ois.close();
            return storedFaces;
        } catch (Exception e) {
            e.printStackTrace();
            return new ArrayList<>();
        }
    }

    private List<String> loadStoredNames() {
        try {
            File nameFile = new File(getFilesDir(), "name.pkl");
            if (!nameFile.exists()) {
                Log.d(TAG, "No stored data found: name.pkl missing.");
            }else{
                Log.d(TAG, "stored data name.pkl  "+nameFile);
            }
            FileInputStream fis = new FileInputStream(nameFile);
            ObjectInputStream ois = new ObjectInputStream(fis);
            List<String> storedNames = (List<String>) ois.readObject();
            ois.close();
            return storedNames;
        } catch (Exception e) {
            e.printStackTrace();
            Log.e(TAG, "Error in stored data name.pkl  "+e);
            return new ArrayList<>();
        }
    }




}