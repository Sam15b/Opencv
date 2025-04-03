package com.example.opencv_camera;

import android.app.ProgressDialog;
import android.content.Intent;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import org.opencv.android.OpenCVLoader;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;

import okhttp3.Call;
import okhttp3.Callback;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.Response;

public class MainActivity extends AppCompatActivity {
    static {
        if(OpenCVLoader.initDebug()){
            Log.d("MainActivity: ","Opencv is loaded");
        }
        else {
            Log.d("MainActivity: ","Opencv failed to load");
        }
    }

    private Button camera_button;
    private EditText usernameEditText;
    private TextView camera_Face_Aut;
    private SharedPreferences sharedPreferences;
    private static final String PREFS_NAME = "FaceDetectionApp";
    private static final String KEY_USERNAME = "detectedName";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        camera_button=findViewById(R.id.camera_button);
        camera_Face_Aut = findViewById(R.id.camera_Face_Aut);
        usernameEditText = findViewById(R.id.username_edittext);

        sharedPreferences = getSharedPreferences(PREFS_NAME, MODE_PRIVATE);

        String savedUsername = sharedPreferences.getString(KEY_USERNAME, "");
        if (savedUsername.isEmpty()) {

            camera_Face_Aut.setEnabled(false);
            Log.d("MainActivity: ","No !!! Value is not present in Local Storage");
        } else {

            camera_Face_Aut.setEnabled(true);
            Log.d("MainActivity: ","Yes !!! Local Storage" + savedUsername);
        }

        File user_data = new File(getFilesDir(), "user_data.json");

        if (user_data.exists()) {
            boolean deleted = user_data.delete();
            Log.d("MainActivity", "user_data.json deleted: " + deleted);
        }



        camera_button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                String username = usernameEditText.getText().toString().trim();

                if (username.isEmpty()) {
                  
                    Toast.makeText(MainActivity.this, "Please enter a username", Toast.LENGTH_SHORT).show();
                } else {
                    
                    Intent intent = new Intent(MainActivity.this, CameraActivity.class);
                    intent.putExtra("USERNAME", username);  
                    intent.addFlags(Intent.FLAG_ACTIVITY_CLEAR_TASK | Intent.FLAG_ACTIVITY_CLEAR_TOP);
                    startActivity(intent);
                }
            }
        });

        camera_Face_Aut.setOnClickListener(v -> {
            String username = sharedPreferences.getString(KEY_USERNAME, "");
            if (username.isEmpty()) {
                Toast.makeText(MainActivity.this, "Username not found", Toast.LENGTH_SHORT).show();
                return;
            }


            ProgressDialog progressDialog = new ProgressDialog(MainActivity.this);
            progressDialog.setMessage("Downloading user data, please wait...");
            progressDialog.setCancelable(false);
            progressDialog.show();

            // Start download process
            downloadAndStoreUserData(username, progressDialog);
        });


    }

    private void downloadAndStoreUserData(String username, ProgressDialog progressDialog) {
        String url = "http://192.168.29.217:5000/download/new";

        OkHttpClient client = new OkHttpClient();
        Request request = new Request.Builder().url(url).build();

        client.newCall(request).enqueue(new Callback() {
            @Override
            public void onFailure(Call call, IOException e) {
                Log.e("MainActivity", "Failed to download user_data.json", e);
                runOnUiThread(() -> {
                    progressDialog.dismiss();
                    Toast.makeText(MainActivity.this, "Download failed! Try again.", Toast.LENGTH_SHORT).show();
                });
            }

            @Override
            public void onResponse(Call call, Response response) throws IOException {
                if (!response.isSuccessful()) {
                    Log.e("MainActivity", "Failed to fetch user_data.json: " + response);
                    runOnUiThread(() -> {
                        progressDialog.dismiss();
                        Toast.makeText(MainActivity.this, "Failed to download data!", Toast.LENGTH_SHORT).show();
                    });
                    return;
                }

                // Get JSON response
                String jsonData = response.body().string();

                // Save JSON to internal storage
                File outputFile = new File(getFilesDir(), "user_data.json");
                try (FileOutputStream fos = new FileOutputStream(outputFile)) {
                    fos.write(jsonData.getBytes());
                }

                Log.d("MainActivity", "user_data.json downloaded successfully at " + outputFile.getAbsolutePath());

                
                runOnUiThread(() -> {
                    progressDialog.dismiss();  
                    Intent intent = new Intent(MainActivity.this, CameraActivity2.class);
                    intent.putExtra("USERNAME", username);
                    intent.putExtra("USER_DATA_PATH", outputFile.getAbsolutePath()); 
                    intent.addFlags(Intent.FLAG_ACTIVITY_CLEAR_TASK | Intent.FLAG_ACTIVITY_CLEAR_TOP);
                    startActivity(intent);
                });
            }
        });
    }
}