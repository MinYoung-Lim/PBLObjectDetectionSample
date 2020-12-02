package com.example.pblobjectdetectionsample;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import com.google.android.gms.tasks.OnCompleteListener;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.android.gms.tasks.Task;
import com.google.firebase.ml.common.modeldownload.FirebaseModelDownloadConditions;
import com.google.firebase.ml.common.modeldownload.FirebaseModelManager;
import com.google.firebase.ml.custom.FirebaseCustomRemoteModel;

import org.tensorflow.lite.Interpreter;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

public class MainActivity extends AppCompatActivity {

    public FirebaseCustomRemoteModel remoteModel;

    public Interpreter interpreter;
    public ByteBuffer modelOutput;
    public Bitmap bitmapInput;
    Button btn_gallery;
    ImageView iv_img;
    TextView tv_text;
    Uri uri;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        btn_gallery = findViewById(R.id.btn_gallery);
        iv_img = findViewById(R.id.iv_img);

        /*btn_gallery.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                //이미지를 선택
                Intent intent = new Intent();
                intent.setType("image/*");
                intent.setAction(Intent.ACTION_GET_CONTENT);
                startActivityForResult(Intent.createChooser(intent, "이미지를 선택하세요."), 0);
            }
        });*/

        Intent intent = new Intent(getApplicationContext(), githubCodeActivity.class);
        startActivity(intent);

    }

    //결과 처리
    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        //request코드가 0이고 OK를 선택했고 data에 뭔가가 들어 있다면
        if(requestCode == 0 && resultCode == RESULT_OK){
            uri = data.getData();
            Log.d("URI", "uri:" + String.valueOf(uri));
            try {
                //Uri 파일을 Bitmap으로 만들어서 ImageView에 집어 넣는다.
                bitmapInput = MediaStore.Images.Media.getBitmap(getContentResolver(), uri);
                iv_img.setImageBitmap(bitmapInput);
            } catch (IOException e) {
                e.printStackTrace();
            }

            //modelDownload();
            //modelLocalDownload();
            //initInterpreter();
            //runInterpreter();
            //detection();

        }
    }




    private void modelLocalDownload() {
        remoteModel =
                new FirebaseCustomRemoteModel.Builder("final_model2").build();
        FirebaseModelManager.getInstance().getLatestModelFile(remoteModel)
                .addOnCompleteListener(new OnCompleteListener<File>() {
                    @Override
                    public void onComplete(@NonNull Task<File> task) {
                        File modelFile = task.getResult();
                        if (modelFile != null) {
                            interpreter = new Interpreter(modelFile);
                        } else {
                            try {
                                InputStream inputStream = getAssets().open("final_model2.tflite");
                                byte[] model = new byte[inputStream.available()];
                                inputStream.read(model);
                                ByteBuffer buffer = ByteBuffer.allocateDirect(model.length)
                                        .order(ByteOrder.nativeOrder());
                                buffer.put(model);
                                interpreter = new Interpreter(buffer);
                            } catch (IOException e) {
                                // File not found?
                            }
                        }

                        Log.e("localDownload", "성공");

                    }
                });
    }

    private void detection() {
        modelOutput.rewind();
        FloatBuffer probabilities = modelOutput.asFloatBuffer();
        try {
            BufferedReader reader = new BufferedReader(
                    new InputStreamReader(getAssets().open("custom_labels.txt")));
            for (int i = 0; i < probabilities.capacity(); i++) {
                String label = reader.readLine();
                float probability = probabilities.get(i);
                Log.e("detection()", String.format("%s: %1.4f", label, probability));
            }
        } catch (IOException e) {
            // File not found?
        }
    }

    private void runInterpreter() {
        Bitmap bitmap = Bitmap.createScaledBitmap(bitmapInput, 300, 300, true);
        ByteBuffer input = ByteBuffer.allocateDirect(300 * 300 * 3 * 4).order(ByteOrder.nativeOrder());
        for (int y = 0; y < 300; y++) {
            for (int x = 0; x < 300; x++) {
                int px = bitmap.getPixel(x, y);

                // Get channel values from the pixel value.
                int r = Color.red(px);
                int g = Color.green(px);
                int b = Color.blue(px);

                // Normalize channel values to [-1.0, 1.0]. This requirement depends
                // on the model. For example, some models might require values to be
                // normalized to the range [0.0, 1.0] instead.
                float rf = (r - 127) / 255.0f;
                float gf = (g - 127) / 255.0f;
                float bf = (b - 127) / 255.0f;

                input.putFloat(rf);
                input.putFloat(gf);
                input.putFloat(bf);
            }
        }

        int bufferSize = 10 * java.lang.Float.SIZE / java.lang.Byte.SIZE;
        modelOutput = ByteBuffer.allocateDirect(bufferSize).order(ByteOrder.nativeOrder());
        interpreter.run(input, modelOutput);



    }

    private void initInterpreter() {
        FirebaseModelManager.getInstance().getLatestModelFile(remoteModel)
                .addOnCompleteListener(new OnCompleteListener<File>() {
                    @Override
                    public void onComplete(@NonNull Task<File> task) {
                        File modelFile = task.getResult();
                        if (modelFile != null) {
                            interpreter = new Interpreter(modelFile);
                        }
                    }
                });
    }

    private void modelDownload() {
        remoteModel =
                new FirebaseCustomRemoteModel.Builder("dog-cat-detector").build();
        FirebaseModelDownloadConditions conditions = new FirebaseModelDownloadConditions.Builder()
                .requireWifi()
                .build();
        FirebaseModelManager.getInstance().download(remoteModel, conditions)
                .addOnSuccessListener(new OnSuccessListener<Void>() {
                    @Override
                    public void onSuccess(Void v) {
                        // Download complete. Depending on your app, you could enable
                        // the ML feature, or switch from the local model to the remote
                        // model, etc.
                        Log.e("모델다운로드", "성공");
                    }

                });


    }
}