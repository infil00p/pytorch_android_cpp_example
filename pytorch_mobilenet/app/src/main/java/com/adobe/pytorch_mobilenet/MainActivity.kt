/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 ~ Copyright 2021 Adobe
 ~
 ~ Licensed under the Apache License, Version 2.0 (the "License");
 ~ you may not use this file except in compliance with the License.
 ~ You may obtain a copy of the License at
 ~
 ~     http://www.apache.org/licenses/LICENSE-2.0
 ~
 ~ Unless required by applicable law or agreed to in writing, software
 ~ distributed under the License is distributed on an "AS IS" BASIS,
 ~ WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 ~ See the License for the specific language governing permissions and
 ~ limitations under the License.
 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

package com.adobe.pytorch_mobilenet

import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.media.ExifInterface
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import org.pytorch.MemoryFormat
import org.pytorch.torchvision.TensorImageUtils
import java.nio.ByteBuffer
import java.nio.ByteOrder

class MainActivity : AppCompatActivity() {

    private var mainBitmap: Bitmap? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Example of a call to a native method
        //findViewById<TextView>(R.id.sample_text).text = stringFromJNI()

        val handler = AssetHandler(this)

        // This is technically a post-processing step
        // Mobilenet classes taken from the Tensorflow Mobilenet example
        var labels = mutableListOf<String>()
        assets.open("mobilenet_v2/labels.txt").bufferedReader().useLines {
                lines -> labels.addAll(lines)
        }

        val button = findViewById<Button>(R.id.getImage)
        val predictButton = findViewById<Button>(R.id.doPredict)
        val predictNHWC = findViewById<Button>(R.id.doPredictWithNHWC)
        val predictTorchvisionNHWC = findViewById<Button>(R.id.doPredictTorchvisionWithNHWC)
        val predictGPU = findViewById<Button>(R.id.doPredictWithGPU)
        val predictGPUNHWC = findViewById<Button>(R.id.doPredictWithGPUNHWC)
        var textView = findViewById<TextView>(R.id.textView)

        button.setOnClickListener {
            try {
                val i = Intent(
                    Intent.ACTION_PICK,
                    MediaStore.Images.Media.EXTERNAL_CONTENT_URI
                )
                startActivityForResult(i, 0)
            } catch (e: Exception) {

            }
        }

        predictButton.setOnClickListener {
            if(mainBitmap != null) {
                val byteCount = mainBitmap!!.byteCount
                // This is critically important, if this is not directly allocated, it will not go
                // past JNI into C++
                var byteBuffer : ByteBuffer = ByteBuffer.allocateDirect(byteCount)
                mainBitmap!!.copyPixelsToBuffer(byteBuffer)

                // Our JNI returns an integer
                var predictVal : Int
                predictVal = startPredict(byteBuffer, mainBitmap!!.height, mainBitmap!!.width);

                // Grab the result from the string
                if(predictVal == -1)
                    predictVal = 0
                val resultString = labels[predictVal]
                runOnUiThread {
                    textView.text = resultString
                    //findObjects.isEnabled = true
                }
            }
        }

        predictNHWC.setOnClickListener {
            if(mainBitmap != null) {
                val byteCount = mainBitmap!!.byteCount
                // This is critically important, if this is not directly allocated, it will not go
                // past JNI into C++
                var byteBuffer : ByteBuffer = ByteBuffer.allocateDirect(byteCount)
                mainBitmap!!.copyPixelsToBuffer(byteBuffer)

                // Our JNI returns an integer
                var predictVal : Int
                predictVal = startPredictWithChannelsLast(byteBuffer, mainBitmap!!.height, mainBitmap!!.width);

                // Grab the result from the string
                if(predictVal == -1)
                    predictVal = 0
                val resultString = labels[predictVal]
                runOnUiThread {
                    textView.text = resultString
                    //findObjects.isEnabled = true
                }
            }
        }

        predictGPU.setOnClickListener {
            if(mainBitmap != null) {
                val byteCount = mainBitmap!!.byteCount
                // This is critically important, if this is not directly allocated, it will not go
                // past JNI into C++
                var byteBuffer : ByteBuffer = ByteBuffer.allocateDirect(byteCount)
                mainBitmap!!.copyPixelsToBuffer(byteBuffer)

                // Our JNI returns an integer
                var predictVal : Int
                predictVal = startPredictWithGPU(byteBuffer, mainBitmap!!.height, mainBitmap!!.width);

                // Grab the result from the string
                if(predictVal == -1)
                    predictVal = 0
                val resultString = labels[predictVal]
                runOnUiThread {
                    textView.text = resultString
                    //findObjects.isEnabled = true
                }
            }
        }

        predictGPUNHWC.setOnClickListener {
            if(mainBitmap != null) {
                val byteCount = mainBitmap!!.byteCount
                // This is critically important, if this is not directly allocated, it will not go
                // past JNI into C++
                var byteBuffer : ByteBuffer = ByteBuffer.allocateDirect(byteCount)
                mainBitmap!!.copyPixelsToBuffer(byteBuffer)

                // Our JNI returns an integer
                var predictVal : Int
                predictVal = startPredictWithGPUNHWC(byteBuffer, mainBitmap!!.height, mainBitmap!!.width);

                // Grab the result from the string
                if(predictVal == -1)
                    predictVal = 0
                val resultString = labels[predictVal]
                runOnUiThread {
                    textView.text = resultString
                    //findObjects.isEnabled = true
                }
            }
        }

        predictTorchvisionNHWC.setOnClickListener {
            if(mainBitmap != null) {
                val byteCount = mainBitmap!!.byteCount
                // This is critically important, if this is not directly allocated, it will not go
                // past JNI into C++
                val width = mainBitmap!!.width
                val height = mainBitmap!!.height
                var byteBuffer : ByteBuffer = ByteBuffer.allocateDirect(width * height * 3 * 4);
                byteBuffer.order(ByteOrder.nativeOrder())
                var floatBuffer = byteBuffer.asFloatBuffer()

                TensorImageUtils.bitmapToFloatBuffer(mainBitmap, 0, 0, 224, 224,
                    TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
                    TensorImageUtils.TORCHVISION_NORM_STD_RGB,
                    floatBuffer, 0, MemoryFormat.CHANNELS_LAST)

                // Our JNI returns an integer
                var predictVal : Int
                predictVal = startPredictWithTorchVision(byteBuffer)

                // Grab the result from the string
                if(predictVal == -1)
                    predictVal = 0
                val resultString = labels[predictVal]
                runOnUiThread {
                    textView.text = resultString
                    //findObjects.isEnabled = true
                }
            }
        }
    }

    override fun onActivityResult(reqCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(reqCode, resultCode, data)
        if (resultCode == RESULT_OK) {
            val imageUri = data!!.data
            val imageStream = contentResolver.openInputStream(imageUri!!)
            val exifStream = contentResolver.openInputStream(imageUri)
            val exif = ExifInterface(exifStream!!)
            val orientation = exif.getAttributeInt(
                ExifInterface.TAG_ORIENTATION,
                ExifInterface.ORIENTATION_NORMAL
            )
            val rotMatrix = Matrix()
            when (orientation) {
                ExifInterface.ORIENTATION_ROTATE_90 -> rotMatrix.postRotate(90f)
                ExifInterface.ORIENTATION_ROTATE_180 -> rotMatrix.postRotate(180f)
                ExifInterface.ORIENTATION_ROTATE_270 -> rotMatrix.postRotate(270f)
            }
            val selectedImage = BitmapFactory.decodeStream(imageStream)
            val rotatedBitmap = Bitmap.createBitmap(
                selectedImage, 0, 0,
                selectedImage.width, selectedImage.height,
                rotMatrix, true
            )

            this.mainBitmap = rotatedBitmap
            runOnUiThread {
                val imageView = findViewById<ImageView>(R.id.imageView)
                imageView.setImageBitmap(mainBitmap)
            }

        } else {
        }
    }

    // We always have some external fun!!!

    external fun startPredict(buffer: ByteBuffer, height: Int, width: Int) : Int

    external fun startPredictWithChannelsLast(buffer: ByteBuffer, height: Int, width: Int) : Int

    external fun startPredictWithGPU(buffer: ByteBuffer, height: Int, width: Int) : Int

    external fun startPredictWithGPUNHWC(buffer: ByteBuffer, height: Int, width: Int) : Int

    external fun startPredictWithTorchVision(buffer: ByteBuffer) : Int


    companion object {
        // Used to load the 'native-lib' library on application startup.
        init {
            System.loadLibrary("native-lib")
        }
    }
}