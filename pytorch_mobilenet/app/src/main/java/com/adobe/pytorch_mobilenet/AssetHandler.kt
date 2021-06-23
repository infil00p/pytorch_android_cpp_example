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

import android.content.Context
import android.content.res.AssetManager
import android.util.Log
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.io.OutputStream

class AssetHandler internal constructor(var mCtx: Context) {
    var LOGTAG = "AssetHandler"

    inner class ModelFileInit internal constructor(
        var mModelName: String,
        var mDataDir: File,
        var mAssetManager: AssetManager,
        var mModelFiles: Array<String>
    ) {
        var mTopLevelFolder: File? = null
        var mResourcesFolder: File? = null
        var mModelFolder: File? = null

        @Throws(IOException::class)
        private fun InitModelFiles() {
            createModelDir()
            copyModelFiles()
        }

        @Throws(IOException::class)
        private fun copyFileUtil(
            files: Array<String>,
            dir: File?
        ) {
            // For this example, we're using the internal storage
            for (file in files) {
                val inputFile = mAssetManager.open("$mModelName/$file")
                var outFile: File
                outFile = File(dir, file)
                val out: OutputStream = FileOutputStream(outFile)
                val buffer = ByteArray(1024)
                var length: Int
                while (inputFile.read(buffer).also { length = it } != -1) {
                    out.write(buffer, 0, length)
                }
                inputFile.close()
                out.flush()
                out.close()
            }
        }

        @Throws(IOException::class)
        private fun copyModelFiles() {
            copyFileUtil(mModelFiles, mModelFolder)
        }


        private fun createTopLevelDir() {
            mTopLevelFolder = File(mDataDir.absolutePath, mModelName)
            mTopLevelFolder!!.mkdir()
        }

        private fun createModelDir() {
            mModelFolder = File(mTopLevelFolder!!.absolutePath, "model")
            mModelFolder!!.mkdir()
        }

        init {
            createTopLevelDir()
            InitModelFiles()
        }
    }

    @Throws(IOException::class)
    private fun Init() {
        val dataDirectory = mCtx.filesDir
        val assetManager = mCtx.assets
        val modelFiles = arrayOf(
            "labels.txt",
            "mobilenet_v2.pt",
            "mobilenet_v2_nhwc.pt"
            )

        val mobilenet_V2 = ModelFileInit(
            "mobilenet_v2",
            dataDirectory,
            assetManager,
            modelFiles
        )
    }

    init {
        try {
            Init()
        } catch (e: IOException) {
            Log.d(LOGTAG, "Unable to get models from storage")
        }
    }
}