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

#include <jni.h>
#include <string>

#include "MobileNet.h"

extern "C"
JNIEXPORT jint JNICALL
Java_com_adobe_pytorch_1mobilenet_MainActivity_startPredict(JNIEnv *env, jobject thiz,
                                                               jobject buffer, jint height,
                                                               jint width) {

    // Endianness matters, ARGB8888 in Android Speak is BGRA in reality
    jbyte* buff = (jbyte*)env->GetDirectBufferAddress(buffer);
    cv::Mat rgbaMat(height, width, CV_8UC4, buff);
    cv::Mat bgrMat;
    cv::cvtColor(rgbaMat, bgrMat, cv::COLOR_BGRA2BGR);
    AdobeExample::MobileNet model;
    std::shared_ptr<std::vector<float> > results = model.getProbs(bgrMat);
    auto resultVec = *results;

    // Process the output from the buffer.
    uint8_t maxValue         = 0;
    int idx                  = -1;
    for (int i = 0; i < resultVec.size(); i++)
    {
        if (resultVec[i] > maxValue)
        {
            maxValue = resultVec[i];
            idx      = i;
        }
    }

    // We have the final index,
    return idx;
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_adobe_pytorch_1mobilenet_MainActivity_startPredictWithTorchVision(JNIEnv *env,
                                                                              jobject thiz,
                                                                              jobject buffer) {
    jbyte* buff = (jbyte*)env->GetDirectBufferAddress(buffer);
    AdobeExample::MobileNet model;
    std::shared_ptr<std::vector<float> > results = model.predict((float*) buff);
    auto resultVec = *results;

    // Process the output from the buffer.
    uint8_t maxValue         = 0;
    int idx                  = -1;
    for (int i = 0; i < 1000; i++)
    {
        if (resultVec[i] > maxValue)
        {
            maxValue = resultVec[i];
            idx      = i;
        }
    }

    // We have the final index,
    return idx;
}