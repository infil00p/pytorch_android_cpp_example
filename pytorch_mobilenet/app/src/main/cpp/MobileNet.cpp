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

#include "MobileNet.h"
#define LOG_TAG "MobileNet"

#define  LOGI(...)  __android_log_print(ANDROID_LOG_INFO,LOG_TAG,__VA_ARGS__)
#define  LOGD(...)  __android_log_print(ANDROID_LOG_DEBUG,LOG_TAG,__VA_ARGS__)
#define  LOGE(...)  __android_log_print(ANDROID_LOG_ERROR,LOG_TAG,__VA_ARGS__)

std::string APP_PATH = "/data/data/org.infil00p.pytorch_mobilenet/files/mobilenet_v2/model/";

namespace AdobeExample {


    // We pull the model from the assets directory onto storage for simplicity

    MobileNet::MobileNet() {
        auto qengines = at::globalContext().supportedQEngines();
        if (std::find(qengines.begin(), qengines.end(), at::QEngine::QNNPACK) !=
            qengines.end())
        {
            at::globalContext().setQEngine(at::QEngine::QNNPACK);
        }

        MobileCallGuard guard;
        mModule = torch::jit::load(APP_PATH + "mobilenet_v2.pt");
        mModule.eval();
    }

    // This is for the OpenCV pre-processing
    MobileNet::SharedPtr MobileNet::predict(cv::Mat & preprocessedData)
    {
        return predict((float *)preprocessedData.data);
    }

    // For this example, we know what the size is.
    MobileNet::SharedPtr MobileNet::predict(float * blob) {
        auto input = torch::from_blob(
                blob,
                torch::IntArrayRef({1, 3, 224, 224}),
                at::TensorOptions(at::kFloat));
        std::vector<torch::jit::IValue> pytorchInputs;
        pytorchInputs.push_back(input);
        auto output = [&]() {
            MobileCallGuard guard;

            return mModule.forward(pytorchInputs);
        }();

        // TODO: Find a better way to get the data out to C++ so we can do stuff with
        // it.  It kinda sucks to iterate over the shape object to calculate the size
        // in bytes of the tensor array.  (We don't always know this)
        if (output.tagKind() == "Tensor")
        {
            auto outTensor = output.toTensor();
            MobileNet::SharedPtr returnVal =
                    std::make_shared<std::vector<float> >(1000, 0);
            auto dataSize = sizeof(float) * 1000;
            memcpy(returnVal->data(), outTensor.data_ptr(), dataSize);
            return returnVal;
        }
        else
        {
            return nullptr;
        }

    }

    // This was shamelessly taken from NVIDIA
    // This definitely works on WinML, and this is the C++ version
    // of the OpenCV pre-processing code that I tested on PyTorch on desktop
    cv::Mat MobileNet::preProcess(cv::Mat & imageBGR, bool nchw) {
        int width  = 224;
        int height = 224;
        // Since this is coming from PyTorch, we need to follow PyTorch's
        // procedure.
        cv::Mat resizedImageBGR, resizedImageRGB, resizedImage;
        cv::resize(imageBGR,
                   resizedImageBGR,
                   cv::Size(width, height),
                   cv::InterpolationFlags::INTER_CUBIC);
        cv::cvtColor(resizedImageBGR,
                     resizedImageRGB,
                     cv::ColorConversionCodes::COLOR_BGR2RGB);
        resizedImageRGB.convertTo(resizedImage, CV_32F, 1.0 / 255);

        cv::Mat channels[3];
        cv::split(resizedImage, channels);
        // Normalization per channel
        // Normalization parameters obtained from
        // https://github.com/onnx/models/tree/master/vision/classification/squeezenet
        channels[0] = (channels[0] - 0.485) / 0.229;
        channels[1] = (channels[1] - 0.456) / 0.224;
        channels[2] = (channels[2] - 0.406) / 0.225;
        cv::merge(channels, 3, resizedImage);
        cv::Mat preprocessedMat;
        if (nchw)
        {
            cv::dnn::blobFromImage(resizedImage, preprocessedMat);
        }
        else
        {
            preprocessedMat = resizedImage;
        }

        return preprocessedMat;
    }

    MobileNet::SharedPtr MobileNet::getProbs(cv::Mat &input) {
        cv::Mat startMat = preProcess(input, true);
        return predict(startMat);
    }
}