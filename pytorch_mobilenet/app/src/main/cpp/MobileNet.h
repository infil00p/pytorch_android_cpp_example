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

#ifndef PYTORCH_MOBILENET_MOBILENET_H
#define PYTORCH_MOBILENET_MOBILENET_H

#include "torch/script.h"
#include "opencv2/opencv.hpp"

namespace AdobeExample {


    struct MobileCallGuard
    {
        // AutoGrad is disabled for mobile by default.
        torch::autograd::AutoGradMode no_autograd_guard{false};
        // This needs to be on (taken from the test application)
        torch::AutoNonVariableTypeMode non_var_guard{true};
        // Disable graph optimizer to ensure list of unused ops are not changed for
        // custom mobile build.
        torch::jit::GraphOptimizerEnabledGuard no_optimizer_guard{false};
    };


    class MobileNet {

    public:

        using SharedPtr = std::shared_ptr<std::vector<float> >;
        MobileNet();
        SharedPtr predict(cv::Mat & preprocessedData);
        SharedPtr predict(float * blob);
        cv::Mat preProcess(cv::Mat & imageBGR, bool nchw);
        SharedPtr getProbs(cv::Mat & input);

    private:
        mutable torch::jit::script::Module mModule;
    };



}



#endif //PYTORCH_MOBILENET_MOBILENET_H
