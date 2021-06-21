### Basic Example of PyTorch Mobile C++ API

This is an example of how to use PyTorch Mobile on Android using the NDK, and tying it
together with pre-processing data written in OpenCV for portability.

This method is required if you're looking to port models across various frameworks for
inference, such as CoreML and WinML so that the pre-processing that was originally written
in Python only has to be re-implemented once, .instead of numerous times.

The use of OpenCV is optional, but was chosen due to it being common and having a good 
correspondence between the Python and C++ APIs.

## HOW-TO RUN

1. Download OpenCV 4.1 for Android and copy the related shared libraries
   into the pytorch_mobilenet/app/src/main/libs directory in
   their respective platform folders (x86, x86_64, armeabi-v7a, arm64-v8a)  (Create a PR if you
   know of a known working AAR of OpenCV with the C++ libraries)
1. Use the Jupyter Notebook in mobilenet_v2_work to export a .pt file from the MobilenetV2 model 
   origianlly from torchvision
1. Copy that file to the pytorch_mobile/app/src/main/assets directory
1. Build and Run the Application.


This example code is licenced under the Apache 2.0 Licence.

All other code and headers are subject to their licences and copyright holders.

