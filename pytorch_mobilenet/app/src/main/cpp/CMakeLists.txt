# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.

cmake_minimum_required(VERSION 3.10.2)

# Declares and names the project.

project("pytorch_mobilenet")

set(build_DIR ${CMAKE_SOURCE_DIR}/../../../build)
set(OpenCV_Android ${CMAKE_SOURCE_DIR}/../../../../../external/OpenCV-android-sdk)

file(GLOB PYTORCH_INCLUDE_DIRS "${build_DIR}/pytorch_android*.aar/headers")
file(GLOB PYTORCH_LINK_DIRS "${build_DIR}/pytorch_android*.aar/jni/${ANDROID_ABI}")


find_library(PYTORCH_LIBRARY pytorch_jni
        PATHS ${PYTORCH_LINK_DIRS}
        NO_CMAKE_FIND_ROOT_PATH)

find_library(FBJNI_LIBRARY fbjni
        PATHS ${PYTORCH_LINK_DIRS}
        NO_CMAKE_FIND_ROOT_PATH)

# If someone would find a good AAR that included headers, that would be awesome
# either that, or something that uses prefab
add_library(lib_opencv SHARED IMPORTED)
set_target_properties(
        lib_opencv
        PROPERTIES
        IMPORTED_LOCATION
        ${OpenCV_Android}/sdk/native/libs/${ANDROID_ABI}/libopencv_java4.so)

# Set all the include directories
# We create an include directory of headers that we want to use independent of
# the Android project, like OpenCV (We are using OpenCV 4.1 here)

include_directories(
        ${PYTORCH_INCLUDE_DIRS}
        ${OpenCV_Android}/sdk/native/jni/include
)

add_library( # Sets the name of the library.
             native-lib
             # Sets the library as a shared library.
             SHARED
             # Provides a relative path to your source file(s).
             native-lib.cpp
             MobileNet.cpp
             MobileNetNHWC.cpp
             MobileNetGPU.cpp
             MobileNetGPUNHWC.cpp)


find_library( # Sets the name of the path variable.
              log-lib
              # Specifies the name of the NDK library that
              # you want CMake to locate.
              log )


target_link_libraries( # Specifies the target library.
                        native-lib PRIVATE
                        -ljnigraphics
                        ${FBJNI_LIBRARY}
                        ${PYTORCH_LIBRARY}
                        lib_opencv
                        ${log-lib} )