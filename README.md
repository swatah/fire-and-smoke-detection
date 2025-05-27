# Fire and Smoke Detection using EdgeYOLO

## 📦 ONNX Output Structure

The model(onnx) outputs a list of detections, where each detection includes:

| Index | Description                              |
| ----- | ---------------------------------------- |
| 0-3   | Bounding Box `[x1, y1, x2, y2]`          |
| 4     | Detection Score (max class probability)  |
| 5     | Background class probability             |
| 6     | Fire class probability                   |
| 7     | Smoke class probability                  |
| 8     | Predicted Class ID (argmax of index 5–7) |

# C++ build tools
 for c++ build we need the following in the home directory if the cmake list is not latered
 ## Rknn_model_zoo 
 git clone https://github.com/airockchip/rknn_model_zoo
 ## Rknn Toolkit
 git clone https://github.com/airockchip/rknn-toolkit2/tree/master
 ## Luckfox toolkit that contains the uclib c, c++, linkers
 git clone https://github.com/LuckfoxTECH/luckfox-pico/tree/main

 # C++ build instruction using cmake
 ```
  cd ~/{PATH}/fire-and-smoke-detection
  mkdir build && cd build
  cmake -DCMAKE_TOOLCHAIN_FILE=../rv1106.toolchain.cmake ..
  make
 ```
 
 
