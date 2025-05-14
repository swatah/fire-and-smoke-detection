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
