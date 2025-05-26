#include "smoke_model.h"
#include <vector>
#include <cstring>
#include <cstdio>

// Example detection result structure (adapt as needed)
typedef struct {
    int x1, y1, x2, y2;
    float score;
    int class_id;
} detection_result_t;

// Postprocess function for smoke model
// output_buf: pointer to model output (float*), output_size: number of floats
// img_width, img_height: original image size
// results: output vector to store detection results
void smoke_postprocess(const float* output_buf, int output_size, int img_width, int img_height, std::vector<detection_result_t>& results)
{
    // Example: Assume output is [x1, y1, x2, y2, score, ...] for a single detection
    // Adapt this logic to your model's output format

    if (output_size < 5) {
        fprintf(stderr, "Output size too small for postprocess\n");
        return;
    }

    detection_result_t det;
    det.x1 = static_cast<int>(output_buf[0] * img_width);
    det.y1 = static_cast<int>(output_buf[1] * img_height);
    det.x2 = static_cast<int>(output_buf[2] * img_width);
    det.y2 = static_cast<int>(output_buf[3] * img_height);
    det.score = output_buf[4];
    det.class_id = (output_size > 5) ? static_cast<int>(output_buf[5]) : 0;

    // Clamp coordinates
    if (det.x1 < 0) det.x1 = 0;
    if (det.y1 < 0) det.y1 = 0;
    if (det.x2 > img_width) det.x2 = img_width;
    if (det.y2 > img_height) det.y2 = img_height;

    // Example: Only add if score is high enough
    if (det.score > 0.5f) {
        results.push_back(det);
    }
}