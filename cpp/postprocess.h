// postprocess.h
#pragma once
#include <vector>
typedef struct {
    int x1, y1, x2, y2;
    float score;
    int class_id;
} detection_result_t;

void smoke_postprocess(const float* output_buf, int output_size, int img_width, int img_height, std::vector<detection_result_t>& results);