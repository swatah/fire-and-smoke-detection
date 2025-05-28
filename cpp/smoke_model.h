#ifndef SMOKE_MODEL_H
#define SMOKE_MODEL_H

#include "rknn_api.h"
#include "utils/image_utils.h"

// Structure to hold smoke model context
typedef struct {
    rknn_context ctx;              // RKNN context
    rknn_mem* input_mem;           // Input memory buffer
    rknn_mem* output_mem;          // Output memory buffer
    rknn_tensor_attr input_attr;   // Input tensor attributes
    rknn_tensor_attr output_attr;  // Output tensor attributes
    int input_width;               // Input image width
    int input_height;              // Input image height
    int input_channel;             // Input image channels
    float input_scale;             // Input scale factor
    int input_zero_point;          // Input zero-point for quantization
} smoke_model_ctx;

// Initialize the smoke detection model
// @param model_path Path to the RKNN model file
// @param ctx Pointer to the smoke model context
// @return 0 on success, -1 on failure
int smoke_model_init(const char* model_path, smoke_model_ctx* ctx);

// Perform inference on an input image
// @param ctx Pointer to the smoke model context
// @param img Pointer to the input image buffer
// @return 0 on success, -1 on failure
int smoke_model_infer(smoke_model_ctx* ctx, image_buffer_t* img);

// Get the model output buffer
// @param ctx Pointer to the smoke model context
// @param output_buf Pointer to store the output buffer address
// @return 0 on success, -1 on failure
int smoke_model_get_output(smoke_model_ctx* ctx, void** output_buf);

// Release model resources
// @param ctx Pointer to the smoke model context
void smoke_model_release(smoke_model_ctx* ctx);

#endif // SMOKE_MODEL_H