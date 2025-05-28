#ifndef SMOKE_H
#define SMOKE_H

#include "rknn_api.h"
#include "utils/image_utils.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    rknn_context ctx;
    rknn_tensor_attr input_attr;
    rknn_tensor_attr output_attr;
    rknn_tensor_mem *input_mem;
    rknn_tensor_mem *output_mem;
    int input_width;
    int input_height;
    int input_channel;
    float input_scale;
    int input_zero_point;
} smoke_model_ctx;

/**
 * Initialize the smoke detection model.
 * 
 * @param model_path Path to the RKNN model file.
 * @param ctx Pointer to the model context struct.
 * @return 0 on success, negative value on failure.
 */
int smoke_model_init(const char *model_path, smoke_model_ctx *ctx);

/**
 * Run inference on a given image.
 * 
 * @param ctx Pointer to the model context.
 * @param img Pointer to the input image buffer.
 * @return 0 on success, negative value on failure.
 */
int smoke_model_infer(smoke_model_ctx *ctx, image_buffer_t *img);

/**
 * Get pointer to model output buffer.
 * 
 * @param ctx Pointer to the model context.
 * @param output_buf Address of pointer to receive output data.
 * @return 0 on success.
 */
int smoke_model_get_output(smoke_model_ctx *ctx, void **output_buf);

/**
 * Release all resources used by the model.
 * 
 * @param ctx Pointer to the model context.
 */
void smoke_model_release(smoke_model_ctx *ctx);

#ifdef __cplusplus
}
#endif

#endif // SMOKE_H
