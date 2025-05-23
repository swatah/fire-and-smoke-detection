#ifndef SMOKE_MODEL_H
#define SMOKE_MODEL_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include "rknn_api.h"
#include "image_utils.h"

typedef struct {
    rknn_context ctx;
    int input_width;
    int input_height;
    int input_channel;
    float input_scale;
    int input_zero_point;
} smoke_model_ctx;

typedef struct {
    int x1, y1, x2, y2;
    float score;
    int class_id;
} detection_result_t;

int smoke_model_init(smoke_model_ctx *ctx, const char *model_path);
int smoke_model_infer(smoke_model_ctx *ctx, image_buffer_t *img, detection_result_t *result);
void smoke_model_release(smoke_model_ctx *ctx);

#ifdef __cplusplus
}
#endif

#endif // SMOKE_MODEL_H
