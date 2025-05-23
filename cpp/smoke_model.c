#include "smoke_model.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int smoke_model_init(smoke_model_ctx *ctx, const char *model_path) {
    fprintf(stderr, "Entering smoke_model_init: model_path=%s\n", model_path);
    if (!ctx || !model_path) {
        fprintf(stderr, "Invalid argument to smoke_model_init\n");
        return -1;
    }

    FILE *fp = fopen(model_path, "rb");
    if (!fp) {
        fprintf(stderr, "Cannot open model file: %s\n", model_path);
        return -1;
    }

    fseek(fp, 0, SEEK_END);
    size_t model_size = ftell(fp);
    rewind(fp);
    fprintf(stderr, "Model size: %zu bytes\n", model_size);

    void *model_data = malloc(model_size);
    if (!model_data) {
        fclose(fp);
        fprintf(stderr, "Failed to allocate memory for model\n");
        return -1;
    }

    size_t read_len = fread(model_data, 1, model_size, fp);
    fclose(fp);
    if (read_len != model_size) {
        free(model_data);
        fprintf(stderr, "Failed to read complete model data: read=%zu, expected=%zu\n", read_len, model_size);
        return -1;
    }

    fprintf(stderr, "Calling rknn_init\n");
    int ret = rknn_init(&ctx->ctx, model_data, model_size, 0, NULL);
    free(model_data);
    if (ret != RKNN_SUCC) {
        fprintf(stderr, "rknn_init failed: %d\n", ret);
        return -1;
    }

    // In smoke_model_init, after rknn_init
rknn_sdk_version sdk_ver;
ret = rknn_query(ctx->ctx, RKNN_QUERY_SDK_VERSION, &sdk_ver, sizeof(sdk_ver));
if (ret == RKNN_SUCC) {
    fprintf(stderr, "RKNN SDK Version: %s\n", sdk_ver.api_version);
} else {
    fprintf(stderr, "Failed to query SDK version: %d\n", ret);
    rknn_destroy(ctx->ctx);
    ctx->ctx = 0;
    return -1;
}

    rknn_tensor_attr input_attr = {0};
    input_attr.index = 0;
    fprintf(stderr, "Querying input tensor attributes\n");
    ret = rknn_query(ctx->ctx, RKNN_QUERY_INPUT_ATTR, &input_attr, sizeof(input_attr));
    if (ret != RKNN_SUCC) {
        fprintf(stderr, "rknn_query input attr failed: %d\n", ret);
        rknn_destroy(ctx->ctx);
        ctx->ctx = 0;
        return -1;
    }

    fprintf(stderr, "Input tensor: n_dims=%d, dims=[%d, %d, %d, %d], scale=%f, zp=%d\n",
            input_attr.n_dims, input_attr.dims[0], input_attr.dims[1],
            input_attr.dims[2], input_attr.dims[3], input_attr.scale, input_attr.zp);

    if (input_attr.n_dims != 4) {
        fprintf(stderr, "Unexpected input tensor dims: %d\n", input_attr.n_dims);
        rknn_destroy(ctx->ctx);
        ctx->ctx = 0;
        return -1;
    }

    // Assume NHWC format: dims[3] is channels, dims[1] is height, dims[2] is width
    if (input_attr.dims[3] != 3) {
        fprintf(stderr, "Unexpected number of channels: %d, expected 3\n", input_attr.dims[3]);
        rknn_destroy(ctx->ctx);
        ctx->ctx = 0;
        return -1;
    }

    ctx->input_channel = input_attr.dims[3]; // C=3 (RGB channels)
    ctx->input_height = input_attr.dims[1];  // H=640
    ctx->input_width = input_attr.dims[2];   // W=640
    ctx->input_scale = (input_attr.scale != 0) ? input_attr.scale : 1.0f;
    ctx->input_zero_point = input_attr.zp;

    fprintf(stderr, "smoke_model_init completed: width=%d, height=%d, channels=%d\n",
            ctx->input_width, ctx->input_height, ctx->input_channel);
    return 0;
}

int resize_image(const image_buffer_t *src, image_buffer_t *dst, int dst_w, int dst_h) {
    fprintf(stderr, "Entering resize_image: src=%p, dst=%p, dst_w=%d, dst_h=%d\n",
            src, dst, dst_w, dst_h);
    if (!src || !dst || !src->virt_addr) {
        fprintf(stderr, "Invalid argument to resize_image\n");
        return -1;
    }

    const int channels = 3;
    size_t new_size = (size_t)dst_w * dst_h * channels;
    fprintf(stderr, "Resizing image to %dx%d, %zu bytes\n", dst_w, dst_h, new_size);

    uint8_t *resized_data;
    if (posix_memalign((void **)&resized_data, 16, new_size) != 0) {
        fprintf(stderr, "Failed to allocate aligned resized image buffer\n");
        return -1;
    }

    uint8_t *src_data = (uint8_t *)src->virt_addr;
    fprintf(stderr, "Source image: %dx%d, size=%zu\n", src->width, src->height, src->size);

    // Validate source image size
    size_t src_size = (size_t)src->width * src->height * channels;
    if (src->size != src_size) {
        fprintf(stderr, "Source image size mismatch: expected=%zu, actual=%zu\n",
                src_size, src->size);
        free(resized_data);
        return -1;
    }

    for (int y = 0; y < dst_h; y++) {
        int src_y = y * src->height / dst_h;
        if (src_y >= src->height) src_y = src->height - 1; // Prevent out-of-bounds
        for (int x = 0; x < dst_w; x++) {
            int src_x = x * src->width / dst_w;
            if (src_x >= src->width) src_x = src->width - 1; // Prevent out-of-bounds
            for (int c = 0; c < channels; c++) {
                resized_data[(y * dst_w + x) * channels + c] =
                    src_data[(src_y * src->width + src_x) * channels + c];
            }
        }
    }

    dst->virt_addr = resized_data;
    dst->width = dst_w;
    dst->height = dst_h;
    dst->size = new_size;

    fprintf(stderr, "resize_image completed successfully\n");
    return 0;
}

int smoke_model_infer(smoke_model_ctx *ctx, image_buffer_t *img, detection_result_t *result) {
    fprintf(stderr, "Entering smoke_model_infer: ctx=%p, img=%p, result=%p\n", ctx, img, result);
    if (!ctx || !img || !img->virt_addr || !result) {
        fprintf(stderr, "Invalid argument to smoke_model_infer\n");
        return -1;
    }

    // Compute expected image size
    size_t img_size = (size_t)img->width * img->height * 3;
    // Workaround: If img->size is 0, use computed size
    if (img->size == 0) {
        fprintf(stderr, "Warning: img->size is 0, using computed size=%zu (width=%d, height=%d, channels=3)\n",
                img_size, img->width, img->height);
        img->size = img_size;
    } else if (img->size != img_size) {
        fprintf(stderr, "Input image size mismatch: expected=%zu, actual=%zu\n",
                img_size, img->size);
        return -1;
    }

    image_buffer_t resized_img = {0};
    fprintf(stderr, "Calling resize_image\n");
    if (resize_image(img, &resized_img, ctx->input_width, ctx->input_height) != 0) {
        fprintf(stderr, "Failed to resize image\n");
        return -1;
    }

    const int width = ctx->input_width;
    const int height = ctx->input_height;
    const int channels = ctx->input_channel;
    size_t input_size = (size_t)width * height * channels;
    fprintf(stderr, "Input buffer size: %zu bytes\n", input_size);

    // Validate resized image size
    if (resized_img.size != input_size) {
        fprintf(stderr, "Resized image size mismatch: expected=%zu, actual=%zu\n",
                input_size, resized_img.size);
        free(resized_img.virt_addr);
        return -1;
    }

    int8_t *input_int8;
    if (posix_memalign((void **)&input_int8, 16, input_size) != 0) {
        fprintf(stderr, "Failed to allocate aligned input buffer\n");
        free(resized_img.virt_addr);
        return -1;
    }

    uint8_t *src = (uint8_t *)resized_img.virt_addr;
    float scale = ctx->input_scale;
    int zero_point = ctx->input_zero_point;
    fprintf(stderr, "Converting image to NHWC int8: scale=%f, zp=%d\n", scale, zero_point);

    if (!src) {
        fprintf(stderr, "Resized image buffer is NULL\n");
        free(input_int8);
        free(resized_img.virt_addr);
        return -1;
    }

    // Convert to NHWC format (HWC layout matches input image)
    for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
            for (int c = 0; c < channels; c++) {
                int src_idx = (h * width + w) * channels + c;
                int dst_idx = (h * width + w) * channels + c; // NHWC: same as HWC
                if (src_idx >= resized_img.size) {
                    fprintf(stderr, "Invalid source index: %d, max=%zu\n", src_idx, resized_img.size);
                    free(input_int8);
                    free(resized_img.virt_addr);
                    return -1;
                }
                int val = (int)((float)src[src_idx] / scale + zero_point);
                if (val > 127) val = 127;
                if (val < -128) val = -128;
                input_int8[dst_idx] = (int8_t)val;
            }
        }
    }

    rknn_input input = {
        .index = 0,
        .buf = input_int8,
        .size = input_size,
        .pass_through = 0,
        .type = RKNN_TENSOR_INT8,
        .fmt = RKNN_TENSOR_NHWC,
    };

    fprintf(stderr, "Calling rknn_inputs_set\n");
    int ret = rknn_inputs_set(ctx->ctx, 1, &input);
    if (ret != RKNN_SUCC) {
        fprintf(stderr, "Failed to set input: %d\n", ret);
        free(input_int8);
        free(resized_img.virt_addr);
        return -1;
    }

    fprintf(stderr, "Calling rknn_run\n");
    ret = rknn_run(ctx->ctx, NULL);
    if (ret != RKNN_SUCC) {
        fprintf(stderr, "Failed to run inference: %d\n", ret);
        free(input_int8);
        free(resized_img.virt_addr);
        return -1;
    }

    rknn_output outputs[1];
    memset(outputs, 0, sizeof(outputs));
    outputs[0].want_float = 1;

    fprintf(stderr, "Calling rknn_outputs_get\n");
    ret = rknn_outputs_get(ctx->ctx, 1, outputs, NULL);
    if (ret != RKNN_SUCC) {
        fprintf(stderr, "Failed to get outputs: %d\n", ret);
        free(input_int8);
        free(resized_img.virt_addr);
        return -1;
    }

    float *out = (float *)outputs[0].buf;
    if (!out) {
        fprintf(stderr, "Output buffer is NULL\n");
        rknn_outputs_release(ctx->ctx, 1, outputs);
        free(input_int8);
        free(resized_img.virt_addr);
        return -1;
    }

    fprintf(stderr, "Output received: x1=%f, y1=%f, x2=%f, y2=%f, score=%f, class_id=%d\n",
            out[0], out[1], out[2], out[3], out[4], (int)out[8]);

    int x1 = (int)(out[0] * img->width);
    int y1 = (int)(out[1] * img->height);
    int x2 = (int)(out[2] * img->width);
    int y2 = (int)(out[3] * img->height);

    if (x1 < 0) x1 = 0;
    if (y1 < 0) y1 = 0;
    if (x2 > img->width) x2 = img->width;
    if (y2 > img->height) y2 = img->height;

    result->x1 = x1;
    result->y1 = y1;
    result->x2 = x2;
    result->y2 = y2;
    result->score = out[4];
    result->class_id = (int)out[8];

    rknn_outputs_release(ctx->ctx, 1, outputs);
    free(input_int8);
    free(resized_img.virt_addr);

    fprintf(stderr, "smoke_model_infer completed successfully\n");
    return 0;
}

void smoke_model_release(smoke_model_ctx *ctx) {
    fprintf(stderr, "Entering smoke_model_release\n");
    if (ctx && ctx->ctx) {
        rknn_destroy(ctx->ctx);
        ctx->ctx = 0;
    }
    fprintf(stderr, "smoke_model_release completed\n");
}