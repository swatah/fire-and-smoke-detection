#include "smoke_model.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "utils/image_utils.h"
#include "rknn_api.h"

int smoke_model_init(const char *model_path, smoke_model_ctx *ctx) {
    int ret;
    rknn_context rknn_ctx = 0;

    // Use model path directly (let RKNN load from file)
    ret = rknn_init(&rknn_ctx, (void *)model_path, 0, 0, NULL);
    if (ret < 0) {
        fprintf(stderr, "rknn_init fail! ret=%d\n", ret);
        return -1;
    }

    // Query IO number
    rknn_input_output_num io_num;
    ret = rknn_query(rknn_ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC) {
        fprintf(stderr, "rknn_query IO num fail! ret=%d\n", ret);
        return -1;
    }

    // Query input tensor attributes
    rknn_tensor_attr input_attr = {0};
    input_attr.index = 0;
    ret = rknn_query(rknn_ctx, RKNN_QUERY_NATIVE_INPUT_ATTR, &input_attr, sizeof(input_attr));
    if (ret != RKNN_SUCC) {
        fprintf(stderr, "rknn_query input attr fail! ret=%d\n", ret);
        return -1;
    }

    // Query output tensor attributes
    rknn_tensor_attr output_attr = {0};
    output_attr.index = 0;
    ret = rknn_query(rknn_ctx, RKNN_QUERY_NATIVE_NHWC_OUTPUT_ATTR, &output_attr, sizeof(output_attr));
    if (ret != RKNN_SUCC) {
        fprintf(stderr, "rknn_query output attr fail! ret=%d\n", ret);
        return -1;
    }

    // Allocate input/output memory
    ctx->input_mem = rknn_create_mem(rknn_ctx, input_attr.size_with_stride);
    ctx->output_mem = rknn_create_mem(rknn_ctx, output_attr.size_with_stride);

    // Set input/output memory
    ret = rknn_set_io_mem(rknn_ctx, ctx->input_mem, &input_attr);
    if (ret < 0) {
        fprintf(stderr, "rknn_set_io_mem input fail! ret=%d\n", ret);
        return -1;
    }
    ret = rknn_set_io_mem(rknn_ctx, ctx->output_mem, &output_attr);
    if (ret < 0) {
        fprintf(stderr, "rknn_set_io_mem output fail! ret=%d\n", ret);
        return -1;
    }

    // Store context
    ctx->ctx = rknn_ctx;
    ctx->input_attr = input_attr;
    ctx->output_attr = output_attr;
    ctx->input_width = input_attr.dims[2];
    ctx->input_height = input_attr.dims[1];
    ctx->input_channel = input_attr.dims[3];
    ctx->input_scale = input_attr.scale;
    ctx->input_zero_point = input_attr.zp;

    return 0;
}

int smoke_model_infer(smoke_model_ctx *ctx, image_buffer_t *img) {
    image_buffer_t dst = {0};
    dst.width = ctx->input_width;
    dst.height = ctx->input_height;
    dst.format = IMAGE_FORMAT_RGB888; // Force RGB888 for model input
    dst.width_stride = ctx->input_attr.w_stride ? ctx->input_attr.w_stride : ctx->input_width;
    dst.height_stride = ctx->input_height;
    dst.size = ctx->input_attr.size_with_stride;
    dst.fd = ctx->input_mem->fd;
    dst.virt_addr = (unsigned char *)ctx->input_mem->virt_addr;

    // Log input image details
    fprintf(stderr, "smoke_model_infer: img->virt_addr=%p, img->size=%zu, img->width=%d, img->height=%d, img->format=0x%x\n",
            img->virt_addr, img->size, img->width, img->height, img->format);

    // Validate source image
    if (!img->virt_addr || img->size < img->width * img->height * 3) {
        fprintf(stderr, "Invalid source image: virt_addr=%p, size=%zu, expected>=%d\n",
                img->virt_addr, img->size, img->width * img->height * 3);
        return -1;
    }

    // Correct img->format if it indicates YUV but data is RGB888
    image_format_t original_format = img->format;
    if (img->format == IMAGE_FORMAT_YUV420SP_NV12 && img->size == img->width * img->height * 3) {
        fprintf(stderr, "Warning: img->format=0x1 (YUV420SP_NV12) but size matches RGB888; correcting to 0x2\n");
        img->format = IMAGE_FORMAT_RGB888; // Correct to RGB888
    }

    // Validate destination buffer
    if (!dst.virt_addr || dst.size < dst.width * dst.height * 3 || dst.fd <= 0) {
        fprintf(stderr, "Invalid destination: virt_addr=%p, size=%zu, expected>=%d, fd=%d\n",
                dst.virt_addr, dst.size, dst.width * dst.height * 3, dst.fd);
        return -1;
    }

    // Create a corrected source image copy to ensure src format is RGB888
    image_buffer_t src_corrected = *img;
    src_corrected.format = IMAGE_FORMAT_RGB888;

    fprintf(stderr, "Calling convert_image: src=%p (%dx%d, fmt=0x%x), dst=%p (%dx%d, fmt=0x%x, stride=%d)\n",
            src_corrected.virt_addr, src_corrected.width, src_corrected.height, src_corrected.format,
            dst.virt_addr, dst.width, dst.height, dst.format, dst.width_stride);
    if (convert_image(&src_corrected, &dst, NULL, NULL, 0) != 0) {
        fprintf(stderr, "convert_image failed\n");
        img->format = original_format; // Restore original format
        return -1;
    }

    int ret = rknn_run(ctx->ctx, NULL);
    if (ret != RKNN_SUCC) {
        fprintf(stderr, "rknn_run failed: %d\n", ret);
        img->format = original_format; // Restore original format
        return -1;
    }

    img->format = original_format; // Restore original format
    return 0;
}

int smoke_model_get_output(smoke_model_ctx *ctx, void **output_buf) {
    // Output is already in ctx->output_mem->virt_addr
    *output_buf = ctx->output_mem->virt_addr;
    return 0;
}

void smoke_model_release(smoke_model_ctx *ctx) {
    if (ctx->ctx) {
        rknn_destroy(ctx->ctx);
        ctx->ctx = 0;
    }
    if (ctx->input_mem) {
        rknn_destroy_mem(ctx->ctx, ctx->input_mem);
        ctx->input_mem = NULL;
    }
    if (ctx->output_mem) {
        rknn_destroy_mem(ctx->ctx, ctx->output_mem);
        ctx->output_mem = NULL;
    }
}