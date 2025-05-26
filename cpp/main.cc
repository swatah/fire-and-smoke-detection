#include <iostream>
#include <cstdlib>
#include <vector>
#include "smoke_model.h"
#include "image_utils.h"
#include "image_drawing.h"
#include "postprocess.h" // or create a postprocess.h and include that

int main(int argc, char** argv) {
    fprintf(stderr, "Entering main: argc=%d\n", argc);
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <model_path> <image_path>\n";
        return -1;
    }

    const char* model_path = argv[1];
    const char* image_path = argv[2];
    fprintf(stderr, "Model path: %s, Image path: %s\n", model_path, image_path);

    smoke_model_ctx ctx{};
    fprintf(stderr, "Calling smoke_model_init\n");
    if (smoke_model_init(model_path, &ctx) != 0) {
        std::cerr << "Failed to initialize model\n";
        return -1;
    }

    image_buffer_t image{};
    fprintf(stderr, "Calling read_image\n");
    if (read_image(image_path, &image) != 0 || !image.virt_addr) {
        std::cerr << "Failed to read image or image buffer is NULL\n";
        smoke_model_release(&ctx);
        return -1;
    }
    fprintf(stderr, "Image loaded: %dx%d, size=%zu\n", image.width, image.height, image.size);

    // After inference
    void* output_buf = nullptr;
    if (smoke_model_infer(&ctx, &image) == 0) {
        smoke_model_get_output(&ctx, &output_buf);

        // You need to know the output size (number of floats)
        int output_size = ctx.output_attr.n_elems; // or set appropriately

        std::vector<detection_result_t> results;
        smoke_postprocess(static_cast<float*>(output_buf), output_size, image.width, image.height, results);

        if (!results.empty()) {
            for (const auto& result : results) {
                std::cout << "Fire detected at (" << result.x1 << "," << result.y1 << ","
                          << result.x2 << "," << result.y2 << ") score=" << result.score << "\n";
                draw_rectangle(&image, result.x1, result.y1,
                               result.x2 - result.x1, result.y2 - result.y1,
                               COLOR_RED, 3);
                draw_text(&image, "FIRE", result.x1, result.y1 - 20, COLOR_RED, 12);
            }
        } else {
            std::cout << "No fire detected\n";
        }
        fprintf(stderr, "Calling write_image\n");
        if (write_image("output.png", &image) != 0) {
            std::cerr << "Failed to write output image\n";
        }
    } else {
        std::cerr << "Inference failed\n";
    }

    fprintf(stderr, "Calling smoke_model_release\n");
    smoke_model_release(&ctx);

    if (image.virt_addr) {
        fprintf(stderr, "Freeing image buffer\n");
        free(image.virt_addr);
        image.virt_addr = nullptr;
    }

    fprintf(stderr, "Exiting main\n");
    return 0;
}