#include <iostream>
#include <cstdlib>
#include "smoke_model.h"
#include "image_utils.h"
#include "image_drawing.h"

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
    if (smoke_model_init(&ctx, model_path) != 0) {
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

    detection_result_t result{};
    fprintf(stderr, "Calling smoke_model_infer\n");
    if (smoke_model_infer(&ctx, &image, &result) == 0) {
        fprintf(stderr, "Inference result: score=%f, class_id=%d\n", result.score, result.class_id);
        if (result.score > 0.5f && result.class_id == 6) {
            std::cout << "Fire detected at (" << result.x1 << "," << result.y1 << ","
                      << result.x2 << "," << result.y2 << ") score=" << result.score << "\n";
            fprintf(stderr, "Drawing rectangle and text\n");
            draw_rectangle(&image, result.x1, result.y1,
                           result.x2 - result.x1, result.y2 - result.y1,
                           COLOR_RED, 3);
            draw_text(&image, "FIRE", result.x1, result.y1 - 20, COLOR_RED, 12);
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