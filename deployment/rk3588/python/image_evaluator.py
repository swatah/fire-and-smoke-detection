
import numpy as np
import cv2
from rknn.api import RKNN

# --------- Config ---------
RKNN_MODEL = '../model/fire_smoke_rk3588_640x640_batch1.rknn'
INPUT_SIZE = 640
CLASSES = ['fire', 'smoke', '__none__']
STRIDES = [8, 16, 32]
CONF_THRESH = 0.3
NMS_THRESH = 0.45

# --------- Utils ---------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def preprocess(image_path):
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_input = img_rgb.astype(np.float32) / 255.0
    img_input = np.expand_dims(np.transpose(img_input, (2, 0, 1)), 0).copy()
    return img, img_input

def decode_outputs(regs, obj_confs, cls_confs):
    boxes, scores, class_ids = [], [], []

    for i in range(3):
        stride = STRIDES[i]

        # Check the shape and squeeze all singleton dimensions
        reg = np.squeeze(regs[i])         # [4, H, W]
        obj = sigmoid(np.squeeze(obj_confs[i]))  # [H, W]
        cls = sigmoid(np.squeeze(cls_confs[i]))  # [C, H, W]

        if reg.ndim != 3 or obj.ndim != 2 or cls.ndim != 3:
            raise ValueError(f"Unexpected shapes at stride {stride}: reg {reg.shape}, obj {obj.shape}, cls {cls.shape}")

        H, W = reg.shape[1], reg.shape[2]
        grid_y, grid_x = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')

        reg = reg.reshape(4, -1)
        obj = obj.reshape(-1)
        cls = cls.reshape(len(CLASSES), -1)

        for idx in range(reg.shape[1]):
            score = obj[idx] * np.max(cls[:-1, idx])  # skip "__none__"
            class_id = np.argmax(cls[:-1, idx])
            if score > CONF_THRESH:
                cx = (grid_x.flat[idx] + 0.5) * stride
                cy = (grid_y.flat[idx] + 0.5) * stride
                w, h = reg[2, idx], reg[3, idx]
                x1 = cx - w / 2
                y1 = cy - h / 2
                x2 = cx + w / 2
                y2 = cy + h / 2
                boxes.append([x1, y1, x2, y2])
                scores.append(score)
                class_ids.append(class_id)

    return np.array(boxes), np.array(scores), np.array(class_ids)

def nms(boxes, scores, iou_thresh):
    indices = cv2.dnn.NMSBoxes(
        boxes.tolist(), scores.tolist(), CONF_THRESH, iou_thresh)
    return indices.flatten() if len(indices) > 0 else []

def visualize(image, boxes, scores, class_ids):
    for box, score, cls_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = map(int, box)
        label = f"{CLASSES[cls_id]} {score:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return image

# --------- Main Inference ---------
def main():
    rknn = RKNN()
    print('[1] Loading RKNN model...')
    rknn.load_rknn(RKNN_MODEL)

    print('[2] Initializing runtime...')
    if rknn.init_runtime(target='rk3588'):
        print('Runtime init failed.')
        exit(-1)

    image_path = '../../../data/images/sample_01.png'
    original, input_tensor = preprocess(image_path)

    print('[3] Running inference...')
    outputs = rknn.inference(inputs=[input_tensor])
    output = outputs if isinstance(outputs, list) else [outputs]
    print(f"Output count: {len(output)}")
    print(f"Shape: {np.array(output[0]).shape}")    
    print(f"Type of outputs: {type(outputs)}")
    print(f"Length of outputs: {len(outputs)}")
    print(f"Type of outputs[0]: {type(outputs[0])}")
    print(f"Length of outputs[0]: {len(outputs[0])}")
    print('[DEBUG] Output lengths and shapes:')
    for i, out in enumerate(outputs):
        print(f"Output[{i}] shape: {np.array(out).shape}")


    # Unpack outputs into [reg0, reg1, reg2, obj0, obj1, obj2, cls0, cls1, cls2]
    regs = outputs[0:3]
    objs = outputs[3:6]
    clss = outputs[6:9]

    print('[4] Post-processing...')
    boxes, scores, class_ids = decode_outputs(regs, objs, clss)
    keep = nms(boxes, scores, NMS_THRESH)

    if len(keep) > 0:
        boxes = boxes[keep]
        scores = scores[keep]
        class_ids = class_ids[keep]

    print(f'[✓] Detections: {len(boxes)}')
    image_with_boxes = visualize(original.copy(), boxes, scores, class_ids)

    cv2.imshow('RKNN YOLOX Fire+Smoke', image_with_boxes)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    rknn.release()

if __name__ == '__main__':
    main()
