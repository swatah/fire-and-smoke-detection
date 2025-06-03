import numpy as np
import cv2
import sys
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

def preprocess_frame(frame):
    img_resized = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    #img_input = img_rgb.astype(np.float32) / 255.0
    img_input = np.expand_dims(np.transpose(img_rgb, (2, 0, 1)), 0).copy()
    return img_resized, img_input

def decode_outputs(regs, obj_confs, cls_confs):
    boxes, scores, class_ids = [], [], []

    for i in range(3):
        stride = STRIDES[i]

        reg = np.squeeze(regs[i])         # [4, H, W]
        obj = sigmoid(np.squeeze(obj_confs[i]))  # [H, W]
        cls = sigmoid(np.squeeze(cls_confs[i]))  # [C, H, W]

        H, W = reg.shape[1], reg.shape[2]
        grid_y, grid_x = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')

        reg = reg.reshape(4, -1)
        obj = obj.reshape(-1)
        cls = cls.reshape(len(CLASSES), -1)

        for idx in range(reg.shape[1]):
            score = obj[idx] * np.max(cls[:-1, idx])  # skip "__none__"
            class_id = np.argmax(cls[:-1, idx])
            print(class_id, score)
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
    if len(sys.argv) < 2:
        print("Usage: python image_evaluator.py <video_device_index>")
        print("Example: python image_evaluator.py 0")
        exit(-1)

    video_index = int(sys.argv[1])
    cap = cv2.VideoCapture(video_index)

    if not cap.isOpened():
        print(f"Failed to open video device {video_index}")
        exit(-1)

    rknn = RKNN()
    print('[1] Loading RKNN model...')
    rknn.load_rknn(RKNN_MODEL)

    print('[2] Initializing runtime...')
    if rknn.init_runtime(target='rk3588'):
        print('Runtime init failed.')
        exit(-1)

    print('[3] Starting video stream...')
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        original, input_tensor = preprocess_frame(frame)

        outputs = rknn.inference(inputs=[input_tensor])
        # outputs is a list of 9 outputs
        regs = outputs[0:3]
        objs = outputs[3:6]
        clss = outputs[6:9]

        boxes, scores, class_ids = decode_outputs(regs, objs, clss)
        keep = nms(boxes, scores, NMS_THRESH)

        if len(keep) > 0:
            boxes = boxes[keep]
            scores = scores[keep]
            class_ids = class_ids[keep]
        else:
            boxes, scores, class_ids = np.array([]), np.array([]), np.array([])
        
        
        image_with_boxes = visualize(original.copy(), boxes, scores, class_ids)

        cv2.imshow('RKNN YOLOX Fire+Smoke - Live', image_with_boxes)

        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'):  # ESC or q to quit
            break

    cap.release()
    cv2.destroyAllWindows()
    rknn.release()

if __name__ == '__main__':
    main()
