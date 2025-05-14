import cv2
import numpy as np


def preprocessing(img, input_size, swap=(2, 0, 1)):
    # if len(img.shape) == 3:
    #     padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    # else:
    #     padded_img = np.ones(input_size, dtype=np.uint8) * 114

    # r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    # resized_img = cv2.resize(
    #     img,
    #     (int(img.shape[1] * r), int(img.shape[0] * r)),
    #     interpolation=cv2.INTER_LINEAR,
    # ).astype(np.uint8)
    # padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    new_w = int(img.shape[1] * r)
    new_h = int(img.shape[0] * r)

    resized_img = cv2.resize(
        img, (new_w, new_h), interpolation=cv2.INTER_LINEAR
    ).astype(np.uint8)

    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 0
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 0

    top = (input_size[0] - new_h) // 2
    left = (input_size[1] - new_w) // 2

    padded_img[top : top + new_h, left : left + new_w] = resized_img

    input_image = padded_img.transpose(swap)
    input_image = np.ascontiguousarray(input_image, dtype=np.float32)
    input_image = np.expand_dims(input_image, axis=0)
    return input_image, padded_img, r


def filter_box(output, scale_range):
    min_scale, max_scale = scale_range
    w = output[:, 2] - output[:, 0]
    h = output[:, 3] - output[:, 1]
    area = w * h
    keep = (area > min_scale**2) & (area < max_scale**2)
    return output[keep]


def nms_numpy(boxes, scores, iou_threshold):
    """Pure NumPy NMS."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        inter_w = np.maximum(0, xx2 - xx1)
        inter_h = np.maximum(0, yy2 - yy1)
        inter_area = inter_w * inter_h

        union_area = areas[i] + areas[order[1:]] - inter_area
        iou = inter_area / (union_area + 1e-6)

        order = order[1:][iou <= iou_threshold]

    return keep


def postprocessing(results, rs):
    outs = postprocess(
        results, num_classes=3, conf_thre=0.60, nms_thre=0.40, class_agnostic=False
    )

    if not isinstance(rs, list):
        rs = [rs]

    for i in range(min(len(rs), len(outs))):
        if outs[i] is not None:
            # print(f"Original outs[{i}] shape: {outs[i].shape}")

            if outs[i].ndim == 1:
                num_elements = outs[i].shape[0]
                if num_elements % 9 == 0:
                    outs[i] = outs[i].reshape(-1, 9)
                    # print(f"Reshaped outs[{i}] to: {outs[i].shape}")
                else:
                    # print(f"Skipping outs[{i}] — unexpected shape {outs[i].shape}")
                    continue

            if outs[i].ndim == 2 and outs[i].shape[1] >= 4:
                outs[i][:, :4] /= rs[i]

    return outs


def postprocess(
    predictions,
    num_classes,
    conf_thre=0.60,
    nms_thre=0.45,
    class_agnostic=False,
    obj_conf_enabled=True,
):
    predictions = predictions[0]  # assume batch size 1
    predictions = np.array(predictions)

    # Convert xywh to xyxy
    boxes = predictions[:, :4].copy()
    boxes[:, 0] = predictions[:, 0] - predictions[:, 2] / 2
    boxes[:, 1] = predictions[:, 1] - predictions[:, 3] / 2
    boxes[:, 2] = predictions[:, 0] + predictions[:, 2] / 2
    boxes[:, 3] = predictions[:, 1] + predictions[:, 3] / 2
    predictions[:, :4] = boxes

    if not obj_conf_enabled:
        predictions[:, 4] = 1.0  # force obj_conf to 1

    class_scores = predictions[:, 5 : 5 + num_classes]
    class_ids = np.argmax(class_scores, axis=1)
    class_confs = class_scores[np.arange(len(class_scores)), class_ids]

    scores = predictions[:, 4] * class_confs
    keep = scores > conf_thre
    predictions = predictions[keep]
    scores = scores[keep]
    class_ids = class_ids[keep]

    if predictions.shape[0] == 0:
        return []

    output = []
    for cls in np.unique(class_ids):
        cls_mask = class_ids == cls
        cls_boxes = predictions[cls_mask, :4]
        cls_scores = scores[cls_mask]

        keep = nms_numpy(cls_boxes, cls_scores, nms_thre)
        selected = predictions[cls_mask][keep]
        selected = np.concatenate(
            [selected, class_ids[cls_mask][keep].reshape(-1, 1)], axis=1
        )
        output.append(selected)

    return np.vstack(output) if output else []


def bboxes_iou(bboxes_a, bboxes_b, xyxy=True):
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    if xyxy:
        tl = np.maximum(bboxes_a[:, None, :2], bboxes_b[:, :2])
        br = np.minimum(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        area_a = np.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], axis=1)
        area_b = np.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], axis=1)
    else:
        tl = np.maximum(
            bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2,
            bboxes_b[:, :2] - bboxes_b[:, 2:] / 2,
        )
        br = np.minimum(
            bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2,
            bboxes_b[:, :2] + bboxes_b[:, 2:] / 2,
        )
        area_a = np.prod(bboxes_a[:, 2:], axis=1)
        area_b = np.prod(bboxes_b[:, 2:], axis=1)

    en = np.prod((tl < br).astype(np.float32), axis=2)
    area_i = np.prod(br - tl, axis=2) * en
    return area_i / (area_a[:, None] + area_b - area_i + 1e-12)


def matrix_iou(a, b):
    lt = np.maximum(a[:, None, :2], b[:, :2])
    rb = np.minimum(a[:, None, 2:], b[:, 2:])
    area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return area_i / (area_a[:, None] + area_b - area_i + 1e-12)


def adjust_box_anns(bbox, scale_ratio, padw, padh, w_max, h_max):
    bbox[:, 0::2] = np.clip(bbox[:, 0::2] * scale_ratio + padw, 0, w_max)
    bbox[:, 1::2] = np.clip(bbox[:, 1::2] * scale_ratio + padh, 0, h_max)
    return bbox


def xyxy2xywh(bboxes):
    bboxes = np.copy(bboxes)
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    return bboxes


def xyxy2cxcywh(bboxes):
    bboxes = np.copy(bboxes)
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    bboxes[:, 0] = bboxes[:, 0] + bboxes[:, 2] * 0.5
    bboxes[:, 1] = bboxes[:, 1] + bboxes[:, 3] * 0.5
    return bboxes
