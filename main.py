import torch
import cv2
import onnxruntime as ort
import os
import numpy as np
from utils.util import full_frame_preprocess, full_frame_postprocess, crop_roi_image, draw_image, scale
from utils.box_utils import predict
from omegaconf import OmegaConf
import torchvision.transforms as transforms

face_model_path = "data/model_weights/version-RFB-640.onnx"
emotion_model_path = r"output\MobileNetV3_small\default\model.onnx"
provider = ['CPUExecutionProvider']
threshold = 0.5
conf = OmegaConf.load("configs/MobileNetV3/default.yaml")
target_dict = {i: label for i, label in enumerate(conf.dataset.targets)}

def hand_preprocess(frame, auto=False)->np.ndarray:
    image = cv2.resize(frame, (640, 480))
    image_mean = np.array([127, 127, 127])
    image = (image - image_mean) / 128
    image = np.transpose(image, [2, 0, 1])
    image = np.expand_dims(image, axis=0)
    image = image.astype(np.float32)
    return image

def emotion_preprocess(frame, auto=False)->np.ndarray:
    image = frame.astype(np.float32)
    image = cv2.resize(frame, (224, 224))
    image = image.transpose((2, 0, 1))
    image = np.expand_dims(image, axis=0)
    image = np.ascontiguousarray(image)
    # convert to tensor
    image = torch.from_numpy(image)
    image = image.float()
    #normalize like the training
    image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)
    # convert to numpy
    image = image.numpy()
    return image

if __name__ == "__main__":
    
    camera_name = "Emotion Detection"
    cap = cv2.VideoCapture(-1)
    face_inf_session = ort.InferenceSession(face_model_path, providers=provider)
    face_outname = [i.name for i in face_inf_session.get_outputs()]
    gesture_inf_session = ort.InferenceSession(emotion_model_path, providers=provider)
    gesture_out_name = [i.name for i in gesture_inf_session.get_outputs()]
    got_roi = False
    cropped_frame = None
    while cap.isOpened():
        ret, frame = cap.read()
        
        try:
            if ret:
                image = hand_preprocess(frame, auto=False)
                # breakpoint()
                confidences, bboxes = face_inf_session.run(None, {face_inf_session.get_inputs()[0].name: image})
                # bbox, score = full_frame_postprocess(out, ratio, dwdh, threshold)
                boxes, labels, probs = predict(frame.shape[1], frame.shape[0], confidences, bboxes, threshold)
                for i in range(boxes.shape[0]):
                    box = scale(boxes[i, :])
                    cropped_frame = frame[box[1]:box[3], box[0]:box[2]].copy()
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                    got_roi = True
                if cropped_frame is not None:
                    cropped_image = emotion_preprocess(cropped_frame)
                    out = gesture_inf_session.run(gesture_out_name, {'images': cropped_image})[0]
                    gesture = target_dict[out.argmax()]
                    draw_image(frame, box, max(out), gesture)
                # if score >= threshold:
                #     cropped_image = crop_roi_image(frame, bbox, (224, 224))
                #     out = gesture_inf_session.run(gesture_out_name, {'images': cropped_image/255})[0]
                #     gesture = target_dict[out.argmax()]
                #     draw_image(frame, bbox, score, gesture)
                cv2.imshow(camera_name, frame)
        except Exception as e:
            print(e)
        #     break
            
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        
