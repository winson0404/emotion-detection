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
emotion_model_path = r"output\Emotion_CustomNet\default\best_model.onnx"
provider = ['CPUExecutionProvider']
threshold = 0.5
conf = OmegaConf.load("configs/CustomNet/default.yaml")
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
    # convert to gray scale
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # image = torch.from_numpy(image)
    image = cv2.resize(frame, (conf.dataset.image_size[0], conf.dataset.image_size[1]))
    image = image.transpose((2, 0, 1))
    image = np.expand_dims(image, axis=0)
    image = np.ascontiguousarray(image)
    image = image.astype(np.float32)
    # convefrt to tensor
    # normalize
    # convert to numpy
    return image

def compute_softmax(x):
    
    # image = torch.from_numpy(image)
    # image = image.float()
    # image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)
    # image = image.numpy()
    # convert to tensor
    #normalize like the training
    # convert to numpy    
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

if __name__ == "__main__":
    
    camera_name = "Emotion Detection"
    cap = cv2.VideoCapture(0)
    face_inf_session = ort.InferenceSession(face_model_path, providers=provider)
    # ort.tools.python.remove_initializer_from_input(face_inf_session)
    face_outname = [i.name for i in face_inf_session.get_outputs()]
    emotion_inf_session = ort.InferenceSession(emotion_model_path, providers=provider)
    emotion_out_name = [i.name for i in emotion_inf_session.get_outputs()]
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
                    out = emotion_inf_session.run(emotion_out_name, {'images': cropped_image})[0]
                    emotion = target_dict[out.argmax()]
                    #apply softmax to out
                    out = compute_softmax(out[0])
                    # print(out)
                    draw_image(frame, box, "", emotion)
                    cropped_frame= None
                # if score >= threshold:
                #     cropped_image = crop_roi_image(frame, bbox, (224, 224))
                #     out = emotion_inf_session.run(emotion_out_name, {'images': cropped_image/255})[0]
                #     emotion = target_dict[out.argmax()]
                #     draw_image(frame, bbox, score, emotion)
                cv2.imshow(camera_name, frame)
        except Exception as e:
            print(e)
        #     break
            
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        
