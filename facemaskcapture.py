import cv2
import torch
from model import FaceMaskNet

class FaceMaskCapture():
    
    def __init__(self, camera_id, xml_path, model_path):
        self.video = cv2.VideoCapture(camera_id)
        self.detect = cv2.CascadeClassifier(xml_path)   
        self.model = torch.load(model_path)
        self.label = ["Mask", "No Mask"]
        
    def read(self):
        while True:
            rt, frame = self.video.read()
            yield frame
        
    def detect_face(self, frame):
        result = self.detect.detectMultiScale(frame)
        return result

    def detect_mask(self, frame, use_gpu = False):
        inputs = None
        
        if use_gpu:
            self.model = self.model.cuda()
            inputs = torch.from_numpy(frame).unsqueeze(0).permute(0,3,1,2).float().cuda()
        else:
            self.model = self.model.cpu()
            inputs = torch.from_numpy(frame).unsqueeze(0).permute(0,3,1,2).float().cpu()
            
        output = self.model(inputs)
        print(output)
        result = torch.argmax(output, 1).item()
        return self.label[result]
            
    def preview(self):
        for frame in self.read():
            faces = self.detect_face(frame)
            for (x, y, w, h) in faces:
                face_frame = frame[x:x+w, y:y+h]
                face_frame = cv2.resize(face_frame, dsize=(48, 48))
                face_frame = face_frame / 255
                label_name = self.detect_mask(face_frame, use_gpu=True)
                frame = cv2.rectangle(frame, pt1=(x, y), pt2=(x+w, y+h), color=(0, 255, 0))
                frame = cv2.putText(frame, label_name, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.imshow("FaceMaskDetection", frame)
            if cv2.waitKey(1) == 27:
                cv2.destroyAllWindows()
                self.video.release()
                break
    
    
        
        
        
