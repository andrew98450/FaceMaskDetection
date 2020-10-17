import cv2
import torch
from model import FaceMaskNet

class FaceMaskCapture():
    
    def __init__(self, camera_id, xml_path, model_path, use_gpu):
        self.video = cv2.VideoCapture(camera_id)
        self.detect = cv2.CascadeClassifier(xml_path)   
        self.use_gpu = use_gpu
        if self.use_gpu:
            self.model = torch.load(model_path).cuda().eval()
        else:
            self.model = torch.load(model_path).cpu().eval()

        self.label = ["No Mask", "Mask"]
        
    def read(self):
        while True:
            rt, frame = self.video.read()
            yield frame
        
    def detect_face(self, frame):
        result = self.detect.detectMultiScale(frame, scaleFactor=1.1)
        return result

    def detect_mask(self, frame):
        inputs = torch.from_numpy(frame).unsqueeze(0).permute(0,3,1,2).float()
        
        if self.use_gpu:
            inputs = inputs.cuda()
        else:
            inputs = inputs.cpu()
            
        output = self.model(inputs)
        result = torch.argmax(output, 1).item()
        return self.label[result]
            
    def preview(self):
        for frame in self.read():
            faces = self.detect_face(frame)
            for (x, y, w, h) in faces:
                face_frame = frame[y:y+h, x:x+w]
                face_frame = cv2.resize(face_frame, dsize=(32, 32))
                face_frame = face_frame / 255
                label_name = self.detect_mask(face_frame)
                frame = cv2.rectangle(frame, pt1=(x, y), pt2=(x+w, y+h), color=(0, 255, 0), thickness=2)
                frame = cv2.putText(frame, label_name, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("FaceMaskDetection", frame)
            if cv2.waitKey(1) == 27:
                cv2.destroyAllWindows()
                self.video.release()
                break
    
    
        
        
        
