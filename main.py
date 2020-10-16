from facemaskcapture import FaceMaskCapture

facemaskdetect = FaceMaskCapture(
    camera_id=0, 
    xml_path="haarcascade_frontalface_default.xml",
    model_path="facemask_model.pth")

print("[+] Exit: Please Press ESC key...")
facemaskdetect.preview()
