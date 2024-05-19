from ultralytics import YOLO, solutions
import matplotlib.pyplot as plt
import torch
import cv2

#device = 'cuda' if torch.cuda.is_available() else 'cpu'
#print(f'Using device: {device}')
#model = YOLO('yolov8n.pt', device = "gpu")

model = YOLO('yolov8n.pt')#.cuda()#.to(device)

#results = model.track('Loki.mp4', show = True, device = '0')


results = model.track('Pic.jpg', device = 0)
classes = model.names

for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    class_ids = [0] * len(model.names)
    for data in result.boxes.data.tolist():
        #print(data)
        class_id = data[6]
        class_id = int(class_id)
        class_ids[class_id] += 1
    print("Class : count")
    for class_name in range(len(class_ids)):
      print(classes[class_name], ':', class_ids[class_name]) 

    #print(class_ids)
    result.show()