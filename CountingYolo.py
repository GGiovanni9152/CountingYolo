from ultralytics import YOLO, solutions

model = YOLO('yolov8n.pt')
classes = model.names

def Class_counting(results):
    global classes
    for result in results:
        class_ids = [0] * len(model.names)
        for data in result.boxes.data.tolist():
            class_id = data[6]
            class_id = int(class_id)
            class_ids[class_id] += 1
        print("Class : count")
        for class_name in range(len(class_ids)):
            if (class_ids[class_name] != 0):
                print(classes[class_name], ':', class_ids[class_name]) 

def Process_Picture(pathname):
    global model
    results = model.track(pathname, device = 0)
    Class_counting(results)

Process_Picture("Pic.jpg")
