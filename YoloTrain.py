from ultralytics import YOLO
import matplotlib.pyplot as plt
import torch
import cv2

model = YOLO('yolov8n.yaml')

results = model.train(data = "config.yaml", epochs = 1, device = 0)





