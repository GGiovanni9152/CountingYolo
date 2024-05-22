import json
import os
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
classes = model.names

class Counter:
    def count(results):
        global classes
        for result in results:
            class_ids = [0] * len(model.names)
            for data in result.boxes.data.tolist():
                class_id = data[6]
                class_id = int(class_id)
                class_ids[class_id] += 1
            return class_ids


class Outputer: #Strategy pattern
    def output_data(self, results, picture_name):
        pass
    
    def output_data_filename(self, results, filename):
        pass


class JsonOutputer(Outputer):
    def output_data(self, data, picture_name):
        file_name = os.path.splitext(picture_name)[0] + "_result.json"
        data = self.preprocess_data(data)
        with open(file_name, 'w') as file: #Changed from x mode
                json.dump(data, file)
    def output_data_filename(self, data, filename):
        data = self.preprocess_data(data)
        with open(filename, 'w') as file:
            json.dump(data, file)
    
    @staticmethod
    def preprocess_data(data):
        results = dict()
        for class_name in range(len(data)):
            if (data[class_name] != 0):
                results[classes[class_name]] = data[class_name]
        return results



class TxtOutputer(Outputer):
    def output_data(self, data, picture_name):
        file_name = os.path.splitext(picture_name)[0] + "_result.txt"
        with open(file_name, 'w') as file:
            for class_name in range(len(data)):
                if (data[class_name] != 0):
                    output = str(classes[class_name]) + ": " + str(data[class_name]) + "\n"
                    file.write(output)
    def output_data_filename(self, data, filename):
        with open(filename, 'w') as file:
             for class_name in range(len(data)):
                if (data[class_name] != 0):
                    output = str(classes[class_name]) + ": " + str(data[class_name]) + "\n"
                    file.write(output)


class ConsoleOutputer(Outputer):
    def output_data(self, data, picture_name = None):
        print("Class : count")
        for class_name in range(len(data)):
            if (data[class_name] != 0):
                print(classes[class_name], ':', data[class_name]) 


class DataOutputer:
    strategy = None

    def __init__(self, strategy : Outputer):
        self.strategy = strategy
    
    def set_strategy(self, strategy : Outputer):
        self.strategy = strategy
    
    def output_data(self, data, picture_name):
        self.strategy.output_data(data, picture_name)
    
    def output_data_filename(self, data, filename):
        self.strategy.output_data_filename(data, filename)

#Сделать поле result
#if results == None: count else: just output
#выиграю время на отсутствии пересчёта для той же пикчи, но возмодны баги
class Picture_Processor:
    picture = None
    data_outputer = DataOutputer(JsonOutputer())

    def __init__(self, picture):
        self.picture = picture

    def set_picture(self, picture):
        self.picture = picture
    
    def get_picture(self):
        return self.picture
    
    def json_mode(self):
        self.data_outputer.set_strategy(JsonOutputer())

    def txt_mode(self):
        self.data_outputer.set_strategy(TxtOutputer())

    def console_mode(self):
        self.data_outputer.set_strategy(ConsoleOutputer())
    
    def process_picture(self, filename = None):
        global model
        results = model.track(self.picture, device = 0)
        classes_nums = Counter.count(results)

        if (filename == None):
            self.data_outputer.output_data(classes_nums, self.picture)
        
        else:
            if (type(self.data_outputer.strategy).__name__ == "ConsoleOutputer"):
                print("Console outputer does not have file output!")
                print("Please, change the output format")
            else:
                self.data_outputer.output_data_filename(classes_nums, filename)

#Tests
Processor = Picture_Processor("Pic.jpg")
Processor.console_mode()
Processor.process_picture()
Processor.process_picture("Console.txt")
