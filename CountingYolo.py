import json
import os
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
classes = model.names

"""def Class_counting(results):
    global classes
    for result in results:
        class_ids = [0] * len(model.names)
        for data in result.boxes.data.tolist():
            class_id = data[6]
            class_id = int(class_id)
            class_ids[class_id] += 1
        return class_ids
        print("Class : count")
        for class_name in range(len(class_ids)):
            if (class_ids[class_name] != 0):
                print(classes[class_name], ':', class_ids[class_name]) 
"""

"""def Result_output(results, picture_name, file_output = False, output_file_name = None, show = False, json_output = True):
    global classes
    if (file_output):
        if (json_output): #Вывод в json
            if (output_file_name == None): #Стандартное имя файла
                file_name = os.path.splitext(picture_name)[0] + "_result.json"
                with open(file_name, 'x') as file:
                    json.dump(results, file)
            else: #Специфическое имя файла
                with open(output_file_name, 'w') as file:
                    json.dump(results, file)
        else: #Вывод в обычный txt
            if (output_file_name == None):
                file_name = os.path.splitext(picture_name)[0] + "_result.txt"
                with open(file_name, 'x') as file:
                    for class_name in range(len(results)):
                        if (results[class_name] != 0):
                            output = str(classes[class_name]) + ": " + str(results[class_name]) + "\n"
                            file.write(output)
            else:
                with open(output_file_name, 'w') as file:
                    for class_name in range(len(results)):
                        if (results[class_name] != 0):
                            output = str(classes[class_name]) + ": " + str(results[class_name]) + "\n"
                            file.write(output)
    else:
        print("Class : count")
        for class_name in range(len(results)):
            if (results[class_name] != 0):
                print(classes[class_name], ':', results[class_name]) 
"""

#Можно через декоратор или просто разные названия функций

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
    
    def output_data_filename(self, results, picture_name, filename):
        pass


class JsonOutputer(Outputer):
    def output_data(self, results, picture_name):
        file_name = os.path.splitext(picture_name)[0] + "_result.json"
        with open(file_name, 'x') as file:
                json.dump(results, file)
   
    def output_data_filename(self, results, picture_name, filename):
        with open(filename, 'w') as file:
            json.dump(results, file)


class TxtOutputer(Outputer):
    def output_data(self, results, picture_name):
        file_name = os.path.splitext(picture_name)[0] + "_result.txt"
        with open(file_name, 'x') as file:
            for class_name in range(len(results)):
                if (results[class_name] != 0):
                    output = str(classes[class_name]) + ": " + str(results[class_name]) + "\n"
                    file.write(output)
    
    def output_data_filename(self, results, picture_name, filename):
        with open(filename, 'w') as file:
             for class_name in range(len(results)):
                if (results[class_name] != 0):
                    output = str(classes[class_name]) + ": " + str(results[class_name]) + "\n"
                    file.write(output)


class ConsoleOutputer(Outputer):
    def output_data(self, results, picture_name = None):
        print("Class : count")
        for class_name in range(len(results)):
            if (results[class_name] != 0):
                print(classes[class_name], ':', results[class_name]) 


class DataOutputer:
    strategy = None

    def __init__(self, strategy : Outputer):
        self.strategy = strategy
    
    def set_strategy(self, strategy : Outputer):
        self.strategy = strategy
    
    def output_data(self, data, picture_name):
        self.strategy.output_data(data, picture_name)
    
    def output_data_filename(self, data, picture_name, filename):
        self.strategy.output_data_filename(data, picture_name, filename)



def Process_Picture(pathname):
    global model
    results = model.track(pathname, device = 0)
    classes_nums = Counter.count(results)

    outputer = DataOutputer(JsonOutputer())
    outputer.output_data(classes_nums, pathname)
    outputer.output_data_filename(classes_nums, pathname, "Json_test.json")

    outputer.set_strategy(TxtOutputer())
    outputer.output_data(classes_nums, pathname)
    outputer.output_data_filename(classes_nums, pathname, "Txt_test.txt")
    
    
    #classes_nums = Class_counting(results)
    #Result_output(classes_nums, pathname)

Process_Picture("Pic.jpg")


#Доделать вывод в json через dict
#Доделать консольный вывод