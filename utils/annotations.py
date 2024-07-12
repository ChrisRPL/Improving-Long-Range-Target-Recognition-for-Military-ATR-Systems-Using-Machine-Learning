import os

def load_yolo_annotations(annotation_path):
    annotations = []
    with open(annotation_path, 'r') as file:
        for line in file:
            data = line.strip().split()
            if len(data) >= 5:
                class_id = int(data[0])
                x_center = float(data[1])
                y_center = float(data[2])
                width = float(data[3])
                height = float(data[4])
                annotations.append((class_id, x_center, y_center, width, height))
    return annotations

def save_annotations(annotations, annotation_path):
    with open(annotation_path, 'w') as file:
        for ann in annotations:
            file.write(' '.join(map(str, ann)) + '\n')

