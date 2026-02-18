import os
import json

def load_coco_annotations(json_path):
   import json
   with open(json_path) as f:
       coco = json.load(f)
   
   # Create image_id to filename mapping
   id_to_filename = {img['id']: img['file_name'] for img in coco['images']}
   
   # Group annotations by image_id
   annotations = {}
   for ann in coco['annotations']:
       img_id = ann['image_id']
       if img_id not in annotations:
           annotations[img_id] = []
           
       bbox = ann['bbox']  # [x,y,width,height]
       category_id = ann['category_id']
       annotations[img_id].append((category_id, bbox))
       
   return annotations, id_to_filename

def save_annotations(annotations, annotation_path):
    with open(annotation_path, 'w') as file:
        for ann in annotations:
            file.write(' '.join(map(str, ann)) + '\n')

