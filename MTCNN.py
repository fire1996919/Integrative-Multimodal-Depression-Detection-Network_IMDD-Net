import os
import cv2
from tqdm import tqdm
from mtcnn import MTCNN



def get_files(path):
    file_info = os.walk(path)
    file_list = []
    for r, d, f in file_info:
        file_list += [os.path.join(r, file) for file in f if file.endswith('.png')]  
    return file_list



def get_dirs(path):
    file_info = os.walk(path)
    dirs = []
    for d, r, f in file_info:
        dirs.append(d)
    return dirs[1:]



def process_faces_in_images(source_folder, target_folder):
    print(f"Processing images in {source_folder}...")
    detector = MTCNN()

    
    dirs = get_dirs(source_folder)

   
    for subfolder in tqdm(dirs):
        
        files = get_files(subfolder)
       
        target_subfolder = subfolder.replace(source_folder, target_folder)
        os.makedirs(target_subfolder, exist_ok=True)

       
        for file_path in files:
           
            img = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)
          
            faces = detector.detect_faces(img)

         
            if len(faces) > 0:
                x, y, width, height = faces[0]['box']
                confidence = faces[0]['confidence']

          
                face_img = img[y:y + height, x:x + width]
                face_img = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)

                target_file_path = os.path.join(target_subfolder, os.path.basename(file_path))
                cv2.imwrite(target_file_path, face_img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                print(f"Processed {file_path} with confidence {confidence}")



if __name__ == '__main__':
    source_folder = r'' 
    target_folder = r''  
    os.makedirs(target_folder, exist_ok=True)

    
    process_faces_in_images(source_folder, target_folder)
