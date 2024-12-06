import os
import cv2
from tqdm import tqdm
from mtcnn import MTCNN


# 获取目录下的所有文件
def get_files(path):
    file_info = os.walk(path)
    file_list = []
    for r, d, f in file_info:
        file_list += [os.path.join(r, file) for file in f if file.endswith('.png')]  # 只处理png文件
    return file_list


# 获取目录下的所有子文件夹
def get_dirs(path):
    file_info = os.walk(path)
    dirs = []
    for d, r, f in file_info:
        dirs.append(d)
    return dirs[1:]


# 处理图片并保存人脸图像
def process_faces_in_images(source_folder, target_folder):
    print(f"Processing images in {source_folder}...")
    detector = MTCNN()

    # 获取源文件夹下所有子文件夹
    dirs = get_dirs(source_folder)

    # 遍历每个子文件夹
    for subfolder in tqdm(dirs):
        # 获取当前子文件夹中的所有png文件
        files = get_files(subfolder)
        # 创建目标子文件夹
        target_subfolder = subfolder.replace(source_folder, target_folder)
        os.makedirs(target_subfolder, exist_ok=True)

        # 遍历并处理每个png文件
        for file_path in files:
            # 读取图片
            img = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)
            # 检测人脸
            faces = detector.detect_faces(img)

            # 如果检测到人脸，裁剪并保存
            if len(faces) > 0:
                x, y, width, height = faces[0]['box']
                confidence = faces[0]['confidence']

                # 裁剪并转换颜色通道回BGR（opencv默认是BGR）
                face_img = img[y:y + height, x:x + width]
                face_img = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)

                # 保存裁剪的人脸图片到目标文件夹
                target_file_path = os.path.join(target_subfolder, os.path.basename(file_path))
                cv2.imwrite(target_file_path, face_img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                print(f"Processed {file_path} with confidence {confidence}")


# 主函数调用
if __name__ == '__main__':
    source_folder = r'D:\Eye\traditional\processed_data\SELECT\NEEDMTCNNN'  # 修改为你的源文件夹路径
    target_folder = r'D:\Eye\traditional\processed_data\SELECT\MTCNNNEW'  # 修改为你要保存的文件夹路径
    os.makedirs(target_folder, exist_ok=True)

    # 调用函数处理人脸
    process_faces_in_images(source_folder, target_folder)
