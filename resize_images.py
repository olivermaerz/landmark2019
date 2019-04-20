import os
import sys
from PIL import Image


def resize_image(path, file_name):
    file_w_path = os.path.join(path, file_name)
    # print("Resizing: "+file_w_path)
    old_image = Image.open(file_w_path)
    width, height = old_image.size
    factor = 300 / min(width, height) 
    new_image = old_image.resize((int(width * factor), int(height * factor)), Image.LANCZOS)
    new_image.save('simages/'+file_name[0]+"/"+file_name, "JPEG", quality=80, optimize=True, progressive=True)
    new_image.close()
    old_image.close()



new_dirs = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f']
for new_dir in new_dirs:
    print('creating new dir: '+new_dir)
    os.mkdir('simages/'+new_dir)

for path, sub_dirs, file_names in os.walk('images'):
    print("Resizing images in folder: "+path)
    for file_name in file_names:
        base_filename, file_extension = os.path.splitext(file_name)
        if file_extension.lower() == '.jpg':
            resize_image(path, file_name)
        else:
            print("found file with extension: "+file_extension.lower())
