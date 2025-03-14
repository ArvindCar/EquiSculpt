import os
from tqdm import tqdm
import shutil

def clean_shapenet(root_dir):
    bad_files = []
    for root, _, files in tqdm(os.walk(root_dir)):
        for f in files:
            if f == 'model_normalized.obj':
                path = os.path.join(root, f)
                try:
                    with open(path, 'rb') as f:
                        content = f.read().decode('latin-1')
                    # Simple OBJ validity check
                    if 'v ' not in content or 'f ' not in content:
                        raise ValueError("Invalid OBJ structure")
                except Exception as e:
                    bad_files.append(path)
    
    print(f"Found {len(bad_files)} corrupted files")
    with open('bad_shapenet_files.txt', 'w') as f:
        f.write('\n'.join(bad_files))
    
    # # Remove bad files
    # for path in bad_files:
    #     os.remove(path)

def del_files(txt_file='bad_shapenet_files.txt'):
    with open(txt_file, 'r') as f:
        files_to_delete = f.read().splitlines()
    
    folders_to_delete = set(os.path.dirname(os.path.dirname(file_path)) for file_path in files_to_delete)
    
    for folder_path in folders_to_delete:
        try:
            shutil.rmtree(folder_path)
            print(f"Deleted folder: {folder_path}")
        except Exception as e:
            print(f"Error deleting folder {folder_path}: {e}")


if __name__ == '__main__':
    # clean_shapenet('./data/ShapeNetCore.v2')
    del_files('bad_shapenet_files.txt')