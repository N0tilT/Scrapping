import csv
import shutil
from pathlib import Path

def reorganize_dataset():
    base_dir = Path("dataset")
    new_dir = Path("dataset_reorganized")
    new_dir.mkdir(exist_ok=True)
    
    with open("annotation_new.csv", "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Absolute Path", "Relative Path", "Class"])
        
        for class_name in ["good", "bad"]:
            class_dir = base_dir / class_name
            for i, file_path in enumerate(class_dir.glob("*.txt")):
                new_name = f"{class_name}_{i:04d}.txt"
                new_path = new_dir / new_name
                shutil.copy2(file_path, new_path)
                
                abs_path = new_path.resolve()
                rel_path = new_path
                writer.writerow([abs_path, rel_path, class_name])

if __name__ == "__main__":
    reorganize_dataset()