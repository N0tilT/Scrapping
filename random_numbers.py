import csv
import random
from pathlib import Path
import shutil

def create_random_dataset():
    base_dir = Path("dataset")
    new_dir = Path("dataset_random")
    new_dir.mkdir(exist_ok=True)
    used_numbers = set()
    
    with open("annotation_random.csv", "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Absolute Path", "Relative Path", "Class"])
        
        for class_name in ["good", "bad"]:
            class_dir = base_dir / class_name
            for file_path in class_dir.glob("*.txt"):
                while True:
                    num = random.randint(0, 10000)
                    if num not in used_numbers:
                        used_numbers.add(num)
                        break
                
                new_name = f"{num:05d}.txt"
                new_path = new_dir / new_name
                shutil.copy2(file_path, new_path)
                
                abs_path = new_path.resolve()
                rel_path = new_path
                writer.writerow([abs_path, rel_path, class_name])

if __name__ == "__main__":
    create_random_dataset()