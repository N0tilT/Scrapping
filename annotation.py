import csv
from pathlib import Path

def create_annotation():
    base_dir = Path("dataset")
    classes = ["good", "bad"]
    with open("annotation.csv", "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Absolute Path", "Relative Path", "Class"])
        
        for class_name in classes:
            class_dir = base_dir / class_name
            for file_path in class_dir.glob("*.txt"):
                abs_path = file_path.resolve()
                rel_path = file_path
                writer.writerow([abs_path, rel_path, class_name])

if __name__ == "__main__":
    create_annotation()