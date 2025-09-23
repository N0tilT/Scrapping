from dataset_iterator import DatasetIterator

if __name__ == "__main__":
    for item in DatasetIterator("annotation.csv", "bad"):
        print(item)