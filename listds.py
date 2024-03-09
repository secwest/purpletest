from huggingface_hub import list_datasets

def list_all_datasets():
    # Get the list of all datasets and their information
    datasets_info = list_datasets()
    
    # Print the names of all datasets
    for dataset_info in datasets_info:
        print(dataset_info.id)

if __name__ == "__main__":
    list_all_datasets()
