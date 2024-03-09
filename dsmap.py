import argparse
from datasets import load_dataset, get_dataset_config_names

def print_fields_and_examples(dataset_name, config_name, split):
    # Load the dataset for the specified configuration and split with updated parameter
    dataset = load_dataset(dataset_name, config_name, split=split, verification_mode='no_checks')
    
    print(f"Configuration: {config_name}, Split: {split}")
    print("Fields and example values:")

    # Print the first five examples from the dataset
    for i, example in enumerate(dataset):
        if i >= 5:  # Limit to first five examples
            break
        print(f"Example {i + 1}:")
        for field, value in example.items():
            print(f"  {field}: {value}")
        print("--------------------------------------------------")

def recursively_process_all(dataset_name, configs):
    for config_name in configs:
        try:
            dataset = load_dataset(dataset_name, config_name, split=None, verification_mode='no_checks')
        except Exception as e:
            print(f"Failed to load dataset '{dataset_name}' with configuration '{config_name}': {e}")
            continue
        
        splits = list(dataset.keys()) if dataset else []
        
        for split in splits:
            # Print fields and examples for each split
            print_fields_and_examples(dataset_name, config_name, split)

def main():
    parser = argparse.ArgumentParser(description="Prints a recursive map of all fields in all configs and splits for a given dataset, with examples from each.")
    parser.add_argument("dataset_name", help="The name of the dataset to analyze.")
    args = parser.parse_args()

    configs = get_dataset_config_names(args.dataset_name)
    if not configs:
        print(f"No configurations found for the dataset '{args.dataset_name}'.")
        return

    recursively_process_all(args.dataset_name, configs)

if __name__ == "__main__":
    main()
