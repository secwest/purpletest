import argparse
from datasets import load_dataset, get_dataset_config_names

def select_option(options, prompt, allow_quit=False, allow_all=False):
    for index, option in enumerate(options, start=1):
        print(f"{index}. {option}")
    if allow_all:
        print(f"{len(options) + 1}. Iterate over all options")
    while True:
        choice = input(prompt)
        if allow_quit and choice.lower() == 'q':
            return 'q'  # Quit signal
        if allow_all and choice == str(len(options) + 1):
            return 'all'  # Iterate over all options
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return options[int(choice)-1]
        else:
            print("Invalid selection. Please enter a number from the list, 'q' to go back, or select 'all'.")


PROMPT_TEMPLATES = {
    'glue': {
        'cola': "Sentence: {sentence}\nLabel (0 or 1 for unacceptable or acceptable):",
        'sst2': "Sentence: {sentence}\nSentiment (0 or 1 for negative or positive):",
        'mrpc': "Sentence 1: {sentence1}\nSentence 2: {sentence2}\nParaphrase (0 or 1):",
        'qqp': "Sentence 1: {sentence1}\nSentence 2: {sentence2}\nParaphrase (0 or 1):",
        'qnli': "Question: {question}\nSentence: {sentence}\nEntailment (0 or 1):",
        'rte': "Question: {question}\nSentence: {sentence}\nEntailment (0 or 1):",
        'wnli': "Sentence 1: {sentence1}\nSentence 2: {sentence2}\nCoreference (0 or 1):",
        'mnli': "Premise: {premise}\nHypothesis: {hypothesis}\nLabel (0, 1, or 2 for entailment, neutral, contradiction):",
        'stsb': "Sentence 1: {sentence1}\nSentence 2: {sentence2}\nSimilarity score (0.0 to 5.0):",
    },
    'super_glue': {
        'boolq': "Passage: {passage}\nQuestion: {question}\nAnswer (True or False):",
        'cb': "Premise: {premise}\nHypothesis: {hypothesis}\nLabel (entailment, contradiction, neutral):",
        'copa': "Premise: {premise}\nChoice1: {choice1}\nChoice2: {choice2}\nCause/Effect (Select 1 or 2):",
        'multirc': "Paragraph: {paragraph}\nQuestion: {question}\nAnswer: {answer}\nCorrect (True or False):",
        'wic': "Sentence 1: {sentence1}\nSentence 2: {sentence2}\nWord: {word}\nSame meaning (True or False):",
        'wsc': "Text: {text}\nQuestion: Does the pronoun in the text refer to the correct entity (True or False)?",
    },
    'squad': {
        'default': "Context: {context}\nQuestion: {question}\nAnswer:",
    },
    'conll2003': {
        'default': "Tokens: {<tokens>}\nNamed Entities:",
    },
    'snli': {
        'default': "Premise: {premise}\nHypothesis: {hypothesis}\nLabel (entailment, neutral, contradiction):",
    },
    'trec': {
        'default': "Question: {text}\nType:",
    },
    'rocstories': {
        'default': "Story: {story}\nWhat happens next?",
    },
    'multi_nli': {
        'default': "Premise: {premise}\nHypothesis: {hypothesis}\nLabel (entailment, neutral, contradiction):",
    },
    'xnli': {
        'default': "Sentence 1: {sentence1}\nSentence 2: {sentence2}\nLabel (entailment, neutral, contradiction):",
    },
    'cnn_dailymail': {
        'default': "Article: {article}\nWrite a summary:",
    },
    'wmt': {
        'default': "Source text: {source_sentence}\nTranslate to target language:",
    },
    # Additional datasets/tasks can be added here...
}


def generate_prompt_text(entry, dataset_name, config_name):
    template = PROMPT_TEMPLATES.get(dataset_name, {}).get(config_name, "")
    
    # Initialize prompt_text with a default value or the template itself
    prompt_text = "Prompt generation for this dataset/task/config is not yet supported. Please customize."
    
    try:
        # Attempt to format the template with the entry
        prompt_text = template.format(**entry)
    except KeyError as missing_key:
        # Handle the missing field
        missing_key = str(missing_key).strip("'")
        available_keys = ', '.join(entry.keys())
        error_msg = f"Missing required field: {missing_key}. Available fields in this entry: {available_keys}."
        prompt_text = f"Error generating prompt: {error_msg}"
    
    return prompt_text


def print_dataset_entries(dataset, dataset_name, config_name, paginate=100):
    for i, entry in enumerate(dataset):
        prompt_text = generate_prompt_text(entry, dataset_name, config_name)
        print(f"Entry {i}: {prompt_text}")
        if paginate and (i + 1) % paginate == 0:
            response = input("Press Enter to continue or 'q' to go back: ")
            if response.lower() == 'q':
                break  # Exit loop

def browse_dataset(args, configs, splits):
    for config_name in configs:
        for split_name in splits:
            print(f"\nProcessing configuration: {config_name}, Split: {split_name}")
            split_dataset = load_dataset(args.dataset_name, config_name, split=split_name)
            print_dataset_entries(split_dataset, args.dataset_name, config_name, paginate=args.paginate)
            print("\nCompleted processing this configuration and split.")

def check_template_fields(dataset, dataset_name, config_name):
    """
    Checks if the fields required by the template exist in the dataset entries
    and returns available fields for diagnostic purposes.
    """
    template = PROMPT_TEMPLATES.get(dataset_name, {}).get(config_name, "")
    # Extract field names from the template
    field_names = set(re.findall(r"\{([^{}<>]+)\}", template))
    
    # Extract field names that require join operations
    join_field_names = set(re.findall(r"\{<([^{}<>]+)>\}", template))
    
    # Combine all required field names
    all_required_fields = field_names.union(join_field_names)
    
    # Fetch a sample entry from the dataset
    sample_entry = next(iter(dataset), {})
    
    # Determine missing and available fields in the sample entry
    missing_fields = [field for field in all_required_fields if field not in sample_entry]
    available_fields = list(sample_entry.keys())
    
    if missing_fields:
        return False, missing_fields, available_fields
    return True, [], available_fields

def main():
    parser = argparse.ArgumentParser(description="Interactively browse a Hugging Face dataset with optional pagination, and output full prompt text for questions.")
    parser.add_argument("dataset_name", help="The name of the dataset to browse.")
    parser.add_argument("-np", "--no-pagination", action="store_true", help="Disable pagination. If not set, pagination defaults to 100 entries.")
    args = parser.parse_args()

    paginate = 0 if args.no_pagination else 100

    while True:
        configs = get_dataset_config_names(args.dataset_name)
        if not configs:
            print("No configurations found for this dataset.")
            break

        print("Available configurations include:")
        for index, config in enumerate(configs, start=1):
            print(f"{index}. {config}")
        print(f"{len(configs)+1}. Do all")
        
        choice = input("Select a configuration by number, 'q' to quit, or select 'Do all': ")
        
        if choice.lower() == 'q':
            break
        
        try:
            choice = int(choice)
            assert 1 <= choice <= len(configs) + 1
        except (ValueError, AssertionError):
            print("Invalid selection. Please enter a valid number or 'q' to quit.")
            continue

        if choice <= len(configs):
            config_name = configs[choice-1]
            dataset = load_dataset(args.dataset_name, config_name, split=None)


  	        # Perform the template field check
            fields_ok, missing_fields, available_fields = check_template_fields(dataset, args.dataset_name, config_name)
            if not fields_ok:
                print(f"Error: The selected dataset configuration '{config_name}' does not contain the required fields for its template. Missing fields: {', '.join(missing_fields)}.")
                print(f"Available fields in the dataset: {', '.join(available_fields)}")
                return  # Exit or handle the error as needed
    

            splits = list(dataset.keys())
            for split in splits:
                split_dataset = dataset[split]
                print(f"Browsing '{args.dataset_name}' dataset with configuration '{config_name}', split '{split}':")
                print_dataset_entries(split_dataset, args.dataset_name, config_name, paginate)
        elif choice == len(configs) + 1:
            # Handle 'Do all' option
            for config in configs:
                dataset = load_dataset(args.dataset_name, config, split=None)
                splits = list(dataset.keys())
                for split in splits:
                    split_dataset = dataset[split]
                    print(f"\nBrowsing '{args.dataset_name}' dataset with configuration '{config}', split '{split}':")
                    print_dataset_entries(split_dataset, args.dataset_name, config, paginate)
            break  # Exit after doing all

if __name__ == "__main__":
    main()
