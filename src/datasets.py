import os
import re
import pandas as pd

class GSM8K:
    
    def __init__(self, path, mode="train"):
        self.path = path
        
        # Checking if the folder exists
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Folder {self.path} does not exist.")
        
        # Checking if the folder contains train.json and test.json
        if not os.path.exists(os.path.join(self.path, "train.jsonl")):
            raise FileNotFoundError(f"Folder {self.path} does not contain train.jsonl.")
        
        if not os.path.exists(os.path.join(self.path, "test.jsonl")):
            raise FileNotFoundError(f"Folder {self.path} does not contain test.jsonl.")
        
        # Loading the json files
        self.train = pd.read_json(os.path.join(self.path, "train.jsonl"), lines=True)
        self.test = pd.read_json(os.path.join(self.path, "test.jsonl"), lines=True)
        
        self.mode = mode
        
    
    # @title Testing library
    @staticmethod
    def find_numbers(x: str) -> list[str]:
        """Finds all numbers in a string."""
        # Search for number, possibly negative (hyphen), with thousand separators
        # (comma), and with a decimal point (period inbetween digits).
        numbers = re.compile(
            r'-?[\d,]*\.?\d+',
            re.MULTILINE | re.DOTALL | re.IGNORECASE,
        ).findall(x)
        return numbers


    @staticmethod
    def find_number(x: str,
                    answer_delimiter: str = 'The answer is') -> str:
        """Finds the most relevant number in a string."""
        # If model uses the answer delimiter, then select the first number following
        # that format.
        if answer_delimiter in x:
            answer = x.split(answer_delimiter)[-1]
            numbers = GSM8K.find_numbers(answer)
            if numbers:
                return numbers[0]

        # In general, select the last number in the string.
        numbers = GSM8K.find_numbers(x)
        if numbers:
            return numbers[-1]
        return ''

    @staticmethod
    def maybe_remove_comma(x: str) -> str:
        # Example: 5,600 -> 5600
        return x.replace(',', '') 
    
    def get_score(self, answer: str, example_id: int):
        # Closer to 0 the better
        correct_answer = float(self[example_id]['answer'])
        answer_from_model = float(GSM8K.maybe_remove_comma(GSM8K.find_number(answer)))
        return abs(correct_answer - answer_from_model)
        
    def verify_answer(self, answer: str, example_id: int):
        # Closer to 0 the better
        correct_answer = float(self[example_id]['answer'])
        answer_from_model = float(GSM8K.maybe_remove_comma(GSM8K.find_number(answer)))
        return correct_answer == answer_from_model
    
    def __str__(self):
        # Returning a string with the name of the dataset and
        # the number of train and test examples 
        return f"GSM8K dataset ({self.mode}) with {len(self.train)} train examples and {len(self.test)} test examples.\n"
    
    def __getitem__(self, idx):
        example = self.train.iloc[idx] if self.mode == "train" else self.test.iloc[idx]
        return example
    
    def __len__(self):
        return len(self.train) if self.mode == "train" else len(self.test)