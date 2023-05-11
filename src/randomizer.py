import os
import random
import re
from pathlib import Path
from typing import Optional


# generated from chatgpt, maybe try to get this to take nested brackets
def curly_bracket_randomiser(text: str) -> str:
    # Find all words in curly brackets
    matches = re.findall(r"\{(.+?)\}", text)
    for match in matches:
        # Split the words inside the curly brackets
        options = match.split("|")
        # Replace the match with a randomly chosen word
        text = text.replace("{" + match + "}", random.choice(options), 1)
    return text


def get_default_random_lists_from_folder(folderpath: str) -> dict:
    results = {}
    for root, _, files in os.walk(folderpath):
        for file in files:
            if file.endswith(".txt"):
                with open(os.path.join(root, file), encoding='utf-8') as listfile:
                    randomlist = [line.rstrip() for line in listfile]
                if not len(randomlist):
                    continue
                text_file = os.path.join(root, file).replace(
                    folderpath, '').strip(os.sep)
                text_file = text_file.replace(os.sep, "/")[:-4]
                results[f"<_{text_file}>"] = randomlist

    return results


def get_random_lists_from_folder(folderpath: str) -> dict:
    results = {}
    for root, _, files in os.walk(folderpath):
        for file in files:
            if file.endswith(".txt"):
                with open(os.path.join(root, file), encoding='utf-8') as listfile:
                    randomlist = [line.rstrip() for line in listfile]
                if not len(randomlist):
                    continue
                text_file = os.path.join(root, file).replace(
                    folderpath, '').strip(os.sep)
                text_file = text_file.replace(os.sep, "/")[:-4]
                results[f"<{text_file}>"] = randomlist

    return results

# thanks curio and gpt 4!
# Function to get a random word from the specified list, if it exists in the dictionary


def get_random_word_from_specified_list(listname: str, list_dict: dict) -> Optional[str]:
    # Check if the list exists in the dictionary, and return a random word if it does
    if f"{listname}" in list_dict:
        return random.choice(list_dict[listname])
    else:
        return None

# Function to find all words surrounded by angle brackets in a string


def get_words_inside_brackets(prompt: str) -> str:
    return re.findall('<.*?>', prompt)

# Function to replace words surrounded by angle brackets with random words from the corresponding list


def replace_words_inside_brackets_with_randomizer(prompt: str, list_dict: dict, max_depth: int, depth: int = 0) -> str:
    # Return the prompt without replacement if the maximum recursion depth has been reached
    if depth >= max_depth:
        return prompt

    # Get all words surrounded by angle brackets
    identified_words = get_words_inside_brackets(prompt)

    # Iterate over the words and replace them with random words from the corresponding lists
    for word in identified_words:
        replacement = get_random_word_from_specified_list(word, list_dict)
        if replacement:
            prompt = prompt.replace(word, replace_words_inside_brackets_with_randomizer(
                replacement, list_dict, max_depth, depth + 1), 1)

    return prompt
