import os
import random
import re
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


def get_random_word_from_specified_list(listname: str, list_dict: dict) -> Optional[str]:
    if listname in list_dict:
        return random.choice(list_dict[listname])
    else:
        return None


def get_words_inside_brackets(prompt: str) -> str:
    return re.findall('<.*?>', prompt)


def replace_words_inside_brackets_with_randomizer(prompt: str, list_dict: dict) -> str:
    if not list_dict or list_dict == {}:
        return prompt

    for _ in range(2):  # run this twice as a cheap way to deal with single recursive lists, TODO: do this better
        identified_words = get_words_inside_brackets(prompt)
        for word in identified_words:
            replacement = get_random_word_from_specified_list(word, list_dict)
            if replacement:
                prompt = prompt.replace(word, replacement, 1)

    return prompt
