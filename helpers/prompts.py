from pydantic import BaseModel
from typing import Union, List, Dict

class Prompts(BaseModel):
    """
    The Prompt class takes in two parameters, a positive prompt and an optional negative prompt.
    The parameters can be of type str, list of str, or dict with integer keys.
    This class provides methods to convert the inputs into str, list, or dict.
    """
    
    # These can be either str, List[str], or Dict[int, str]
    prompt: Union[str, List[str], Dict[int, str]]
    neg_prompt: Union[str, List[str], Dict[int, str]] = ""  # Optional and default to empty string

    def as_string(self):
        """ Convert the prompt and neg_prompt into a string. """
        cond = self._to_string(self.prompt)
        uncond = self._to_string(self.neg_prompt)
        return cond, uncond

    def as_list(self):
        """ Convert the prompt and neg_prompt into a list. If neg_prompt is shorter, pad it with empty string.
            If neg_prompt is longer, cut it off to match the length of prompt.
        """
        cond = self._to_list(self.prompt)
        uncond = self._to_list(self.neg_prompt, len(cond))
        if len(uncond) > len(cond):
            uncond = uncond[:len(cond)]
        return cond, uncond

    def as_dict(self):
        """ Convert the prompt and neg_prompt into a dict. """
        cond = self._to_dict(self.prompt)
        uncond = self._to_dict(self.neg_prompt, list(cond.keys()))
        return cond, uncond

    def _to_string(self, prompt):
        """ Helper method to convert prompt into a string. """
        if isinstance(prompt, str):
            return prompt
        elif isinstance(prompt, list):
            return ', '.join(prompt)
        elif isinstance(prompt, dict):
            return ', '.join(prompt.values())
        else:
            return ''

    def _to_list(self, prompt, length=1):
        """ Helper method to convert prompt into a list. """
        if isinstance(prompt, str):
            return [prompt]*length
        elif isinstance(prompt, list):
            return prompt if prompt else [""] * length
        elif isinstance(prompt, dict):
            return list(prompt.values()) if prompt else [""] * length
        else:
            return [""] * length

    def _to_dict(self, prompt, keys=[0]):
        """ Helper method to convert prompt into a dict. """
        if isinstance(prompt, str):
            return {keys[0]: prompt}
        elif isinstance(prompt, list):
            return dict(zip(range(len(prompt)), prompt)) if prompt else {0: ""}
        elif isinstance(prompt, dict):
            return prompt if prompt else {0: ""}
        else:
            return dict(zip(keys, [""] * len(keys)))

"""
# Only run test cases when this script is running directly
if __name__ == '__main__':
    # Revised example 1: both prompt and neg_prompt are dictionaries with integer keys
    prompt1 = Prompt(prompt={0:"cat in space", 20: "cat sushi"}, neg_prompt={0:"dogs", 1:"pugs"})
    assert prompt1.as_list() == (["cat in space", "cat sushi"], ["dogs", "pugs"])
    assert prompt1.as_string() == ("cat in space, cat sushi", "dogs, pugs")
    assert prompt1.as_dict() == ({0: "cat in space", 20: "cat sushi"}, {0: "dogs", 1: "pugs"})

    # example 2: prompt is a dictionary and neg_prompt is a string
    prompt2 = Prompt(prompt={0:"cat in space", 20: "cat sushi"}, neg_prompt="dogs")
    assert prompt2.as_list() == (["cat in space", "cat sushi"], ["dogs", "dogs"])
    assert prompt2.as_string() == ("cat in space, cat sushi", "dogs")
    assert prompt2.as_dict() == ({0: "cat in space", 20: "cat sushi"}, {0: "dogs"})

    # example 3: prompt is a list and neg_prompt is a string
    prompt3 = Prompt(prompt=["cat in space", "cat sushi"], neg_prompt="dogs")
    assert prompt3.as_list() == (["cat in space", "cat sushi"], ["dogs", "dogs"])
    assert prompt3.as_string() == ("cat in space, cat sushi", "dogs")
    assert prompt3.as_dict() == ({0: "cat in space", 1: "cat sushi"}, {0: "dogs"})

    # example 4: prompt is a string and neg_prompt is a list
    prompt4 = Prompt(prompt="cat in space", neg_prompt=["dogs", "pugs"])
    assert prompt4.as_list() == (["cat in space"], ["dogs"])
    assert prompt4.as_string() == ('cat in space', 'dogs, pugs')
    assert prompt4.as_dict() == ({0: 'cat in space'}, {0: 'dogs', 1: 'pugs'})

    # example 5: prompt is a string and neg_prompt is not provided
    prompt5 = Prompt(prompt="cat in space")
    assert prompt5.as_list() == (["cat in space"], [""])
    assert prompt5.as_string() == ("cat in space", "")
    assert prompt5.as_dict() == ({0: "cat in space"}, {0: ""})

    # example 6: prompt is a list and neg_prompt is not provided
    prompt6 = Prompt(prompt=["cat in space", "cat sushi"])
    assert prompt6.as_list() == (["cat in space", "cat sushi"], ["", ""])
    assert prompt6.as_string() == ("cat in space, cat sushi", "")
    assert prompt6.as_dict() == ({0: "cat in space", 1: "cat sushi"}, {0: ""})
"""