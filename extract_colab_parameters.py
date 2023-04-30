# Self-contained utility for extracting parameters from a Colab notebook
# and saving them to a file.
# #@param metadata turns a Python variable into a form field that users can enter. 
# If you want your ipynb to be the one source of truth, then 
# use this to generate the schema it uses. 

import json
import re

import re

def extract_parameters(input_list):
    output_list = []
    
    for line in input_list:
        print(f"line: {line}")
        # Skip lines that do not contain #@param
        if "#@param" not in line:
            continue

        # Use regular expressions to parse the line
        name_match = re.search(r"(\w+)\s*=", line)
        value_match = re.search(r"=\s*(.*?)\s*#@", line)
        type_match = re.search(r"type:\"(.*?)\"", line)
        constraints_match = re.search(r"\[(.*?)\]", line)


        if name_match and value_match and type_match:
            name = name_match.group(1)
            default = value_match.group(1)
            param_type = type_match.group(1)
            
            if constraints_match:
                constraints = constraints_match.group(1).split(", ")
            else:
                constraints = None
                
            # Convert default value to the appropriate type
            if param_type == "integer":
                default = int(default)
            elif param_type == "boolean":
                default = default == "True"
                
            output_list.append({
                "name": name,
                "default": default,
                "constraints": constraints,
                "type": param_type
            })
    
    return output_list

def extract_colab_params(ipynb_file):
    variables = []

    with open(ipynb_file) as f:
        notebook = json.load(f)

    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            code = ''.join(cell['source'])

            # Find all variables with the #@param annotation
            params = re.findall(r'(.+?)#\@param\s+\{(.+?)\}', code)
            variables.append(extract_parameters(params))

    return variables

# take ipynb file from --notebook parameter given in command line
# and extract parameters from it

import argparse

parser = argparse.ArgumentParser(description='Extract parameters from a Colab notebook')

parser.add_argument('--notebook', type=str, help='Colab notebook file')
args = parser.parse_args()

variables = extract_colab_params(args.notebook)
print(f"Extracted {len(variables)} variables from {args.notebook}")

# json pretty print the variables
print(json.dumps(variables, indent=2))

