# Self-contained utility for extracting parameters from a Colab notebook
# and saving them to a file.
# #@param metadata turns a Python variable into a form field that users can enter. 
# If you want your ipynb to be the one source of truth, then 
# use this to generate the schema it uses. 

import json
import re
import json
from jinja2 import Template


param_pattern = r"^(.*?)#\@param(.*)$"
variable_pattern = r"^(\w+)\s*=\s*(.*)$"
type_pattern = r"\{type\s*:\s*['\"](.*)['\"]\}"
constraints_pattern = r"\[(.*)\]"

COLAB_TO_PYTHON_TYPE_MAP = {
    "string": "str",
    "integer": "int",
    "number": "float",
    "boolean": "bool"
}

def strip_quotes_from_embedded_list_items(input_string):
    split_strings = input_string.split(',')

    # Removing double quotes from each substring
    no_quotes_strings = [s.strip(' \'"') for s in split_strings]

    # Joining the modified substrings
    output_string = ', '.join(no_quotes_strings)
    return output_string

def generate_parameters_file(parameters):
    methods = list(set([obj["method"] for obj in parameters]))

    with open("parameters.tmpl.py") as file:
        template = Template(file.read())

    with open("parameters.py", "w") as output:
        output.write(template.render(objects=parameters, methods=methods))

def extract_from_line(line):
    """Extracts the variable name, value, type, and constraints from a line of code containing a Colab #@param annotation"""
    variable_name, variable_value, type, constraints = None, None, None, None

    # break up the line into the variable and the type/constraints
    match = re.match(param_pattern, line)

    # yep definitely a #@param annotation
    if match:
        variable_and_value = match.group(1).strip(' ')
        type_and_constraint = match.group(2).strip(' ')

        variable_match = re.match(variable_pattern, variable_and_value)
        if variable_match:
            variable_name = variable_match.group(1)
            variable_value = variable_match.group(2).strip('"')

        type_match = re.search(type_pattern, type_and_constraint)
        if type_match:
            type = type_match.group(1)

        constraints_match = re.search(constraints_pattern, type_and_constraint)
        if constraints_match:
            constraints = constraints_match.group(1)

        # turn constraints into a single string
        constraints = strip_quotes_from_embedded_list_items(constraints) if constraints else None
        
        # rule: if there are constraints, then the type is a string, always even if
        if constraints and type is not None:
            type = "string"

        # map colab to python type
        type = COLAB_TO_PYTHON_TYPE_MAP.get(type, None) if type else None

    else:
        print("Not a #@param annotation: skipping")

    return variable_name, variable_value, type, constraints



def extract_colab_params(ipynb_file):
    variables = []

    with open(ipynb_file) as f:
        notebook = json.load(f)

    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            current_method_name = "(Global)"
            params = []
            for line in cell['source']:
                # if we detect a python method definition, set current_method_name to the python method name
                if(line.startswith("def ")): 
                    current_method_name = line.split("def ")[1].split("(")[0]
                
                # if we detect a Google Colab #@param annotation, extract and add it to the list of parameters
                if "#@param" in line:
                    variable_name, variable_value, type_value, constraints_value = extract_from_line(line)
                    params.append( {
                        "method": current_method_name,
                        "name": variable_name,
                        "default": variable_value,
                        "constraints": constraints_value,
                        "type": type_value
                    })
                    print(f"### {current_method_name}, {variable_name} = '{variable_value}', type = '{type_value}', constraints = '{constraints_value}'")
            variables.extend(params)

    # print(f"## extracted {len(variables)} variables from {ipynb_file}")
    return variables

# take ipynb file from --notebook parameter given in command line
# and extract parameters from it
def main():
    import argparse
    parser = argparse.ArgumentParser(description='Extract parameters from a Colab notebook')

    parser.add_argument('--notebook', type=str, help='Colab notebook file')
    args = parser.parse_args()

    parameters = extract_colab_params(args.notebook)
    print(f"Extracted {len(parameters)} variables from {args.notebook}")

    generate_parameters_file(parameters)


    # json pretty print the variables
    print(json.dumps(parameters, indent=2))


if __name__ == "__main__":
   main()
   
