# Self-contained utility for extracting parameters from a Colab notebook
# and saving them to a file.
# #@param metadata turns a Python variable into a form field that users can enter. 
# If you want your ipynb to be the one source of truth, then 
# use this to generate the schema it uses. 

import json
import re
import json
from jinja2 import Template

pattern = re.compile(r'\s*(\w+)\s*=\s*([^#\n]+)\s*#@param(?:.*type:\s*([\'\"])(\w+)\3)?(?:\s*\[(.*)\])?')

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
                    match = pattern.match(line)
                    if match:
                        variable_name, value, _, var_type, constraints = match.groups()   
                        value = re.sub(r'^\s*["\']*|["\']*\s*$', '', value)         
                        constraints = strip_quotes_from_embedded_list_items(constraints) if constraints else None
                        if constraints:
                            var_type = "str"

                        # map colab to python type
                        var_type = COLAB_TO_PYTHON_TYPE_MAP.get(var_type, None)
                        params.append( {
                            "method": current_method_name,
                            "name": variable_name,
                            "default": value,
                            "constraints": constraints,
                            "type": var_type
                        })
                        print(f"### {current_method_name}, {variable_name} = '{value}', type = '{var_type}', constraints = '{constraints}'")
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
   
