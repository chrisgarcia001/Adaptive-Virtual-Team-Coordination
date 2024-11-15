import os

# Read the specified input file.
# @param lines: True or False. If True, return a list of lines, otherwise return the contents as a single string.
def read_file(input_filepath, lines=False):
    with open(input_filepath, 'r') as f:
        if lines:
            return f.readlines()
        else:
            return ''.join(f.readlines())

# Write text to a file, forcing the creation of any containing folders if they do not exist.	
def write_file(output_text, filename):
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    with open(filename, "w") as f:
        f.write(output_text)
