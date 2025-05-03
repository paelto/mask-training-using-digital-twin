import os

def to_full_path(relative_path):
    if os.name == 'nt':
        absolute_path = os.path.abspath(relative_path.replace('/','\\'))
        extended_path = r"\\?\{}".format(absolute_path)
        print(extended_path)
        return extended_path
    else:
        return relative_path
