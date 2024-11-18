def flatten_list(nested_list: list):
    flattened = []
    for item in nested_list:
        if isinstance(item, list):
            flattened.extend(item)
        else:
            flattened.append(item)
    return flattened

def char_count(s: str):
    return {char: s.count(char) for char in set(s)}