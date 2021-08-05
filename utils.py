def load_data(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as rf:
        for line in rf:
            data.append(line)
    return data
