import csv


def read_csv(file):
    try:
        with open(file, mode='r', encoding='utf8') as csvfile:
            reader = csv.reader(csvfile)
            return list([row[0].strip(), row[1].strip()] for row in reader)
    except OSError:
        return list()


whisper_ja_replace = read_csv("dicts.whisper.ja.csv")
