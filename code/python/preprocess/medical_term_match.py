import ahocorasick
import pandas as pd


class AcMatch(object):
    # A method to find match words between the dictionary and the text paragraph
    def __init__(self, dict_file):
        self.dict_file = dict_file

    def add_trie(self):
        fp = open(self.dict_file)
        dict_list = []
        self.A = ahocorasick.Automaton()
        for line in fp:
            dict_list.append(line.rstrip("\n"))
        fp.close()
        for index, word in enumerate(dict_list):
            self.A.add_word(word, (index, word))
        self.A.make_automaton()

    def preprocess(self, text_col):
        medical_str = ""
        for i in self.A.iter(text_col):
            medical_str += i[1][1]+ " "
        medical_str = medical_str.replace('\n', '')
        return medical_str



def main():
    data_dir = "../../data/"
    dict_dir = "../../dictionaries/"
    dict_file = "wordlist.txt"
    filename = "NOTEEVENTS_sampleforpandas.csv"
    output_file_name = "medical_term_only.csv"

    names = ["row_id", "subject_id", "hadm_id", "chart_date", "chart_time", "store_time", "category",
            "description", "cgid", "iserror", "text"]

    match_process = AcMatch(dict_dir+dict_file)
    match_process.add_trie()

    chunksize = 10 ** 6
    for chunk in pd.read_csv(data_dir+filename, names = names, chunksize=chunksize):
        chunk["text"] = chunk["text"].apply(match_process.preprocess)
        chunk.to_csv(data_dir + output_file_name, mode='a', index=False, header=False)

if __name__=='__main__':
    main()