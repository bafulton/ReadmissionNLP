import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    import imp

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import *
from multiprocessing import Pool
import numpy as np
import nltk
import re

location_of_source_data = "./data/NOTEEVENTS.csv"
output_csv_name = "./data/NOTEEVENTS_CLEANED.csv"
med_dict_location = "./dictionaries/medical_dict.txt"

medical_spell_check_removal = False


fp = open(med_dict_location)
dict_list = []
for line in fp:
    dict_list.append(line.lower().rstrip("\n"))
fp.close()

nltk.download('stopwords')
stop = stopwords.words("english")

def parellel_dataframes(incoming_df, applied_func, df_chunks, number_cores, header):
    split_dataframe = np.array_split(incoming_df[header], df_chunks)
    pool = Pool(number_cores)
    final_dataframe = pd.concat(pool.map(applied_func, split_dataframe))
    pool.close()
    pool.join()
    final_dataframe = pd.DataFrame(final_dataframe)
    incoming_df[header] = final_dataframe[header]

    return incoming_df

def parellel_dataframes_for_spell_check(incoming_df, applied_func, df_chunks, number_cores, header):
    split_dataframe = np.array_split(incoming_df[header], df_chunks)
    pool = Pool(number_cores)
    final_dataframe = pd.concat(pool.map(applied_func, split_dataframe))
    pool.close()
    pool.join()

    final_dataframe = pd.DataFrame(final_dataframe)
    final_dataframe.reset_index(inplace=True)
    print(final_dataframe)
    incoming_df[header] = final_dataframe[0].values
    print(incoming_df[header])
    return incoming_df


def remove_stop_words(incoming_df):

    incoming_df = incoming_df.apply(lambda x: ' '.
                                                    join([word for word in x.split() if word not in (stop)]))
    return incoming_df


def remove_values_between_brackets(incoming_df):
    brack_val_remove = re.compile(r"\[.*?\]")
    incoming_df = incoming_df.replace({brack_val_remove: " "}, regex=True)

    return incoming_df


def remove_all_non_alpha_chars(incoming_df):
    remove_special_chars = re.compile(r'[^a-zA-Z]+')
    incoming_df = incoming_df.replace({remove_special_chars: " "}, regex=True)

    return incoming_df


def stem_words(incoming_df):
    port_stem = PorterStemmer()
    incoming_df = incoming_df.apply(lambda x: ' '.
                                                    join([port_stem.stem(word) for word in x.split()]))
    return incoming_df


def remove_white_spaces(incoming_df):
    incoming_df = incoming_df.apply(lambda x: " ".join(x.split()))

    return incoming_df


def capitalize_everything(incoming_df):
    incoming_df = incoming_df.str.lower()

    return incoming_df


def remove_single_letter(incoming_df):
    incoming_df = incoming_df.apply(lambda x: ' '.join([w for w in x.split() if len(w) > 1]))

    return incoming_df


def remove_specific_words(incoming_df):
    custom_stop_words = ['admission', 'date', 'discharge', 'date','birth', 'sex', 'service', 'medicine',
                         'allergies', 'g/dl','k/ul','mg/dl','meq/l','mmol/l','iu/l']
    incoming_df = incoming_df.apply(lambda x: ' '.join( [w for w in x.split() if w not in custom_stop_words]))

    return incoming_df


def remove_medication(incoming_df):
    freq = re.compile(r"((by mouth)|(q\sd)|(on odd days)|(times a day)|(times per day)|(at dinner)|"
                      r"(at breakfast)|(at bedtime)|(q\. day)|(q d\.)|(q a\.m\.)|(q p\.m\.)|(q am)|"
                      r"(q pm)|(q day)|(q\.day)|(q\.hs)|(meq)|(q\.d\.)|(b\.i\.d\.)|(t\.i\.d\.)|"
                      r"(q\.i\.d)|(q\dh)|(p\.r\.n\.)|(p\.o\.)|\b(daily)\b|\b(sig)\b|\b(day)\b|"
                      r"\b(times)\b|\b(tablet)\b)")
    incoming_df = incoming_df.replace({freq: " "}, regex=True)

    return incoming_df

def remove_units(incoming_df):

    dose = re.compile(r"(\b(micrograms)\b|\b(microgram)\b|\b(mgs)\b|\b(mg)\b|\b(ml)\b|\b(milligrams)\b|\b(milligram)\b|"
                      r"\b(milliequivalents)\b|\b(po)\b)")
    incoming_df = incoming_df.replace(dose, " ", regex=True)

    return incoming_df

def remove_text_numbers(incoming_df):


    num = re.compile(r"\b(zero)\b|\b(one)\b|\b(two)\b|\b(three)\b|\b(four)\b|\b(five)\b|\b(six)\b|\b(seven)\b|\b(eight)\b"
                     r"|\b(nine)\b|\b(ten)\b|\b(eleven)\b|\b(twelve)\b|\b(thirteen)\b|\b(fourteen)\b|\b(fifteen)\b|"
                     r"\b(sixteen)\b|\b(seventeen)\b|\b(eighteen)\b")

    incoming_df = incoming_df.replace(num, " ", regex=True)

    return incoming_df

def medical_dict_replace(incoming_df):
    if incoming_df.size > 0:
        spell_checked_input = ' '.join(set(dict_list).intersection(incoming_df[0].split()))
        incoming_df = pd.Series(spell_checked_input)

    return incoming_df

def extract_section(txt, sec_headr = "Discharge Condition:"):
	# Given the column of "Text" from the noteeevnts.csv data, it extracts some
	# selected sections as separate columns in a new dataframe.
    matched_sec = None
    txt = str(txt).lower()
    # extract the target section with header
    m = re.search("\n\n" + sec_headr + r".*?" + "\n\n", txt, re.IGNORECASE | re.DOTALL)
    if m is not None:
        matched_sec = m.group()
        # strip the header off
        matched_sec = matched_sec[matched_sec.find(':')+2:-2]
    return matched_sec

if __name__ == '__main__':
    new_parse = True
    new_section_divider = True

    if new_parse:
        chunksize = 10000
        print("BEGINNING PARSE")

        TextFileReader = pd.read_csv(location_of_source_data,
                                     chunksize=chunksize,iterator=True, escapechar='\\',
                                     converters={"text": lambda x: x.replace('\r\n', '\n')})

        print("CONCAT ALL CHUNKED CSV")
        note_events_df = pd.concat(TextFileReader, ignore_index=True)
        note_events_df.columns = map(str.lower, note_events_df.columns)
        print("PICKLE CSV")

        print("EXTRACTED AND READING PICKLE")
        # note_events_df = pd.read_pickle("./note_events_all.pkl")
        print(note_events_df.head())
        print(note_events_df.size)
        print(note_events_df.shape)
        print()

        print('NOTES EXTRACTED')
        print(note_events_df.size)
        print(note_events_df.shape)
        # print(note_events_df.tail())
        # print(note_events_df['text'][0])

    sec_headrs = [r"((discharge condition.*?:)|(condition at discharge.*?:))",\
                  r"((discharge disposition.*?:)|(discharge status.*?:))",
                  r"discharge diagnos.*?:",r"discharge medicat.*?:",\
                  r"discharge instruction.*?:"]


    if new_section_divider:
        print("EXTRACTING HEADERS")
        for sec_headr in sec_headrs:
            note_events_df[sec_headr] = note_events_df["text"].apply(lambda x: extract_section(x, sec_headr))

        note_events_df.rename(index=str, columns={'((discharge condition.*?:)|(condition at discharge.*?:))':'DISCHARGE_CONDITION',
                                       "((discharge disposition.*?:)|(discharge status.*?:))":'DISCHARGE_DISPOSITION',
                                       'discharge diagnos.*?:':'DISCHARGE_DIAGNOSIS',
                                       'discharge medicat.*?:':'DISCHARGE_MEDICATION',
                                       'discharge instruction.*?:':'DISCHARGE_INSTRUCTIONS'}, inplace=True)

        note_events_df = note_events_df.fillna('')
        print("SAVING NEWLY EXTRACTED HEADERS CSV")
        note_events_df.to_csv("./extracted_headers_all.csv", mode='w', index=False)
        print("SAVING NEWLY EXTRACTED HEADERS PKL")
        # note_events_df.to_pickle("./extracted_headers_all.pkl")

        print("HEADERS ALL EXTRACTED")

        print(note_events_df.shape)
        print(note_events_df.tail())
        print(list(note_events_df))

    # note_events_df = pd.read_pickle("./extracted_headers_all.pkl")
    print(note_events_df.head())

    headers_of_interest = ['text', 'DISCHARGE_CONDITION', 'DISCHARGE_DISPOSITION',
                           'DISCHARGE_DIAGNOSIS', 'DISCHARGE_MEDICATION',
                           'DISCHARGE_INSTRUCTIONS']

    # headers_of_interest = ['DISCHARGE_DISPOSITION']

    for header_name in headers_of_interest:

        if medical_spell_check_removal:
            print("MEDICAL DICTIONARY REPLACEMENT")
            note_events_df = parellel_dataframes_for_spell_check(note_events_df,medical_dict_replace, header=header_name,
                                             df_chunks=16, number_cores=8)
        else:

            print(note_events_df[header_name].head())
            print("REMOVING STOP WORDS")
            note_events_df = parellel_dataframes(note_events_df,remove_stop_words, header=header_name,
                                                 df_chunks=16, number_cores=8)
            print("REMOVING BRACKET STAR STAR WORDS")
            note_events_df = parellel_dataframes(note_events_df,remove_values_between_brackets, header=header_name,
                                                 df_chunks=16, number_cores=8)
            print("REMOVING ALL NON ALPHA/NON SPACES")
            note_events_df = parellel_dataframes(note_events_df,remove_all_non_alpha_chars, header=header_name,
                                                 df_chunks=16, number_cores=8)
            print("REMOVE ALL WHITE SPACES")
            note_events_df = parellel_dataframes(note_events_df,remove_white_spaces, header=header_name,
                                                 df_chunks=16, number_cores=8)
            print("CAPITALIZE ALL LETTERS")
            note_events_df = parellel_dataframes(note_events_df,capitalize_everything, header=header_name,
                                                 df_chunks=16, number_cores=8)
            print("REMOVE SINGLE LETTERS")
            note_events_df = parellel_dataframes(note_events_df,remove_single_letter, header=header_name,
                                                 df_chunks=16, number_cores=8)
            print("REMOVE CUSTOM STOP WORDS")
            note_events_df = parellel_dataframes(note_events_df,remove_specific_words, header=header_name,
                                               df_chunks=16, number_cores=8)
            print("REMOVE MEDICATION")
            note_events_df = parellel_dataframes(note_events_df, remove_medication, header=header_name,
                                                df_chunks=16, number_cores=8)
            print("REMOVE UNITS")
            note_events_df = parellel_dataframes(note_events_df, remove_units, header=header_name,
                                                 df_chunks=16, number_cores=8)
            print("REMOVE TEXT NUMBERS")
            note_events_df = parellel_dataframes(note_events_df, remove_text_numbers, header=header_name,
                                                 df_chunks=16, number_cores=8)
            print("REMOVE ALL WHITE SPACES...AGAIN")
            note_events_df = parellel_dataframes(note_events_df,remove_white_spaces, header=header_name,
                                                 df_chunks=16, number_cores=8)
            note_events_df = note_events_df.fillna('')


    print("SAVING TRAINING SENTENCES")
    note_events_df.to_csv(output_csv_name, encoding='utf-8', index=False)
    # print("SAVING PICKLED CLEANED UP")
    # note_events_df.to_pickle("./note_events_clean_repeat_remove.pkl")
    # chunksize = 10000
    # print("BEGINNING PARSE")
    # TextFileReader = pd.read_csv("C:/Users/sunnyr/PycharmProjects/clinspell/note_events_clean_repeat_remove.csv",
    #                              chunksize=chunksize, iterator=True, escapechar='\\')
    # print("CONCAT ALL CHUNKED CSV")
    # note_events_df = pd.concat(TextFileReader, ignore_index=True)
    # note_events_df.to_pickle("./note_events_clean.pkl")
    # note_events_df = pd.read_pickle("./note_events_clean.pkl")
    # note_events_df = note_events_df.fillna('delete')
    # print(note_events_df.DISCHARGE_CONDITION)
    #
    # print(pd.Series(' '.join(note_events_df.TEXT).split()).value_counts()[:100])
    # note_events_df.to_csv('training_notes_repeat_remove.txt', encoding='utf-8', index=False)


