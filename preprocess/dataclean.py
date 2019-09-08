import pandas as pd
import re


def lower_text(text_col):
    # Transform all text to lower case
    output = text_col.lower()

    return output

def rm_common_words(text_col):
    # Remove common words in all summaries
    common_words = re.compile("(sex:\s*f)|(sex:\s*m)|(service)|(completed by)")
    output = re.sub(common_words, " ", text_col)
    return output

def rm_dateinfo(text_col):
    # Remove date related information
    date_info = re.compile("(admission date)|(date of admission)|(discharge date)|(date of discharge)|(date of birth)")
    time_info = re.compile("(([0-1]{0,1}[0-9]( )?(AM|am|aM|Am|PM|pm|pM|Pm))|"
                           "(([0]?[1-9]|1[0-2])(:|\.)[0-5][0-9]( )?(AM|am|aM|Am|PM|pm|pM|Pm))|"
                           "(([0]?[0-9]|1[0-9]|2[0-3])(:|\.)[0-5][0-9]))")
    output = re.sub(date_info, " ", text_col)
    output = re.sub(time_info, " ", output)

    return output


def rm_content_in_bracket(text_col):
    # Remove all information in [] bracket
    output = re.sub(r'(\[\*\*)([^\*]+)(\*\*\])', " ", text_col)

    return output

def rm_testresults(text_col):
    # Remove lab test measurement
    measurement = re.compile("((g/dl)|(k/ul)|(mg/dl)|(meq/l)|(mmol/l)|(iu/l))")
    output = re.sub(measurement, " ", text_col)
    return output

def rm_medication_related(text_col):
    # Remove medication frequency and dose
    # Reference https://github.com/farinstitute/ReadmissionRiskDoctorNotes/blob/master/02_PrepareText.ipynb
    freq = re.compile("((by mouth)|(q\sd)|(on odd days)|(times a day)|(times per day)|(at dinner)|"
                           "(at breakfast)|(at bedtime)|(q\. day)|(q d\.)|(q a\.m\.)|(q p\.m\.)|(q am)|"
                           "(q pm)|(q day)|(q\.day)|(q\.hs)|(meq)|(q\.d\.)|(b\.i\.d\.)|(t\.i\.d\.)|"
                           "(q\.i\.d)|(q\dh)|(p\.r\.n\.)|(p\.o\.)|(daily))")
    dose = re.compile("((micrograms)|(microgram)|(mgs)|(mg)|(ml)|(milligrams)|(milligram)|(milliequivalents))")
    output = re.sub(freq, " ", text_col)
    output = re.sub(dose, " ", output)
    return output

def rm_text_num(text_col):
    # Remove text number
    num = re.compile("(zero)|(one)|(two)|(three)|(four)|(five)|(six)|(seven)|(eight)|(nine)|"
          "(ten)|(eleven)|(twelve)|(thirteen)|(fourteen)|(fifteen)|(sixteen)|(seventeen)|(eighteen)")

    output = re.sub(num, " ", text_col)
    return output

def keep_letter_only(text_col):
    # Keep letters only
    letter = re.compile("[^a-z]")
    output = re.sub(letter, " ", text_col)
    return output

def rm_duplicate_whitespace(text_col):
    # Remove duplicate whitespace
    output = re.sub(" +"," ", text_col)

    return output

def preprocess(text_col):
    output = lower_text(text_col)
    output = rm_common_words(output)
    output = rm_dateinfo(output)
    output = rm_content_in_bracket(output)
    output = rm_testresults(output)
    output = rm_medication_related(output)
    output = rm_text_num(output)
    output = keep_letter_only(output)
    output = rm_duplicate_whitespace(output)

    return output


def main():
    data_dir = "../../data/"
    filename = "NOTEEVENTS_sampleforpandas.csv"
    output_file_name = "cleanstep.csv"
    names = ["row_id", "subject_id", "hadm_id", "chart_date", "chart_time", "store_time", "category",
             "description", "cgid", "iserror", "text"]

    chunksize = 10 ** 6
    for chunk in pd.read_csv(data_dir+filename, names = names, chunksize=chunksize):
        chunk["text"] = chunk["text"].apply(preprocess)
        chunk.to_csv(data_dir + output_file_name, mode='a', index=False, header=False)

if __name__ == '__main__':
    main()