#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 11:02:31 2018

@author: Dinesh
"""
import pandas as pd
import re
import os

def extract_section(txt, sec_headr = "Discharge Condition:"):
	# Given the column of "Text" from the noteeevnts.csv data, it extracts some
	# selected sections as separate columns in a new dataframe.
    matched_sec = None
    txt = str(txt).lower()
    # extract the target section with header
    m = re.search("\n\n"+sec_headr+r".*?"+"\n\n",txt,re.IGNORECASE | re.DOTALL)
    if m is not None: 
        matched_sec = m.group()
        # strip the header off
        matched_sec = matched_sec[matched_sec.find(':')+2:-2]
    return matched_sec

def main():
    data_dir = "../../../data/"
#    filename = "discharge_notes_sample.csv"
    filename = "NOTEEVENTS_sampleforhumans.csv"
#    output_file_name = "extracted_sections.csv"
    output_file_name = "new_extracted_sections.csv"
    names = ["row_id", "subject_id", "hadm_id", "chart_date", "chart_time", "store_time", "category",
             "description", "cgid", "iserror", "text"]
    

# =============================================================================
#     List of the sections to extract as columns, in the order they will be saved:
#     0) Discharge Condition
#     1) Discharge Disposition
#     2) Discharge Diagnosis/es
#     3) Discharge Medication
#     4) Discharge Instruction
#    Add More sections to extract as columns if needed, in lower case
# =============================================================================
    
    sec_headrs = [r"((discharge condition.*?:)|(condition at discharge.*?:))",\
                  r"discharge disposition.*?:",r"discharge diagnos.*?:",r"discharge medicat.*?:",\
                  r"discharge instruction.*?:"]

    df = pd.read_csv(os.path.join(data_dir,filename),usecols=names,escapechar='\\')
#    df = df[:20]
    for sec_headr in sec_headrs:
        df[sec_headr] = df["text"].apply(lambda x: extract_section(x,sec_headr))
    df[sec_headrs].to_csv(os.path.join(data_dir,output_file_name), mode='w', index=False, header=False)
    print("The saved csv file",output_file_name, "has shape:",df[sec_headrs].shape)
        
if __name__ == '__main__':
    main()