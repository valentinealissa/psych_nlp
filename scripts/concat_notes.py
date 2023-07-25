#!/usr/bin/env python

import os
import pandas as pd
import argparse


def combine_notes(data_frame):
    data_frame = data_frame.sort_values(by=['PAT_ID', 'NOTE_ID', 'CONTACT_SERIAL_NUM', 'SPECIFIED_DATETIME', 'LINE'])
    data_frame['NOTE_TEXT'] = data_frame['NOTE_TEXT'].str.replace(' ', '')
    data_frame['NEW_NOTE_TEXT'] = data_frame.groupby(['NOTE_ID', 'CONTACT_SERIAL_NUM'])['NOTE_TEXT'].transform(lambda x: ' '.join(x))
    return data_frame.drop(columns=["LINE", "NOTE_TEXT"]).drop_duplicates()


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Generate concatenated outpatient notes')
    # parser.add_argument('--data_path',
    #                     dest='data_path',
    #                     type=str,
    #                     help='Path to data files')
    # parser.add_argument('--in_file',
    #                     dest='in_file',
    #                     type=str,
    #                     help='File name for input data file')
    # parser.add_argument('--out_file',
    #                     dest='out_file',
    #                     type=str,
    #                     help='File name for output data file')
    # args = parser.parse_args()
    data_path = "/Users/valena17/psych_nlp/data/"
    in_file = "PSYCH_NOTES_092020.csv"
    out_file = "CONCAT_PSYCH_NOTES_092020.csv"
    _filename = os.path.join(data_path, in_file)
    _out_filename = os.path.join(data_path, out_file)
    # _filename = os.path.join(args.data_path, args.in_file)
    # _out_filename = os.path.join(args.data_path, args.out_file)
    with open(_filename) as f:
        df = pd.read_csv(f, parse_dates=[4, 8, 9], quotechar="\"")
        df = df.drop(columns=['Unnamed: 0.1', 'Unnamed: 0'])  # Unnamed: 0.1, Unnamed: 0 --- drop columns with nothing
    # max_notes_patient1 = df.loc[(df["PAT_ID"] == "Z2801050")]
    # max_notes_patient2 = df.loc[(df["MRN"] == "7186513")]
    # max_notes_patient3 = df.loc[(df["MRN"] == "8607090")]
    # notes_3patients = pd.concat([max_notes_patient1, max_notes_patient2, max_notes_patient3])
    # test = combine_notes(notes_3patients)
    # test.to_csv("/Users/valena17/psych_nlp/data/agg_sample_3patients.csv")
    concat_notes = combine_notes(df)
    concat_notes.to_csv(_out_filename)  # defaults to ‘utf-8’ encoding
