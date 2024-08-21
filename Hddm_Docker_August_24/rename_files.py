# Program that renames files that create complications for downloading
# Veronika Wendler


# Rename Files that have a : in their filename 

import os, sys, pickle, time, csv
import pandas as pd
import numpy as np

model_base_name = 'painreward_behavioural_data_combined_new_'
model_names = [
                'r0', 'r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7', 'r8', 'r9', 'r10',
                'r11', 'r12', 'r13', 'r14', 'r15', 'r16', 'r17', 'r18','r19', 'r20',
                'r21', 'r22', 'r23', 'r24','r25', 'r26', 'r27', 
                'r28', 'r29', 'r30',
                'r31','r32', 'r33','r34','r35', 'r36', 'r37', 'r38', 'r39', 'r40',
                'r41','r42', 'r43','r44','r45', 'r46', 'r47', 'r48', 'r49', 'r50', 
                'r51', 'r52', 'r53'
               ]

for m in model_names:
    file_directory = f'/home/jovyan/OfficialTutorials/figures/{model_base_name}{m}/diagnostics'
    # list of files in the directory
    for filename in os.listdir(file_directory):
        # full file path
        full_path = os.path.join(file_directory, filename)
        # replace problematic characters
        new_filename = filename.replace(':', '_').replace('(', '_').replace('[', '_').replace(')', '_').replace(']', '_').replace('T.M', 'T_M').replace('T.P', 'T_P').replace('T.low', 'T_low')
        new_full_path = os.path.join(file_directory, new_filename)
        
        # Rename file
        os.rename(full_path, new_full_path)