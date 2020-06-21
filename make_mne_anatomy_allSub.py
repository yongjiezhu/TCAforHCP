#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 19:06:06 2019

@author: Yongjie
"""

import os
import hcp
storage_dir = '/run/media/yozhu/Seagate/1-NeuralData/mne-hcp-data'

# This is where the data are after downloading from HCP
hcp_path = storage_dir + '/HCP'

# this is the new directory to be created
subjects_dir = storage_dir + '/hcp-subjects'

# it will make the subfolders 'bem', 'mir', 'surf' and 'label'
# and populate it with symbolic links if possible and write some files
# where information has to be presented in a different format to satisfy MNE.

# this is where the coregistration matrix is written as
# `105923-head_mri-transform.fif`
recordings_path = storage_dir + '/hcp-meg'

# 
for subject in os.listdir(hcp_path):
    hcp.make_mne_anatomy(subject = subject,
                         hcp_path=hcp_path,
                         recordings_path=recordings_path,
                         subjects_dir=subjects_dir)

