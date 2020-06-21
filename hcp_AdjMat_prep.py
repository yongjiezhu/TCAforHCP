#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 14:24:14 2019
===============================================================
Mearsuring task-related dynamics of time-frequency connectivity
===============================================================
(V2: if os.path.join(data_path2save):)
1.load subjects data one by one (using os.path.exists(directory))
  someeimes subjects not have morte or wrkmem exprements.
2.hcp_read_epochs to read the preprocessd epochs
3.hcp.anatomy.compute_forward_stack:We'll now use a convenience function to get our forward and source models
  instead of computing them by hand.
4.mne.compute_covariance
5. 
@author: Yongjie
"""
import os
import hcp
import mne
import numpy as np
from hcp import preprocessing as preproc
from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs

# path setting
storage_dir = '/run/media/yozhu/Seagate/1-NeuralData/mne-hcp-data'
hcp_path = os.path.join(storage_dir, 'HCP')
recordings_path = os.path.join(storage_dir, 'hcp-meg')
subjects_dir = os.path.join(storage_dir, 'hcp-subjects')
data_type = 'task_motor' # 'task_working_memory' 'task_motor'
f_type = 'Motort'      # 'Wrkmem' 'Motort' 
run_index = 0
onset = 'stim' #
# scan subjects
for subject in os.listdir(hcp_path):
    # if exist data_type in this sub
    fname = os.path.join(hcp_path,subject,'MEG',f_type)
    if not os.path.exists(fname):
        continue
    
    data_path2save=os.path.join(storage_dir,'wpli',subject,f_type,onset)
    if os.path.exists(data_path2save):
        continue
    # let's get the rpochs data
    hcp_epochs = hcp.read_epochs(onset=onset, subject=subject,
                                 data_type=data_type,hcp_path=hcp_path)
    
    hcp_epochs.resample(sfreq=256) # save memory
    # lets use a convenience function to get our forward and source models
    src_outputs = hcp.anatomy.compute_forward_stack(
            subject=subject,subjects_dir=subjects_dir,
            hcp_path=hcp_path, recordings_path=recordings_path,
            # seed up computations here. Setting add_dist to True may improve the accuracy
            src_params=dict(add_dist=False),
            info_from=dict(data_type=data_type, run_index=run_index))
    fwd = src_outputs['fwd']
    del src_outputs
    #=================================================================
    # just using baseline to compute the noise cov
    # after this, using empty room noise
    #noise_cov = mne.compute_covariance(hcp_epochs, tmax=-0.5,method=['shrunk', 'empirical'])
    #=================================================================
    # Now we can compute the noise covariance. For this purpose we will apply
    # the same filtering as was used for the computations of the ERF
    raw_noise = hcp.read_raw(subject=subject, hcp_path=hcp_path,
                             data_type='noise_empty_room')
    raw_noise.load_data()
    # apply ref channel correction and drop ref channels
    preproc.apply_ref_correction(raw_noise)
    
    # Note: MNE complains on Python 2.7
    raw_noise.filter(0.50, None, method='iir',
                     iir_params=dict(order=4, ftype='butter'), n_jobs=1)
    raw_noise.filter(None, 60, method='iir',
                     iir_params=dict(order=4, ftype='butter'), n_jobs=1)
    #Note that using the empty room noise covariance will inflate the SNR of the
    #evkoked and renders comparisons  to `baseline` rather uninformative.
    noise_cov = mne.compute_raw_covariance(raw_noise, method='empirical')
    del raw_noise
    #=======================================================================
    # inverse operator
    inverse_operator = make_inverse_operator(hcp_epochs.info, fwd, 
                                             noise_cov=noise_cov, loose=0.2,depth=0.8)
    method = 'MNE'
    snr = 1.
    lambda2 = 1./snr**2
    src = inverse_operator['src']
    stcs_list=[apply_inverse_epochs(hcp_epochs[sti], inverse_operator, lambda2, method, label=None,
                                    pick_ori="normal", return_generator=True) for sti in hcp_epochs.event_id]
    sti_names=[sti for sti in hcp_epochs.event_id]
    labels = mne.read_labels_from_annot(subject,parc='aparc',
                                        subjects_dir=subjects_dir)
    label_names=[label.name for label in labels]
    label_ts_list = [mne.extract_label_time_course(stcs_list[ite], labels, src, mode='mean_flip',
                                                   return_generator=False) for ite in range(len(stcs_list))]
    
    del stcs_list,noise_cov,inverse_operator,src
    # using mne build-in function
    from mne.connectivity import spectral_connectivity
    cwt_freqs = np.linspace(4,45,42)
    cwt_n_cycles=np.linspace(3,15,42)
    sfreq = hcp_epochs.info['sfreq']  # the sampling frequency
    del hcp_epochs
    con_methods = 'wpli'
    con_list=[]
    for label_ts in label_ts_list:
        con, freqs, times, n_epochs, n_tapers = spectral_connectivity(
                label_ts, method=con_methods, mode='cwt_morlet', sfreq=sfreq, cwt_freqs=cwt_freqs,
                cwt_n_cycles=cwt_n_cycles, faverage=False, n_jobs=1)
        con_list.append(con.astype(np.float32))
    
    del label_ts_list
    # save
    data_path2save=os.path.join(storage_dir,con_methods,subject,f_type,onset)
    if not os.path.exists(data_path2save):
        os.makedirs(data_path2save)
    from scipy import io
    io.savemat(data_path2save+'/allData.mat',dict(label_names=label_names,sti_names=sti_names,
                                                  con_list=con_list,freqIndex=freqs,tIndex=times),oned_as='row')
    del con_list