from data.preprocess import tune_preprocessing, preprocess,HP_DICT

import glob
import os
from fire import Fire


def preprocess_data(audio_loc,seg_loc,out_ext,\
                    audio_subdirs='',seg_subdirs='',\
                        audio_ext='.wav',seg_ext='.txt',parallel = True):
    

    ## default: grabs all subdirs



    audio_dirs = glob.glob(os.path.join(audio_loc,audio_subdirs))
    seg_dirs = glob.glob(os.path.join(seg_loc,seg_subdirs))
    out_dirs = [os.path.join(o,out_ext) for o in audio_dirs]

    split_aud_sub = audio_subdirs.split('/')
    split_seg_sub = seg_subdirs.split('/')
    aud_sub_depth=len(split_aud_sub)
    seg_sub_depth=len(split_seg_sub)

    audio_tags = [a.split('/')[-aud_sub_depth] for a in audio_dirs]
    seg_tags = [s.split('/')[-seg_sub_depth] for s in seg_dirs]
    valid = True
    for tag in seg_tags:
        try:
            assert tag in audio_tags
        except:
            print(f"{tag} not in audio tags")
            valid = False
    for tag in audio_tags:
        try:
            assert tag in seg_tags
        except:
            print(f"{tag} not in seg tags")
            valid = False 

    assert valid
    

    #print(audio_dirs[:5],seg_dirs[:5])
    audio_files = sum([glob.glob(os.path.join(a,'*' + audio_ext)) for a in audio_dirs],[])
    seg_files = sum([glob.glob(os.path.join(s,'*' + seg_ext)) for s in seg_dirs],[])
    audio_tags = [a.split('.wav')[0].split('/')[-1] for a in audio_files]
    seg_tags = [s.split('nrd.txt')[0].split('/')[-1] for s in seg_files]


    valid = True
    for tag in seg_tags:
        try:
            assert tag in audio_tags
        except:
            print(f"{tag} not in audio tags")
            valid = False
    for tag in audio_tags:
        try:
            assert tag in seg_tags
        except:
            print(f"{tag} not in seg tags")
            valid = False 

    assert valid
    #print(audio_files,seg_files)
    #print(audio_files[:5],seg_files[:5])
    hps = tune_preprocessing(audio_files,seg_files,HP_DICT)

    preprocess(audio_dirs,out_dirs,hps,audio_ext=audio_ext,parallel=parallel)

if __name__ == '__main__':

    Fire(preprocess_data)