# Kaldi - tutorial para treino de modelo acústico

According to Kaldi's [tutorial for dummies](http://kaldi-asr.org/doc/kaldi_for_dummies.html),
the directory tree for new projects must follow the structure below:

```
           path/to/kaldi/egs/my_base_dir/
                                 ├─ path.sh
                                 ├─ cmd.sh
                                 ├─ run.sh
                                 │ 
  .--------------.-------.-------:------.------.-------------.
  |              |       |       |      |      |             |
 MFCC/         data/   utils/  steps/  exp/  local/        conf/
  └─ make_mfcc   |                             └─ score.sh   ├─ decode.config
  .--------------:--------------.                            └─ mfcc.conf
  │              │              │
train/          test/         local/
  ├─ spkTR_1/    ├─ spkTE_1/    └─ dict/
  ├─ spkTR_2/    ├─ spkTE_2/        ├─ lexicon.txt
  ├─ spkTR_3/    ├─ spkTE_3/        ├─ non_silence_phones.txt
  ├─ spkTR_n/    ├─ spkTE_n/        ├─ optional_silence.txt
  │              │                  ├─ silence_phones.txt
  ├─ spk2gender  ├─ spk2gender      └─ extra_questions.txt
  ├─ wav.scp     ├─ wav.scp            
  ├─ text        ├─ text               
  ├─ utt2spk     ├─ utt2spk            
  └─ corpus.txt  └─ corpus.txt         
```

* __fb\_00\_create\_envtree.sh__ :
This script creates the directory structure shown above, except the `spkXX_n`
inside the `data/train` and `data/test` folders. Notice that the data-dependent
files (inside the `data` dir), although created, they __DO NOT__ have any
content yet. IOW, they're only initialized as empty files. A stupid choice of
the developer.

* __fb\_01\_split\_train\_test.sh__:
This script fulfills the `data/train` and `data/test` directories. The data is
divided as training set and test set, and the files within the dirs are
data-dependent. The folders `train/spkTR_n` and `test/spkTE_n` contain
symbolic links to the actual wav-transcription base dir.

* __fb\_02\_define\_localdict.sh__:
This script specially fulfills the files inside `local/dict` dir. A dependency
is the `g2p` software, which files must be in the same directory of the `fb_02_define_localdict.sh` script. 
The`g2p` software is available at https://gitlab.com/fb-nlp/nlp.git.

* __util\run.sh__:
This file is the script for training the acoustic models.

* __util\RESULTS__:
This file contains the results of the acoustic models obtained using the Demo Audio Corpora, which is available at [https://gitlab.com/fb-asr/fb-am-tutorial/demo-corpora.git][1].   


If you are using the Demo corpora or another similar small corpora, you will need to change the value of the `num_utts_subset` parameter in the file `kaldi/egs/YOUR_PROJECT_NAME/steps/nnet2/get_egs.sh`, from 300 to 20 in order to the [DNN script works properly][2].   



A nice tutorial by [Eleanor Chodroff](https://www.eleanorchodroff.com/tutorial/kaldi/kaldi-training.html) 
might also be worthy taking a look at.


[1]:https://gitlab.com/fb-asr/fb-am-tutorial/demo-corpora.git
[2]:https://groups.google.com/forum/#!msg/kaldi-help/e2EHVCQGE_Y/0uwBkGm9BQAJ
[3]:https://www.eleanorchodroff.com/tutorial/kaldi/

__Grupo FalaBrasil (2018)__   
__Author: Cassio Batista - cassio.batista.13@gmail.com__   
        __Ana Larissa Dias - larissa.engcomp@gmail.com__
