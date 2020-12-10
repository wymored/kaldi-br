#!/usr/bin/env bash

# Copyright 2018 AIShell-Foundation(Authors:Jiayu DU, Xingyu NA, Bengu WU, Hao ZHENG)
#           2018 Beijing Shell Shell Tech. Co. Ltd. (Author: Hui BU)
# Apache 2.0

# AISHELL-2 provides:
#  * a Mandarin speech corpus (~1000hrs), free for non-commercial research/education use
#  * a baseline recipe setup for large scale Mandarin ASR system
# For more details, read $KALDI_ROOT/egs/aishell2/README.txt

data_url=https://gitlab.com/fb-audio-corpora/lapsbm16k/-/archive/master/lapsbm16k-master.tar.gz
lex_url=https://gitlab.com/fb-nlp/nlp-resources/-/raw/master/res/lexicon.utf8.dict.gz

data=corpus

nj=6
stage=0
gmm_stage=1
dnn_stage=1

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

if [ $stage -le 0 ]; then
  echo "[$(date +'%F %T')] $0: download data (85M)" | lolcat
  fblocal/download_data.sh $data $data_url || exit 1
  #fblocal/link_local_data.sh --nj 8 ${HOME}/fb-gitlab/fb-audio-corpora $data || exit 1

  echo "[$(date +'%F %T')] $0: download lexicon" | lolcat
  fblocal/download_lexicon.sh $data $lex_url data/local/dict || exit 1
fi

# prepare trn/dev/tst data, lexicon, lang etc
if [ $stage -le 1 ]; then
  # convert to Kaldi lexicon format
  echo "$0: prep dict" | lolcat
  fblocal/prepare_dict.sh data/local/dict || exit 1
  utils/validate_dict_dir.pl data/local/dict || exit 1

  # wav.scp, text(word-segmented), utt2spk, spk2utt
  echo "$0: prep data" | lolcat
  fblocal/prep_data.sh --nj $nj --split-random true $data data
  utils/validate_data_dir.sh --non-print --no-feats data/train || exit 1
  utils/validate_data_dir.sh --non-print --no-feats data/test  || exit 1

  # arpa LM
  echo "$0: training arpa lm" | lolcat
  local/train_lms.sh data/local/dict/lexicon.txt data/train/text data/local/lm || exit 1

  # L
  echo "$0: prep lang: compile L" | lolcat
  utils/prepare_lang.sh --position-dependent-phones false \
    data/local/dict "<UNK>" data/local/lang data/lang || exit 1

  # G compilation, check LG composition
  echo "$0: format lm: compile G, check LG composition" | lolcat
  utils/format_lm.sh data/lang data/local/lm/3gram-mincount/lm_unpruned.gz \
    data/local/dict/lexicon.txt data/lang_test || exit 1
fi

# GMM
if [ $stage -le 2 ]; then
  fblocal/run_gmm.sh --nj $nj --stage $gmm_stage --decode true || exit 1
fi

# chain
if [ $stage -le 3 ]; then
  fblocal/chain/run_tdnn.sh --nj $nj --stage $dnn_stage
fi

##local/show_results.sh
##
##exit 0;
