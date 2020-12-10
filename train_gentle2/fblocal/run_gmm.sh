#!/usr/bin/env bash
# Copyright 2018 AIShell-Foundation(Authors:Jiayu DU, Xingyu NA, Bengu WU, Hao ZHENG)
#           2018 Beijing Shell Shell Tech. Co. Ltd. (Author: Hui BU)
#           2018 Emotech LTD (Author: Xuechen LIU)
# Apache 2.0

set -e

# number of jobs
nj=6
stage=1
decode=true

. ./cmd.sh
[ -f ./path.sh ] && . ./path.sh;
. ./utils/parse_options.sh

# Now make MFCC features.
if [ $stage -le 1 ]; then
  # mfccdir should be some place with a largish disk where you
  # want to store MFCC features.
  echo "$0: compute mfcc and cmvn" | lolcat
  for part in train test; do
    steps/make_mfcc_pitch.sh \
      --pitch-config conf/pitch.conf --cmd "$train_cmd" --nj $nj \
      data/$part exp/make_mfcc/$part mfcc || exit 1
    steps/compute_cmvn_stats.sh data/$part exp/make_mfcc/$part mfcc || exit 1
    utils/fix_data_dir.sh data/$part || exit 1
  done

  ## subset the training data for fast startup
  #for x in 100 300; do
  #  utils/subset_data_dir.sh data/train ${x}000 data/train_${x}k
  #done
  utils/subset_data_dir.sh data/train 200 data/train_100k  # Cassio
  utils/subset_data_dir.sh data/train 400 data/train_300k  # Cassio
fi

# mono
if [ $stage -le 2 ]; then
  # training
  echo "$0: training mono" | lolcat
  steps/train_mono.sh --cmd "$train_cmd" --nj $nj \
    data/train_100k data/lang exp/mono || exit 1

  # decoding
  if $decode ; then
    echo "$0: creating mono graph" | lolcat
    utils/mkgraph.sh data/lang_test exp/mono exp/mono/graph || exit 1
    echo "$0: decoding mono" | lolcat
    steps/decode.sh --cmd "$decode_cmd" --config conf/decode.conf --nj $nj \
      exp/mono/graph data/test exp/mono/decode_test
  fi

  # alignment
  echo "$0: aligning mono" | lolcat
  steps/align_si.sh --cmd "$train_cmd" --nj $nj \
    data/train_300k data/lang exp/mono exp/mono_ali || exit 1
fi

# tri1
if [ $stage -le 3 ]; then
  # training
  echo "$0: training tri deltas (1st pass)" | lolcat
  steps/train_deltas.sh --cmd "$train_cmd" \
   2500 20000 data/train_300k data/lang exp/mono_ali exp/tri1 || exit 1
   #4000 32000 data/train_300k data/lang exp/tri deltas_ali exp/tri1 || exit 1

  # decoding
  if $decode ; then
    echo "$0: creating tri deltas graph (1st pass)" | lolcat
    utils/mkgraph.sh data/lang_test exp/tri1 exp/tri1/graph || exit 1
    echo "$0: decoding tri deltas (1st pass)" | lolcat
    steps/decode.sh --cmd "$decode_cmd" --config conf/decode.conf --nj $nj \
      exp/tri1/graph data/test exp/tri1/decode_test
  fi

  # alignment
  echo "$0: aligning tri deltas (1st pass)" | lolcat
  steps/align_si.sh --cmd "$train_cmd" --nj $nj \
    data/train data/lang exp/tri1 exp/tri1_ali || exit 1
fi

# tri2
if [ $stage -le 4 ]; then
  # training
  echo "$0: training tri deltas (2nd pass)" | lolcat
  steps/train_deltas.sh --cmd "$train_cmd" \
   2500 20000 data/train data/lang exp/tri1_ali exp/tri2 || exit 1
   #7000 56000 data/train data/lang exp/tri1_ali exp/tri2 || exit 1

  # decoding
  if $decode ; then
    echo "$0: creating tri deltas (2nd pass) graph" | lolcat
    utils/mkgraph.sh data/lang_test exp/tri2 exp/tri2/graph
    echo "$0: decoding tri deltas (2nd pass)" | lolcat
    steps/decode.sh --cmd "$decode_cmd" --config conf/decode.conf --nj $nj \
      exp/tri2/graph data/test exp/tri2/decode_test
  fi

  # alignment
  echo "$0: aligning tri deltas (2nd pass)" | lolcat
  steps/align_si.sh --cmd "$train_cmd" --nj $nj \
    data/train data/lang exp/tri2 exp/tri2_ali || exit 1
fi

# tri3
if [ $stage -le 5 ]; then
  # training [LDA+MLLT]
  echo "$0: training tri lda mllt" | lolcat
  steps/train_lda_mllt.sh --cmd "$train_cmd" \
    --splice-opts "--left-context=3 --right-context=3" \
   5000 40000 data/train data/lang exp/tri2_ali exp/tri3 || exit 1
   #10000 80000 data/train data/lang exp/tri2_ali exp/tri3 || exit 1

  # decoding
  if $decode ; then
    echo "$0: creating tri lda mllt graph" | lolcat
    utils/mkgraph.sh data/lang_test exp/tri3 exp/tri3/graph || exit 1
    echo "$0: decoding tri lda mllt" | lolcat
    steps/decode.sh --cmd "$decode_cmd" --nj $nj --config conf/decode.conf \
      exp/tri3/graph data/test exp/tri3/decode_test
  fi

  # alignment
  echo "$0: aligning tri lda mllt" | lolcat
  steps/align_si.sh --cmd "$train_cmd" --nj $nj \
    data/train data/lang exp/tri3 exp/tri3_ali || exit 1
  #steps/align_fmllr.sh --nj 5 --cmd "$train_cmd" \
  #  data/train data/lang exp/tri3 exp/tri3_ali
fi

## tri4
#if [ $stage -le 6 ]; then
#  # training [LDA+MLLT+SAT]
#  echo "$0: training sat (1st pass)" | lolcat
#  steps/train_sat.sh --cmd "$train_cmd" \
#   5000 100000 data/train data/lang exp/tri3_ali exp/tri4 || exit 1
#
#  # decoding
#  if $decode ; then
#    echo "$0: creating tri sat graph (1st pass)" | lolcat
#    utils/mkgraph.sh data/lang_test exp/tri4 exp/tri4/graph || exit 1
#    echo "$0: decoding tri sat (1st pass)" | lolcat
#    steps/decode.sh --cmd "$decode_cmd" --nj $nj --config conf/decode.conf \
#      exp/tri4/graph data/test exp/tri4/decode_test
#  fi
#
#  # alignment
#  echo "$0: aligning sat (1st pass)" | lolcat
#  steps/align_fmllr.sh --cmd "$train_cmd" --nj $nj \
#    data/train data/lang exp/tri4 exp/tri4_ali || exit 1
#fi
#
## tri5
#if [ $stage -le 7 ]; then
#  # training [LDA+MLLT+SAT]
#  echo "$0: training sat (2nd pass)" | lolcat
#  steps/train_sat.sh --cmd "$train_cmd" \
#   10000 300000 data/train data/lang exp/tri4_ali exp/tri5 || exit 1
#
#  # decoding
#  if $decode ; then
#    echo "$0: creating tri sat graph (2nd pass)" | lolcat
#    utils/mkgraph.sh data/lang_test exp/tri5 exp/tri5/graph || exit 1
#    echo "$0: decoding tri sat (2nd pass)" | lolcat
#    steps/decode.sh --cmd "$decode_cmd" --nj $nj --config conf/decode.conf \
#      exp/tri5/graph data/test exp/tri5/decode_test
#  fi
#
#  ## alignment
#  #echo "$0: aligning sat (2nd pass)" | lolcat
#  #steps/align_fmllr.sh --cmd "$train_cmd" --nj $nj \
#  #  data/train data/lang exp/tri5 exp/tri5_ali || exit 1
#fi

echo "$0: success"
