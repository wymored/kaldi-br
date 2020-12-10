#!/usr/bin/env bash

# _1b is as _1a, but with pitch feats, i-vector and dropout schedule added, referenced from wsj

# basic info:
# steps/info/chain_dir_info.pl exp/chain/tdnn_1f_nopitch_ivec_sp/exp/chain/tdnn_1f_nopitch_ivec_sp/: num-iters=578 nj=2..8 num-params=19.3M dim=43+100->4520 combine=-0.082->-0.081 (over 6) xent:train/valid[384,577,final]=(-0.863,-0.752,-0.740/-0.901,-0.791,-0.784) logprob:train/valid[384,577,final]=(-0.083,-0.076,-0.075/-0.084,-0.077,-0.076)

# results:
# local/chain/compare_wer.sh exp/chain/tdnn_1f_nopitch_ivec_sp/
# Model                tdnn_1f_nopitch_ivec_sp
# Num. of params             19.3M
# WER(%)                     8.81
# Final train prob        -0.0749
# Final valid prob        -0.0756
# Final train prob (xent)   -0.7401
# Final valid prob (xent)   -0.7837

set -e

# configs for 'chain'
affix=all
stage=0
train_stage=-10
get_egs_stage=-10
dir=exp/chain/tdnn_1b  # Note: _sp will get added to this
decode_iter=

# training options
num_epochs=4
initial_effective_lrate=0.001
final_effective_lrate=0.0001
max_param_change=2.0
final_layer_normalize_target=0.5
num_jobs_initial=1  # Cassio
num_jobs_final=1  # Cassio
nj=6  # Cassio
minibatch_size=128
dropout_schedule='0,0@0.20,0.3@0.50,0'
frames_per_eg=150,110,90
remove_egs=false  # Cassio
common_egs_dir=
xent_regularize=0.1

# End configuration section.
echo "$0 $@"  # Print the command line for logging

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

dir=${dir}${affix:+_$affix}_sp
train_set=train
test_sets=test  # Cassio
ali_dir=exp/tri3_ali
treedir=exp/chain/tri4_cd_tree_sp
lang=data/lang_chain

if [ $stage -le 5 ]; then
  mfccdir=mfcc_hires
  for datadir in ${train_set} ${test_sets}; do
    echo "$0: add volume perturbation, compute mfcc and cmvn for $datadir set" | lolcat
    #utils/copy_data_dir.sh \
    fbutils/data/modify_speaker_info.sh --utts-per-spk-max 2 \
      data/${datadir} data/${datadir}_hires
    utils/data/perturb_data_dir_volume.sh data/${datadir}_hires || exit 1;
    #steps/make_mfcc_pitch.sh --nj $nj --pitch-config conf/pitch.conf \
    steps/make_mfcc.sh --nj $nj \
      --mfcc-config conf/mfcc_hires.conf \
      data/${datadir}_hires exp/make_mfcc/ ${mfccdir}
    steps/compute_cmvn_stats.sh data/${datadir}_hires exp/make_mfcc ${mfccdir}
    ##utils/data/limit_feature_dim.sh 0:39 data/${datadir}_hires data/${datadir}_hires_nopitch
    ##steps/compute_cmvn_stats.sh data/${datadir}_hires_nopitch exp/make_mfcc ${mfccdir}
    ## Cassio
    #utils/fix_data_dir.sh data/${datadir}_hires_nopitch
    #utils/validate_data_dir.sh --non-print --no-feats data/${datadir}_hires_nopitch
    utils/fix_data_dir.sh data/${datadir}_hires
    utils/validate_data_dir.sh --non-print --no-feats data/${datadir}_hires
  done
fi

# extract ivector from unified data using the trained
if [ $stage -le 6 ]; then
  # We'll use about a quarter of the data.
  mkdir -p exp/chain/diag_ubm_${affix}
  temp_data_root=exp/chain/diag_ubm_${affix}

  #num_utts_total=$(wc -l < data/${train_set}_hires_nopitch/utt2spk)
  num_utts_total=$(wc -l < data/${train_set}_hires/utt2spk)
  num_utts=$[$num_utts_total/4]
  #utils/data/subset_data_dir.sh data/${train_set}_hires_nopitch \
  utils/data/subset_data_dir.sh data/${train_set}_hires \
    $num_utts ${temp_data_root}/${train_set}_subset

  echo "$0: computing a PCA transform from the hires data." | lolcat
  steps/online/nnet2/get_pca_transform.sh --cmd "$train_cmd" \
    --splice-opts "--left-context=3 --right-context=3" \
    --max-utts 10000 --subsample 2 \
    --dim $(feat-to-dim scp:${temp_data_root}/${train_set}_subset/feats.scp -) \
    ${temp_data_root}/${train_set}_subset \
    exp/chain/pca_transform_${affix}

  echo "$0: training the diagonal UBM." | lolcat
  # Use 512 Gaussians in the UBM.
  steps/online/nnet2/train_diag_ubm.sh --cmd "$train_cmd" --nj $nj \
    --num-frames 700000 \
    --num-threads 4 \
    ${temp_data_root}/${train_set}_subset 512 \
    exp/chain/pca_transform_${affix} exp/chain/diag_ubm_${affix}

  echo "$0: training the iVector extractor" | lolcat
  steps/online/nnet2/train_ivector_extractor.sh \
    --cmd "$train_cmd" --nj $nj --num-processes 2 --num-threads 2 \
    data/${train_set}_hires exp/chain/diag_ubm_${affix} \
    exp/chain/extractor_${affix} || exit 1;
    #data/${train_set}_hires_nopitch exp/chain/diag_ubm_${affix} \

  for datadir in ${train_set} ${test_sets}; do
    echo "$0: extracting iVector online from $datadir set" | lolcat
    #steps/online/nnet2/copy_data_dir.sh --utts-per-spk-max 2 \
    fbutils/data/modify_speaker_info.sh --utts-per-spk-max 2 \
      data/${datadir}_hires data/${datadir}_hires_max2
      #data/${datadir}_hires_nopitch data/${datadir}_hires_nopitch_max2
    steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj $nj \
      data/${datadir}_hires_max2 \
      exp/chain/extractor_${affix} \
      exp/chain/ivectors_${datadir}_${affix} || exit 1;
      #data/${datadir}_hires_nopitch_max2 \
  done
fi

if [ $stage -le 7 ]; then
  # Get the alignments as lattices (gives the LF-MMI training more freedom).
  # use the same num-jobs as the alignments
  echo "$0: get alignments as lattices" | lolcat
  nj=$(cat $ali_dir/num_jobs) || exit 1;
  steps/align_fmllr_lats.sh --nj $nj --cmd "$train_cmd" data/$train_set \
    data/lang exp/tri3 exp/tri4_sp_lats
  rm exp/tri4_sp_lats/fsts.*.gz # save space
fi

if [ $stage -le 8 ]; then
  # Create a version of the lang/ directory that has one state per phone in the
  # topo file. [note, it really has two states.. the first one is only repeated
  # once, the second one has zero or more repeats.]
  echo "$0: gen topo" | lolcat
  rm -rf $lang
  cp -r data/lang $lang
  silphonelist=$(cat $lang/phones/silence.csl) || exit 1;
  nonsilphonelist=$(cat $lang/phones/nonsilence.csl) || exit 1;
  # Use our special topology... note that later on may have to tune this
  # topology.
  steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist >$lang/topo
fi

if [ $stage -le 9 ]; then
  # Build a tree using our new topology. This is the critically different
  # step compared with other recipes.
  echo "$0: building tree" | lolcat
  steps/nnet3/chain/build_tree.sh --frame-subsampling-factor 3 \
    --cmd "$train_cmd" 5000 data/$train_set $lang $ali_dir $treedir
    #--context-opts "--context-width=2 --central-position=1" \
fi

#if [ $stage -le 10 ]; then
#  echo "$0: creating neural net configs using the xconfig parser" | lolcat
#  feat_dim=$(feat-to-dim scp:data/${train_set}_hires/feats.scp -)
#  num_targets=$(tree-info $treedir/tree | grep num-pdfs | awk '{print $2}')
#  learning_rate_factor=$(echo "print (0.5/$xent_regularize)" | python)
#  opts="l2-regularize=0.002"
#  linear_opts="orthonormal-constraint=1.0"
#  output_opts="l2-regularize=0.0005 bottleneck-dim=96" 
#  #output_opts="l2-regularize=0.0005 bottleneck-dim=256" 
#
#  mkdir -p $dir/configs
#  cat <<EOF > $dir/configs/network.xconfig
#  input dim=100 name=ivector
#  input dim=$feat_dim name=input
#
#  # please note that it is important to have input layer with the name=input
#  # as the layer immediately preceding the fixed-affine-layer to enable
#  # the use of short notation for the descriptor
#  fixed-affine-layer name=lda input=Append(-1,0,1,ReplaceIndex(ivector, t, 0)) affine-transform-file=$dir/configs/lda.mat
#
#  # the first splicing is moved before the lda layer, so no splicing here
#  relu-batchnorm-dropout-layer name=tdnn1 $opts dim=1280
#  linear-component name=tdnn2l dim=256 $linear_opts input=Append(-1,0)
#  relu-batchnorm-dropout-layer name=tdnn2 $opts input=Append(0,1) dim=1280
#  linear-component name=tdnn3l dim=256 $linear_opts
#  relu-batchnorm-dropout-layer name=tdnn3 $opts dim=1280
#  linear-component name=tdnn4l dim=256 $linear_opts input=Append(-1,0)
#  relu-batchnorm-dropout-layer name=tdnn4 $opts input=Append(0,1) dim=1280
#  linear-component name=tdnn5l dim=256 $linear_opts
#  relu-batchnorm-dropout-layer name=tdnn5 $opts dim=1280 input=Append(tdnn5l, tdnn3l)
#  linear-component name=tdnn6l dim=256 $linear_opts input=Append(-3,0)
#  relu-batchnorm-dropout-layer name=tdnn6 $opts input=Append(0,3) dim=1280
#  linear-component name=tdnn7l dim=256 $linear_opts input=Append(-3,0)
#  relu-batchnorm-dropout-layer name=tdnn7 $opts input=Append(0,3,tdnn6l,tdnn4l,tdnn2l) dim=1280
#  linear-component name=tdnn8l dim=256 $linear_opts input=Append(-3,0)
#  relu-batchnorm-dropout-layer name=tdnn8 $opts input=Append(0,3) dim=1280
#  linear-component name=tdnn9l dim=256 $linear_opts input=Append(-3,0)
#  relu-batchnorm-dropout-layer name=tdnn9 $opts input=Append(0,3,tdnn8l,tdnn6l,tdnn4l) dim=1280
#  linear-component name=tdnn10l dim=256 $linear_opts input=Append(-3,0)
#  relu-batchnorm-dropout-layer name=tdnn10 $opts input=Append(0,3) dim=1280
#  linear-component name=tdnn11l dim=256 $linear_opts input=Append(-3,0)
#  relu-batchnorm-dropout-layer name=tdnn11 $opts input=Append(0,3,tdnn10l,tdnn8l,tdnn6l) dim=1280
#  linear-component name=prefinal-l dim=256 $linear_opts
#
#  relu-batchnorm-layer name=prefinal-chain input=prefinal-l $opts dim=1280
#  output-layer name=output include-log-softmax=false dim=$num_targets $output_opts
#
#  relu-batchnorm-layer name=prefinal-xent input=prefinal-l $opts dim=1280
#  output-layer name=output-xent dim=$num_targets learning-rate-factor=$learning_rate_factor $output_opts
#
#EOF
#  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
#fi

if [ $stage -le 10 ]; then
  echo "[$(date +'%F %T')] $0: creating neural net configs using the xconfig parser" | lolcat

  num_targets=$(tree-info $treedir/tree | grep num-pdfs | awk '{print $2}')
  feat_dim=$(feat-to-dim scp:data/${train_set}_hires/feats.scp -)
  learning_rate_factor=$(echo "print (0.5/$xent_regularize)" | python)
  #opts="l2-regularize=0.002"
  #linear_opts="orthonormal-constraint=1.0"
  #output_opts="l2-regularize=0.0005 bottleneck-dim=96" 
  ##output_opts="l2-regularize=0.0005 bottleneck-dim=256" 

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=100 name=ivector
  input dim=$feat_dim name=input

  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor
  fixed-affine-layer name=lda input=Append(-1,0,1,ReplaceIndex(ivector, t, 0)) affine-transform-file=$dir/configs/lda.mat

  # the first splicing is moved before the lda layer, so no splicing here
  relu-batchnorm-layer name=tdnn1 dim=1024
  relu-batchnorm-layer name=tdnn2 input=Append(-1,0,1,2) dim=1024
  relu-batchnorm-layer name=tdnn3 input=Append(-3,0,3) dim=1024
  relu-batchnorm-layer name=tdnn4 input=Append(-3,0,3) dim=1024
  relu-batchnorm-layer name=tdnn5 input=Append(-3,0,3) dim=1024
  relu-batchnorm-layer name=tdnn6 input=Append(-6,-3,0) dim=1024

  ## adding the layers for chain branch
  relu-batchnorm-layer name=prefinal-chain input=tdnn6 dim=1024 target-rms=0.5
  output-layer name=output include-log-softmax=false dim=$num_targets max-change=1.5

  # adding the layers for xent branch
  # This block prints the configs for a separate output that will be
  # trained with a cross-entropy objective in the 'chain' models... this
  # has the effect of regularizing the hidden parts of the model.  we use
  # 0.5 / args.xent_regularize as the learning rate factor- the factor of
  # 0.5 / args.xent_regularize is suitable as it means the xent
  # final-layer learns at a rate independent of the regularization
  # constant; and the 0.5 was tuned so as to make the relative progress
  # similar in the xent and regular final layers.
  relu-batchnorm-layer name=prefinal-xent input=tdnn6 dim=1024 target-rms=0.5
  output-layer name=output-xent dim=$num_targets learning-rate-factor=$learning_rate_factor max-change=1.5

EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi


if [ $stage -le 11 ]; then
  echo "$0: training dnn" | lolcat
  steps/nnet3/chain/train.py --stage $train_stage \
    --cmd "$decode_cmd" \
    --feat.online-ivector-dir exp/chain/ivectors_${train_set}_${affix} \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.00005 \
    --chain.apply-deriv-weights false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --egs.dir "$common_egs_dir" \
    --egs.stage $get_egs_stage \
    --egs.opts "--frames-overlap-per-eg 0" \
    --egs.chunk-width 150 \
    --trainer.dropout-schedule $dropout_schedule \
    --trainer.num-chunk-per-minibatch $minibatch_size \
    --trainer.frames-per-iter 1500000 \
    --trainer.num-epochs $num_epochs \
    --trainer.optimization.num-jobs-initial $num_jobs_initial \
    --trainer.optimization.num-jobs-final $num_jobs_final \
    --trainer.optimization.initial-effective-lrate $initial_effective_lrate \
    --trainer.optimization.final-effective-lrate $final_effective_lrate \
    --trainer.max-param-change $max_param_change \
    --cleanup.remove-egs $remove_egs \
    --use-gpu=wait \
    --feat-dir data/${train_set}_hires \
    --tree-dir $treedir \
    --lat-dir exp/tri4_sp_lats \
    --dir $dir  || exit 1;
fi

if [ $stage -le 12 ]; then
  # Note: it might appear that this $lang directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  echo "$0: creating dnn graph" | lolcat
  utils/mkgraph.sh --self-loop-scale 1.0 data/lang_test $dir $dir/graph
fi

graph_dir=$dir/graph
if [ $stage -le 13 ]; then
  echo "$0: decoding dnn" | lolcat
  for test_set in $test_sets; do
    steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
      --nj $nj --cmd "$decode_cmd" \
      --online-ivector-dir exp/chain/ivectors_${test_set}_${affix} \
      $graph_dir data/${test_set}_hires $dir/decode_${test_set} || exit 1;
  done
fi

# NOTE: created by Cassio
# this step is equivalent to stage 17 of mini librispeech recipe
# The step by step guide is on the README files that comes gzipped with the
# aspire pretrained available on Kaldi's website: https://kaldi-asr.org/models/m1
# #Iamnotmakingthisup
if [ $stage -le 14 ] ; then
  echo "$0: prepare online decoding" | lolcat
  # FIXME no online pitch?? - Cassio
  steps/online/nnet3/prepare_online_decoding.sh \
    --mfcc-config conf/mfcc_hires.conf \
    --online-cmvn-config conf/online_cmvn.conf \
    $lang exp/chain/extractor_all $dir ${dir}_chain_online

  echo "$0: creating graph" | lolcat
  utils/mkgraph.sh --self-loop-scale 1.0 \
    data/lang_test ${dir}_chain_online ${dir}_chain_online/graph_pp

  echo "$0: online decode" | lolcat
  # note: we just give it "data/${data}" as it only uses the wav.scp, the
  # feature type does not matter.
  for test_set in $test_sets; do
    steps/online/nnet3/decode.sh \
      --acwt 1.0 --post-decode-acwt 10.0 --nj $nj --cmd "$decode_cmd" \
      ${dir}_chain_online/graph_pp \
      data/${test_set}_hires \
      ${dir}_chain_online/decode_${test_set}_hires || exit 1
  done
fi

echo "$0: success"
