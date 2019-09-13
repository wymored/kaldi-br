#!/bin/bash
#
# Cassio Batista   - https://cassota.gitlab.io/
# Ana Larissa Dias - larissa.engcomp@gmail.com

TAG="DNN-iVec"

function usage() {
    echo "usage: (bash) $0 OPTIONS"
    echo "eg.: $0 --nj 2 --use_gpu false"
    echo ""
    echo "OPTIONS"
    echo "  --nj         number of parallel jobs  "
    echo "  --use_gpu    specifies whether run on GPU or on CPU  "
}

if test $# -eq 0 ; then
    usage
    exit 1
fi

while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
        --nj)
            nj="$2"
            shift # past argument
            shift # past value
        ;;
        --use_gpu)
            use_gpu="$2"
            shift # past argument
            shift # past value
        ;;
        *)  # unknown option
            echo "[$TAG] unknown flag $1"
            shift # past argument
            exit 0
        ;;
    esac
done

if [[ -z $nj || -z $use_gpu ]] ; then
    echo "[$TAG] a problem with the arg flags has been detected"
    exit 1
fi

#This script is a modified version of the ../rm/s5/local/online/run_nnet2.sh that trains the DNN model with iVectors to online decoding.

stage=1
train_stage=-10
use_gpu=false
dir=exp/nnet2_online/nnet_a


#DNN parameters 
minibatch_size=512
num_epochs=8 
num_epochs_extra=5 
num_hidden_layers=2
initial_learning_rate=0.02 
final_learning_rate=0.004
pnorm_input_dim=3000 
pnorm_output_dim=300

#DNN parameters for small data
#pnorm_input_dim=1000 
#pnorm_output_dim=200


. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

if $use_gpu; then
  if ! cuda-compiled; then
    cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.  Otherwise, call this script with --use-gpu false
EOF
  fi
  parallel_opts="--gpu 1"
  num_threads=1
else
  # Use 4 nnet jobs just like run_4d_gpu.sh so the results should be
  # almost the same, but this may be a little bit slow.
  num_threads=$nj
  parallel_opts="--num-threads $num_threads"
fi

echo
echo "============== [$TAG] DNN WITH iVECTORS TRAINING =============="
echo

# stages 1 through 3 run in run_nnet2_common.sh.
local/online/run_nnet2_common.sh --stage  $stage || exit 1;


if [ $stage -le 4 ]; then
  steps/nnet2/train_pnorm_simple2.sh --stage $train_stage \
    --splice-width 7 \
    --feat-type raw \
    --online-ivector-dir exp/nnet2_online/ivectors \
    --cmvn-opts "--norm-means=false --norm-vars=false" \
    --num-threads "$num_threads" \
    --minibatch-size "$minibatch_size" \
    --parallel-opts "$parallel_opts" \
    --num-jobs-nnet 4 \
    --num-epochs $num_epochs \
    --add-layers-period 1 \
    --num-hidden-layers $num_hidden_layers \
    --mix-up 4000 \
    --initial-learning-rate $initial_learning_rate \
    --final-learning-rate $final_learning_rate \
    --cmd "$decode_cmd" \
    --pnorm-input-dim $pnorm_input_dim \
    --pnorm-output-dim $pnorm_output_dim \
    data/train data/lang exp/tri3b_ali $dir  || exit 1;
fi

if [ $stage -le 5 ]; then
  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj $nj \
    data/test exp/nnet2_online/extractor exp/nnet2_online/ivectors_test || exit 1;
fi

if [ $stage -le 7 ]; then
  # If this setup used PLP features, we'd have to give the option --feature-type plp
  # to the script below.
  steps/online/nnet2/prepare_online_decoding.sh data/lang exp/nnet2_online/extractor \
    "$dir" ${dir}_online || exit 1;
fi

if [ $stage -le 8 ]; then
  # do the actual online decoding with iVectors.
  steps/online/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj $nj \
    exp/tri3b/graph data/test ${dir}_online/decode &
  wait
fi

echo
echo "============== [$TAG] FINISHED RUNNING DNN WITH iVECTORS =============="
echo
exit 0;
