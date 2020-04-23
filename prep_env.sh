#!/bin/bash
#
# Create environment tree for training acoustic models with Kaldi
# This scripts used to be called 'fb_00' in the old days.
#
# Grupo FalaBrasil (2020)
# Federal University of Pará (UFPA)
#
# Author: Apr 2020
# Cassio Batista - https://cassota.gitlab.io/

echo -e "\033[94m  ____                         \033[93m _____     _           \033[0m"
echo -e "\033[94m / ___| _ __ _   _ _ __   ___  \033[93m|  ___|_ _| | __ _     \033[0m"
echo -e "\033[94m| |  _ | '__| | | | '_ \ / _ \ \033[93m| |_ / _\` | |/ _\` |  \033[0m"
echo -e "\033[94m| |_| \| |  | |_| | |_) | (_) |\033[93m|  _| (_| | | (_| |    \033[0m"
echo -e "\033[94m \____||_|   \__,_| .__/ \___/ \033[93m|_|  \__,_|_|\__,_|    \033[0m"
echo -e "                  \033[94m|_|      \033[32m ____                _ _\033[0m\033[91m  _   _ _____ ____    _   \033[0m"
echo -e "                           \033[32m| __ ) _ __ __ _ ___(_) |\033[0m\033[91m| | | |  ___|  _ \  / \          \033[0m"
echo -e "                           \033[32m|  _ \| '_ / _\` / __| | |\033[0m\033[91m| | | | |_  | |_) |/ ∆ \        \033[0m"
echo -e "                           \033[32m| |_) | | | (_| \__ \ | |\033[0m\033[91m| |_| |  _| |  __// ___ \        \033[0m"
echo -e "                           \033[32m|____/|_|  \__,_|___/_|_|\033[0m\033[91m \___/|_|   |_|  /_/   \_\       \033[0m"
echo -e ""

if test $# -ne 1 ; then
    echo "A script to create the environment tree for training acoustic models"
    echo "according to Kaldi's pattern."
    echo "Ref.: http://kaldi-asr.org/doc/kaldi_for_dummies.html"
    echo "Ref.: https://www.eleanorchodroff.com/tutorial/kaldi/kaldi-training.html"
    echo
    echo "Usage: $0 <proj_dir>"
    echo "  <proj_dir> is the path for your project *within* kaldi/egs dir."
    echo "      e.g.: ./$0 ${HOME}/kaldi/egs/MEUPROJETO"
    exit 1
elif [ -d "$1" ] ; then
    echo -n "$0: warning: '$1' exists as dir. Override? [y/N] "
    read ans
    if [ "$ans" != "y" ] ; then
        echo "$0: aborted."
        exit 0
    else
        rm -rf ${1}/s5/{data,exp,mfcc}
    fi
# https://stackoverflow.com/questions/8426058/getting-the-parent-of-a-directory-in-bash
elif [ "$(basename $(readlink -f $(dirname "$1")))" != "egs" ] ; then
    echo "$0: Error: '$1' must be inside /path/to/kaldi/egs"
    exit 1
fi

PROJECT_DIR="$(readlink -f "$1")"/s5
KALDI_ROOT="$(readlink -f $(dirname "$(dirname "$1")"))"
MINILIBRI_DIR=${KALDI_ROOT}/egs/mini_librispeech/s5

mkdir -p $PROJECT_DIR/
cp -r fbegs/* $PROJECT_DIR/

ln -sf ${MINILIBRI_DIR}/conf/  ${PROJECT_DIR}/ 
ln -sf ${MINILIBRI_DIR}/local/ ${PROJECT_DIR}/
ln -sf ${MINILIBRI_DIR}/steps/ ${PROJECT_DIR}/
ln -sf ${MINILIBRI_DIR}/utils/ ${PROJECT_DIR}/

ln -sf ${MINILIBRI_DIR}/path.sh ${PROJECT_DIR}/path.sh
sed 's/"queue.pl/"run.pl/g' ${MINILIBRI_DIR}/cmd.sh > ${PROJECT_DIR}/cmd.sh

echo "$0: all set up! check out your project at '$(readlink -f $PROJECT_DIR)'"
tree $PROJECT_DIR -I corpus

exit 0