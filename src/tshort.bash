#!/bin/bash -l
#SBATCH -N 1
#SBATCH -c 12
#SBATCH -p short           # Use short partition
#SBATCH --time=18:00:00
#SBATCH --output=log/%j.output
#SBATCH --error=log/%j.error

nvidia-smi
module load anaconda3/2022.05
module load ffmpeg/20190305 

source activate /work/van-speech-nlp/jindaznb/visenv/

# 43390064
# python 6_3_g2p_N_best_phoneme_pt_comma.py --speaker_id M05


# 43376021
# python 6_3_g2p_N_best_phoneme_pt_comma.py --speaker_id M01
# python 6_3_g2p_N_best_phoneme_pt_comma.py --speaker_id M02

# 43376025
# python 6_3_g2p_N_best_phoneme_pt_comma.py --speaker_id M03
# python 6_3_g2p_N_best_phoneme_pt_comma.py --speaker_id M04
# python 6_3_g2p_N_best_phoneme_pt_comma.py --speaker_id F01

# 43376045
# python 6_3_g2p_N_best_phoneme_pt_comma.py --speaker_id F03
# python 6_3_g2p_N_best_phoneme_pt_comma.py --speaker_id F04




# python 6_2_g2p_phoneme_pt.py --speaker_id F04

# 43342796
# python 6_2_g2p_phoneme_pt.py --speaker_id M01

# 43342816
# python 6_2_g2p_phoneme_pt.py --speaker_id M02


# 43342817
# python 6_2_g2p_phoneme_pt.py --speaker_id M03

# 43342818
# python 6_2_g2p_phoneme_pt.py --speaker_id M04

# 43342823
# python 6_2_g2p_phoneme_pt.py --speaker_id M05