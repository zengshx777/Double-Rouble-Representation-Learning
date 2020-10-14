#
# Requires download and extraction of IHDP_1000, we provide the version we download in the /data
#
mkdir results
mkdir results/results_ihdp_1000

python cfr_param_search.py configs/IHDP_1000_EB.txt 100

python evaluate.py configs/IHDP_1000_EB.txt 1
