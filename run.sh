# create env
conda create -n=UPP python=3.7.9 -y
source activate UPP
pip install loguru
conda install -c bioconda viennarna -y
conda install -c paddle paddlepaddle-gpu -y
pip install networkx
pip install h5py

# run rl to optimize energy parameter. Might take hours
CUDA_VISIBLE_DEVICES=0 python main.py -c tasks.rl.config

# run refining 
mkdir tasks/rl/final_trial
python refine.py
cp -r tasks/rl/final_trial/ result/