## Create env & run 
The checkpoints are not saved in our training process. The results can be reproduced by running:
>bash run.sh

The test results are saved in the folder _./result_

## Method
![Illustration of Our Framework.](./proc.png)

We aim to predict the unpaired probability by estimating the energy parameters in ViennaRNA.

* Reinforcement learning module: As illustrated in the figure, we first extract features from the RNA sequence and the predicted structure by a LSTM module and a GCN module. Then the energy parameters are predicted from the extracted feature vector. The ViennaRNA predict the unpair probability with the given parameters. To optimize the parameters in the deep learning model, reinforcement learning strategies are used.

* Refining module: In practice, we noticed that few RNA structures cannot be accurately predicted by ViennaRNA even with the optimized energy parameters. We compare the predicted probability (_Prob_v_) by ViennaRNA with the binary value (_Prob_b_) given by LinearFold, and for the sequences whose RMSD(Prob_v, Prob_b) is larger than the threshold, we refine their predicted probability as u*Prob_v+(1-u)*Prob_b.

All the hyperparameters are optimized by linear searching. 

