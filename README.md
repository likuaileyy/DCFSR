# DCFSR

In target detection sensor systems, the accurate association of interrupted trajectories has emerged as a critical challenge in the field of target tracking due to their complex spatio-temporal characteristics. To address the limitations of existing deep learning methods, which suffer from insufficient kinematic constraint modeling, this letter proposes an innovative Dual-Constraint Framework Integrating Similarity and Rationality-Based Interrupted Trajectory Association (DCFSR). The framework integrates two key elements: “morphological trajectory similarity” and “interruption motion rationality”. By embedding the physical laws of interrupted motion into the training process and employing a dual-constraint collaborative verification mechanism, DCFSR achieves morphological-kinematic modeling, significantly enhancing the model's cognitive capabilities in interrupted scenarios. Experimental results on the Danish Maritime Interrupted Trajectory Dataset demonstrate a maximum 22.4\% improvement in the average F1 score, providing an interpretable and highly robust solution for trajectory association in complex environments. 

# Train.py is the training file; test.py is the testing file; dcfsr_model.py contains the model source code.
# Training Method:
1) Open Train.py, every time it runs, a weight file is generated. Set the 'dir' (dataset path) at line 334; set 'model_save_path' (output file path) at line 336; and specify the 'model' (model algorithm) at line 345.
2) Test.py performs multiple methods with 50 Monte Carlo experiments unifiedly. Set the 'dir' (dataset path) at line 542; add output result paths to 'result_dirs', where the number of 'result_dirs' determines the number of methods; and add weight file paths to 'models'. 

This translation covers the instructions on how to configure and use the training and testing scripts as described in your original text.
