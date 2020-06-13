# KMC-SVPG
Code for distributed Stein variational policy gradient on a kinetic monte-carlo materials synthesis environment. The paper is available on arxiv.

**Main Dependencies**

mpi4py

horovod

Tensorflow v2

tensorflow_probability 0.9

gym

numpy

matplotlib

**Installation Instructions**

First, make sure you have the core dependencies above. Unzip the two zip files which contain the core KMC engine and the openAI environment. Then, install the kmcsim package: navigate to the kmcsim folder, and from the terminal:

``python setup.py install``

Similarly, install the kmc open ai environment:

``pip install -e . ``

Finally, install the svpg code

``python setup.py install``

**Running the script**

Once installed, from the svpg folder launch

``mpirun -np 16 python run_main.py``

This will run A2C and SVPG for 16 agents on the KMC sim. Change the number of episodes in run_main.py . The core of the code is in svpg/train_rkv where details such as the hyperparameters are set. Work is ongoing to parallelize the simulations so that it will execute an order of magnitude faster than at present; we expect release by the end of 2020. 

Once the code has executed, you can plot the results and test the agents (generating the same figures in the manuscript) via the provided notebook in the folder kmc-openai-env/supporting files/SVPG_KMC_Results_June12_Production.ipynb

