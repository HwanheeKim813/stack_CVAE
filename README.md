stack_CVAE
=============

We designed a novel generative model, Stack-CVAE, that combines stack-RNN with CVAE to generate structural expressions in SMILES. Stack-CVAE is based on CVAE, which produces substances similar to, but not the same as, the substances used for training. The model used to generate molecules in the proposed model was the Stacked Conditional Variation AutoEncoder (Stack-CVAE), which acts as an agent in reinforcement learning so that the resulting chemical formulas have the desired chemical properties and show high binding affinity with specific target proteins.

REQUIREMENTS:
-------------
    pip install -r requirements.txt

installation with anaconda
-------------

<pre>
<code>
# Clone the reopsitory to your desired directory
git clone https://github.com/HwanheeKim813/stack_CVAE.git
cd stack_CVAE
# Create new conda environment with Python 3.6
conda create -n release python=3.6
# Activate the environment
conda activate stack_CVAE
# Install conda dependencies
conda install -c rdkit rdkit nox cairo
conda install pytorch=1.1.0 torchvision=0.2.1 -c pytorch
# Instal pip dependencies
pip install requirements.txt

#install data
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1CHlXD2wyvMRxg6uI7TtDi0g4jJSwh9zG' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1CHlXD2wyvMRxg6uI7TtDi0g4jJSwh9zG" -O data.zip && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1_1k5D_btqr14_WdzfPPwKNxEJYjw9SDn' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1_1k5D_btqr14_WdzfPPwKNxEJYjw9SDn" -O data_ppi.zip && rm -rf /tmp/cookies.txt
#unzip data
unzip data.zip
unzip data_ppi.zip

</code>
</pre>

Run
-------------
<pre>
<code>
python stack_CVAE_BA_RA_rein.py
</code>
</pre>
