stack_CVAE
=============

We designed a novel generative model, Stack-CVAE, that combines stack-RNN with CVAE to generate structural expressions in SMILES. Stack-CVAE is based on CVAE, which produces substances similar to, but not the same as, the substances used for training. The model used to generate molecules in the proposed model was the Stacked Conditional Variation AutoEncoder (Stack-CVAE), which acts as an agent in reinforcement learning so that the resulting chemical formulas have the desired chemical properties and show high binding affinity with specific target proteins.

installation with anaconda
-------------

<pre>
<code>
# Clone the reopsitory to your desired directory
git clone https://github.com/HwanheeKim813/stack_CVAE.git

cd stack_CVAE

# Create new conda environment with Python 3.7
conda create -n stack_CVAE python=3.7
# Activate the environment
conda activate stack_CVAE
# Install conda dependencies
conda install -c rdkit rdkit nox cairo

# Install pip dependencies
pip install -r requirements.txt
pip install git+https://github.com/reymond-group/RAscore.git@master

#install data
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1CvmznJFNiMu_k20MKKtdV0hL4v2KogVe' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1CvmznJFNiMu_k20MKKtdV0hL4v2KogVe" -O data.zip && rm -rf /tmp/cookies.txt
#unzip data
unzip data.zip

</code>
</pre>

Run
-------------
<pre>
<code>
python stackCVAE_rein_train.py
</code>
</pre>
