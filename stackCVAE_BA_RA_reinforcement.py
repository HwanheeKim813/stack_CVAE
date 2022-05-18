import torch
import torch.nn.functional as F
import numpy as np
from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.rdMolDescriptors import CalcTPSA

class BA_Reinforcement_3(object):
    def __init__(self, generator, get_reward,
                 target_pSeq_dict,groupD_pSeq_dict,side_pSeq_dict, ppi_graph, dis_Prot_list,
                 DeepPurpose_model, SAscore_model, minPSeqLen=1000):

        super(BA_Reinforcement_3, self).__init__()
        self.generator = generator
        self.get_reward = get_reward		
        self.ppi_graph = ppi_graph
        self.target_pSeq_dict = target_pSeq_dict
        self.groupD_pSeq_dict = groupD_pSeq_dict
        self.side_pSeq_dict = side_pSeq_dict
        self.dis_Prot_list = dis_Prot_list
        self.DeepPurpose_model = DeepPurpose_model
        self.SAscore_model = SAscore_model
        self.minPSeqLen = minPSeqLen

    def policy_gradient(self, data, n_batch=10, gamma=0.97,
                        std_smiles=False, grad_clipping=None, **kwargs):

        # allow BP
        self.generator.cuda()

        rl_loss = 0
        self.generator.optimizer.zero_grad()
        total_reward = 0
        for _ in range(n_batch):
            reward = 0

            trajectory = '<>'
            while reward == 0:
                props = [np.random.normal(421.59,40,size=1).item(),np.random.normal(3.321,0.3,size=1).item(),np.random.normal(91.11,9,size=1).item()]
                trajectory, latent_vector = self.generator.evaluate(data, prop=props)

                if std_smiles:
                    try:
                        mol = Chem.MolFromSmiles(trajectory[1:-1])
                        trajectory = '<' + Chem.MolToSmiles(mol) + '>'
                        reward = self.get_reward(trajectory[1:-1], 
                                                 self.target_pSeq_dict,
                                                 self.groupD_pSeq_dict,
                                                 self.side_pSeq_dict,
                                                 self.ppi_graph,
                                                 self.dis_Prot_list,
                                                 self.DeepPurpose_model,
                                                 self.SAscore_model,
                                                 self.minPSeqLen,
                                                 **kwargs)
                        prop = [ExactMolWt(mol), MolLogP(mol), CalcTPSA(mol)]

                    except:
                        reward = 0
                else:
                    mol = Chem.MolFromSmiles(trajectory[1:-1])
                    reward = self.get_reward(trajectory[1:-1], 
                                                 self.target_pSeq_dict,
                                                 self.groupD_pSeq_dict,
                                                 self.side_pSeq_dict,
                                                 self.ppi_graph,
                                                 self.dis_Prot_list,
                                                 self.DeepPurpose_model,
                                                 self.SAscore_model,
                                                 self.minPSeqLen,
                                                 **kwargs)
                    prop = [ExactMolWt(mol), MolLogP(mol), CalcTPSA(mol)]
            trajectory_input = data.char_tensor(trajectory)
            discounted_reward = reward
            total_reward += reward


            if self.generator.layer_type == 'LSTM':
                encoder_hidden = (self.generator.init_hidden(1),self.generator.init_hidden(1))
                decoder_hidden = (self.generator.init_hidden(1),self.generator.init_hidden(1))
              
            elif self.generator.layer_type == 'GRU':
                encoder_hidden = self.generator.init_hidden(1)
                decoder_hidden = self.generator.init_hidden(1)
            if self.generator.has_stack:
                encoder_stack = self.generator.init_stack(1)
                decoder_stack = self.generator.init_stack(1)	
            else:
                stack = None

            prop = torch.tensor(prop).cuda()
            latent_vector, KLD = self.generator.encoder(trajectory_input[:-1],trajectory_input[1:],prop,encoder_hidden,encoder_stack)

            for p in range(len(trajectory)-1):
                output, decoder_hidden = self.generator.sample(latent_vector,trajectory_input[p],trajectory_input[p+1],prop,decoder_hidden)
                log_probs = F.log_softmax(output, dim=1)
                top_i = trajectory_input[p+1]
                rl_loss -= (log_probs[0, top_i]*discounted_reward)
                discounted_reward = discounted_reward * gamma

        # Doing backward pass and parameters update
        rl_loss = rl_loss / n_batch
        total_reward = total_reward / n_batch

        print("total_reward: " + str(total_reward) + "\t" + "rl_loss: " + str(rl_loss))

        rl_loss.backward()
        if grad_clipping is not None:
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), grad_clipping)

        self.generator.optimizer.step()
        
        return total_reward, rl_loss.item()


