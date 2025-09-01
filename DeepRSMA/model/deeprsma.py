import torch
from torch.utils.data import Dataset, DataLoader
import os
from data import RNA_dataset, Molecule_dataset, RNA_dataset_independent, Molecule_dataset_independent, WordVocab
from model import RNA_feature_extraction, GNN_molecule, mole_seq_model, cross_attention
import torch.nn as nn

class DeepRSMA(nn.Module):
    def __init__(self, hidden_dim=128):
        super(DeepRSMA, self).__init__()
        self.hidden_dim = hidden_dim
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # RNA graph + seq
        self.rna_graph_model = RNA_feature_extraction(self.hidden_dim)
        
        # Mole graph
        self.mole_graph_model = GNN_molecule(self.hidden_dim)
        # Mole seq
        self.mole_seq_model = mole_seq_model(self.hidden_dim)

        # Cross fusion module
        self.cross_attention = cross_attention(self.hidden_dim)
        
        self.line1 = nn.Linear(self.hidden_dim*2, 1024)
        self.line2 = nn.Linear(1024, 512)
        self.line3 = nn.Linear(512, 1)
        self.dropout = nn.Dropout(0.2)
        
        self.rna1 = nn.Linear(self.hidden_dim, self.hidden_dim*4)
        self.mole1 = nn.Linear(self.hidden_dim, self.hidden_dim*4)
        
        self.rna2 = nn.Linear(self.hidden_dim*4, self.hidden_dim)
        self.mole2 = nn.Linear(self.hidden_dim*4, self.hidden_dim)
        
        self.relu = nn.ReLU()
    
    def forward(self, rna_batch, mole_batch):
        rna_out_seq,rna_out_graph, rna_mask_seq, rna_mask_graph, rna_seq_final, rna_graph_final = self.rna_graph_model(rna_batch, self.device)
        
        mole_graph_emb, mole_graph_final = self.mole_graph_model(mole_batch)
        
        mole_seq_emb, _, mole_mask_seq = self.mole_seq_model(mole_batch, self.device)
        
        mole_seq_final = (mole_seq_emb[-1]*(mole_mask_seq.to(self.device).unsqueeze(dim=2))).mean(dim=1).squeeze(dim=1)


        # mole graph
        flag = 0
        mole_out_graph = []
        mask = []
        for i in mole_batch.graph_len:
            count_i = i
            x = mole_graph_emb[flag:flag+count_i]
            temp = torch.zeros((128-x.size()[0]), self.hidden_dim).to(self.device)
            x = torch.cat((x, temp),0)
            mole_out_graph.append(x)
            mask.append([] + count_i * [1] + (128 - count_i) * [0])
            flag += count_i
        mole_out_graph = torch.stack(mole_out_graph).to(self.device)
        mole_mask_graph = torch.tensor(mask, dtype=torch.float)
        
        context_layer, attention_score = self.cross_attention([rna_out_seq, rna_out_graph, mole_seq_emb[-1], mole_out_graph], [rna_mask_seq.to(self.device), rna_mask_graph.to(self.device), mole_mask_seq.to(self.device), mole_mask_graph.to(self.device)], self.device)

        
        out_rna = context_layer[-1][0]
        out_mole = context_layer[-1][1]
        
        # Affinity Prediction Module
        rna_cross_seq = ((out_rna[:, 0:512]*(rna_mask_seq.to(self.device).unsqueeze(dim=2))).mean(dim=1).squeeze(dim=1) + rna_seq_final ) / 2
        rna_cross_stru = ((out_rna[:, 512:]*(rna_mask_graph.to(self.device).unsqueeze(dim=2))).mean(dim=1).squeeze(dim=1) + rna_graph_final) / 2        

        rna_cross = (rna_cross_seq + rna_cross_stru) / 2
        rna_cross = self.rna2(self.dropout((self.relu(self.rna1(rna_cross)))))

        
        mole_cross_seq = ((out_mole[:,0:128]*(mole_mask_seq.to(self.device).unsqueeze(dim=2))).mean(dim=1).squeeze(dim=1) + mole_seq_final) / 2
        mole_cross_stru = ((out_mole[:,128:]*(mole_mask_graph.to(self.device).unsqueeze(dim=2))).mean(dim=1).squeeze(dim=1) + mole_graph_final) / 2
        
        mole_cross = (mole_cross_seq + mole_cross_stru) / 2
        mole_cross = self.mole2(self.dropout((self.relu(self.mole1(mole_cross)))))   
        
        out = torch.cat((rna_cross, mole_cross),1)
        out = self.line1(out)
        out = self.dropout(self.relu(out))
        out = self.line2(out)
        out = self.dropout(self.relu(out))
        out = self.line3(out)
        
        return out