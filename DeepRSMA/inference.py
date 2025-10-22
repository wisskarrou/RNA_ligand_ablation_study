import torch
from data import RNA_dataset, Molecule_dataset
from torch.utils.data import Dataset, DataLoader
import os
import torch
from data import RNA_dataset, Molecule_dataset, Molecule_dataset_independent, RNA_dataset_independent
from model import GNN_rna, GNN_molecule, mole_seq_model, cross_attention2
from torch_geometric.loader import DataLoader
import numpy as np
import os
from torch.autograd import Variable
import torch.nn as nn
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer
from scipy.stats import pearsonr,spearmanr
from sklearn.metrics import  mean_squared_error

torch.set_printoptions(profile="full")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
hidden_dim = 128
seed_dataset = 2
RNA_type = 'All_sf'
rna_dataset = RNA_dataset(RNA_type)
molecule_dataset = Molecule_dataset(RNA_type)


class CustomDualDataset(Dataset):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2

       
        assert len(self.dataset1) == len(self.dataset2)

    def __getitem__(self, index):

        return self.dataset1[index], self.dataset2[index]

    def __len__(self):
        
        return len(self.dataset1)  



def average_multiple_lists(lists):
    # 假设所有列表长度相同
    return [sum(item)/len(lists) for item in zip(*lists)]


class mole_and_rna(nn.Module):
    def __init__(self):
        super(mole_and_rna, self).__init__()
        
        self.rna_graph_model = GNN_rna(hidden_dim)
        self.mole_graph_model = GNN_molecule(hidden_dim)
        
        self.mole_seq = mole_seq_model(hidden_dim)
        
        self.cross_attention2 = cross_attention2(hidden_dim)
        
        
        self.line1 = nn.Linear(hidden_dim*2, 1024)
        self.line2 = nn.Linear(1024, 512)
        self.line3 = nn.Linear(512, 1)
        self.dropout = nn.Dropout(0.2)
        
        self.rna1 = nn.Linear(hidden_dim, hidden_dim*4)
        self.mole1 = nn.Linear(hidden_dim, hidden_dim*4)
        
        self.rna2 = nn.Linear(hidden_dim*4, hidden_dim)
        self.mole2 = nn.Linear(hidden_dim*4, hidden_dim)
        
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
    
    def forward(self, rna_batch, mole_batch):
        rna_out_seq,rna_out_graph, rna_mask_seq, rna_mask_graph, rna_seq_final, rna_graph_final = self.rna_graph_model(rna_batch, device)
        
        mole_graph_emb, mole_graph_final = self.mole_graph_model(mole_batch)
        
        mole_seq_emb, _, mole_mask_seq = self.mole_seq(mole_batch, device)
        
        mole_seq_final = (mole_seq_emb[-1]*(mole_mask_seq.to(device).unsqueeze(dim=2))).mean(dim=1).squeeze(dim=1)


        # mole graph
        flag = 0
        mole_out_graph = []
        mask = []
        batch_node = mole_batch.batch.tolist()
        for i in range(batch_node[len(batch_node)-1] + 1):
            x = mole_graph_emb[flag:flag+batch_node.count(i)]
            temp = torch.zeros((128-x.size()[0]), hidden_dim).to(device)
            x = torch.cat((x, temp),0)
            mole_out_graph.append(x)
            mask.append([] + batch_node.count(i) * [1] + (128 - batch_node.count(i)) * [0])
            
            flag += batch_node.count(i)
        mole_out_graph = torch.stack(mole_out_graph).to(device)
        mole_mask_graph = torch.tensor(mask, dtype=torch.float)
        
        context_layer, attention_score = self.cross_attention2([rna_out_seq, rna_out_graph, mole_seq_emb[-1], mole_out_graph], [rna_mask_seq.to(device), rna_mask_graph.to(device), mole_mask_seq.to(device), mole_mask_graph.to(device)], device)

        out_rna = context_layer[-1][0]
        out_mole = context_layer[-1][1]
           
        rna_cross_seq = ((out_rna[:, 0:512]*(rna_mask_seq.to(device).unsqueeze(dim=2))).mean(dim=1).squeeze(dim=1) + rna_seq_final ) / 2
        rna_cross_stru = ((out_rna[:, 512:]*(rna_mask_graph.to(device).unsqueeze(dim=2))).mean(dim=1).squeeze(dim=1) + rna_graph_final) / 2        
        
        rna_cross = (rna_cross_seq + rna_cross_stru) / 2
        rna_cross = self.rna2(self.dropout((self.relu(self.rna1(rna_cross)))))

        mole_cross_seq = ((out_mole[:,0:128]*(mole_mask_seq.to(device).unsqueeze(dim=2))).mean(dim=1).squeeze(dim=1) + mole_seq_final) / 2
        mole_cross_stru = ((out_mole[:,128:]*(mole_mask_graph.to(device).unsqueeze(dim=2))).mean(dim=1).squeeze(dim=1) + mole_graph_final) / 2
        
        mole_cross = (mole_cross_seq + mole_cross_stru) / 2
        mole_cross = self.mole2(self.dropout((self.relu(self.mole1(mole_cross)))))

        out = torch.cat((rna_cross, mole_cross),1)
        out = self.line1(out)
        out = self.dropout(self.relu(out))
        out = self.line2(out)
        out = self.dropout(self.relu(out))
        out = self.line3(out)

        return out
    
class regressor_stratified_cv:
    def __init__(self,n_splits=10,n_repeats=2,group_count=10,random_state=0,strategy='quantile'):
        self.group_count=group_count
        self.strategy=strategy
        self.cvkwargs=dict(n_splits=n_splits,n_repeats=n_repeats,random_state=random_state)  #Added shuffle here
        self.cv=RepeatedStratifiedKFold(**self.cvkwargs)
        self.discretizer=KBinsDiscretizer(n_bins=self.group_count,encode='ordinal',strategy=self.strategy)  
            
    def split(self,X,y,groups=None):
        kgroups=self.discretizer.fit_transform(y[:,None])[:,0]
        return self.cv.split(X,kgroups,groups)
    
    def get_n_splits(self,X,y,groups=None):
        return self.cv.get_n_splits(X,y,groups)


y_pred_all = []
i = 0
p_list = []
m_list = []
s_list = []
r_list = []
kf = regressor_stratified_cv(n_splits=5, n_repeats=1, random_state=seed_dataset, group_count=5, strategy='uniform')

for train_id,test_id in kf.split(rna_dataset, rna_dataset.y):
    max_p = -1
    max_s = -1
    max_rmse = 0
    i = i + 1
    model = mole_and_rna()
    model.to(device)
    model_dict = torch.load('save/model5fold_All_sf2_'+str(i)+'_2.pth', map_location=device)

    model.load_state_dict(model_dict)
    print("Fold", i)
    
    # Combine RNA Dataset and Mole Dataset
    test_dataset = CustomDualDataset(rna_dataset[test_id], molecule_dataset[test_id])


    test_loader = DataLoader(
        test_dataset, batch_size=1, num_workers=1, drop_last=False, shuffle=False
    )
    
    # test
    with torch.set_grad_enabled(False):
        test_loss = 0
        model.eval()
        y_label = []
        y_pred = []
        for step, (batch_rna_test,batch_mole_test) in enumerate(test_loader):

            label = Variable(torch.from_numpy(np.array(batch_rna_test.y))).float()
            score = model(batch_rna_test.to(device), batch_mole_test.to(device))
            n = torch.squeeze(score, 1)
            logits = torch.squeeze(score).detach().cpu().numpy()
            label_ids = label.to('cpu').numpy()
            y_label = y_label + label_ids.flatten().tolist()
            y_pred = y_pred + logits.flatten().tolist()

        
    p = pearsonr(y_label, y_pred)
    s = spearmanr(y_label, y_pred)
    rmse = np.sqrt(mean_squared_error(y_label, y_pred))

    p_list.append(p[0])
    s_list.append(s[0])
    r_list.append(rmse)

print('p:', np.mean(p_list), p_list)
print('s:', np.mean(s_list), s_list)
print('rmse:', np.mean(r_list), r_list)



