import pandas as pd
from sklearn import linear_model

family = "Riboswitch"
ablations = ["none","target_swap","ligand_swap"]

df_inference_list = []

for ablation in ablations:

    if ablation=="none":
        seeds = [0]
    else:
        seeds = [0,1,2]

    for seed in seeds:

        if ablation in ["target_swap","ligand_swap"]:
            test_data_pos = f"data/{family}_pos_v1_{ablation}_seed{seed}.csv"
            test_data_neg = f"data/{family}_neg_v1_{ablation}_seed{seed}.csv"

        else:
            test_data_pos = f"data/{family}_pos_v1.csv"
            test_data_neg = f"data/{family}_neg_v1.csv"

        df1 = pd.read_csv(test_data_pos, sep='\t', header=0)  #Test dataset positive
        df1.fillna(0, inplace=True)

        df2 = pd.read_csv(test_data_neg, sep='\t', header=0)  #Test dataset negative
        df2.fillna(0, inplace=True)

        features_file_path = f"features/RFECV_model_features_{family}.txt"

        with open(features_file_path,'r') as features_file:

            top_model_feat = [line.rstrip('\n') for line in features_file]

        dataset = f"../Data_preprocessing/Riboswitch_output/Final_dataset_{family}_v1.csv"
        model_dataset = pd.read_csv(dataset, sep='\t', header=0)
        X_dataset = model_dataset[top_model_feat].to_numpy()
        Y_dataset = model_dataset['pKd'].to_numpy()
        model = linear_model.LinearRegression()
        model.fit(X_dataset, Y_dataset)

        pos_dataset = df1[top_model_feat].to_numpy()
        pos_pred = model.predict(pos_dataset)
        df1["predicted_affinity"] = (pos_pred>=4.0).astype(float)

        neg_dataset = df2[top_model_feat].to_numpy()
        neg_pred = model.predict(neg_dataset)
        df2["predicted_affinity"] = (neg_pred>=4.0).astype(float)

        df_inference = pd.concat([df1,df2])[["Target_RNA_name","name","pKd","predicted_affinity"]].rename({"Target_RNA_name":"RNA_id","name":"mol_id","pKd":"true_affinity"}, axis=1)
        df_inference = df_inference.assign(ablation=ablation)
        df_inference = df_inference.assign(seed=seed)
        df_inference_list.append(df_inference)

df_inference_global = pd.concat(df_inference_list, ignore_index=True)

df_inference_global.to_csv("inference_results.csv", index=False)