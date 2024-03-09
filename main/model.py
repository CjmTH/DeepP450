import numpy as np
import pandas as pd
import pickle as pkl
import random
import os
import pdb
from tqdm import tqdm, trange
from threading import Thread, Lock
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, average_precision_score, \
    jaccard_score, balanced_accuracy_score
from rdkit import Chem
from rdkit.Chem import AllChem
import esm
import torch as th
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from unicore.modules import init_bert_params
from unicore.data import (
    Dictionary, NestedDictionaryDataset, TokenizeDataset, PrependTokenDataset,
    AppendTokenDataset, FromNumpyDataset, RightPadDataset, RightPadDataset2D,
    RawArrayDataset, RawLabelDataset,
)
from unimol.data import (
    KeyDataset, ConformerSampleDataset, AtomTypeDataset,
    RemoveHydrogenDataset, CroppingDataset, NormalizeDataset,
    DistanceDataset, EdgeTypeDataset, RightPadDatasetCoord,
)
from unimol.models.transformer_encoder_with_pair import TransformerEncoderWithPair
from unimol.models.unimol import NonLinearHead, GaussianLayer

def set_random_seed(random_seed=1024): # set random seed
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    np.random.seed(random_seed)
    th.manual_seed(random_seed)
    th.cuda.manual_seed(random_seed)
    th.cuda.manual_seed_all(random_seed)  # if you are using multi-GPU.
    th.backends.cudnn.benchmark = False
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.enabled = False

# smiles to 3D conformation
def calculate_molecule_3D_structure():
    def get_smiles_list_():
        data_df = pd.read_csv("../data/data_0/raw/train_0.csv")
        smiles_list = data_df["smiles"].tolist()
        smiles_list = list(set(smiles_list))
        print(len(smiles_list))
        return smiles_list

    def calculate_molecule_3D_structure_(smiles_list):
        n = len(smiles_list)
        global p
        index = 0
        while True:
            mutex.acquire()
            if p >= n:
                mutex.release()
                break
            index = p
            p += 1
            mutex.release()

            smiles = smiles_list[index]
            print(index, ':', round(index / n * 100, 2), '%', smiles)

            molecule = Chem.MolFromSmiles(smiles)
            molecule = AllChem.AddHs(molecule)
            atoms = [atom.GetSymbol() for atom in molecule.GetAtoms()]
            coordinate_list = []
            result = AllChem.EmbedMolecule(molecule, randomSeed=42, useRandomCoords=True, maxAttempts=1000)
            if result != 0:
                print('EmbedMolecule failed', result, smiles)
                mutex.acquire()
                with open('../data/data_0/result/invalid_smiles.txt', 'a') as f:
                    f.write('EmbedMolecule failed' + ' ' + str(result) + ' ' + str(smiles) + '\n')
                mutex.release()
                continue
            try:
                AllChem.MMFFOptimizeMolecule(molecule)
            except:
                print('MMFFOptimizeMolecule error', smiles)
                mutex.acquire()
                with open('../data/data_0/result/invalid_smiles.txt', 'a') as f:
                    f.write('MMFFOptimizeMolecule error' + ' ' + str(smiles) + '\n')
                mutex.release()
                continue
            coordinates = molecule.GetConformer().GetPositions()

            assert len(atoms) == len(coordinates), "coordinates shape is not align with {}".format(smiles)
            coordinate_list.append(coordinates.astype(np.float32))

            global smiles_to_conformation_dict
            mutex.acquire()
            smiles_to_conformation_dict[smiles] = {'smiles': smiles, 'atoms': atoms, 'coordinates': coordinate_list}
            mutex.release()

    mutex = Lock()
    os.system('rm ../data/data_0/result/invalid_smiles.txt')
    smiles_list = get_smiles_list_()
    global smiles_to_conformation_dict
    smiles_to_conformation_dict = {}
    global p
    p = 0
    thread_count = 16
    threads = []
    for i in range(thread_count):
        threads.append(Thread(target=calculate_molecule_3D_structure_, args=(smiles_list,)))
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    pkl.dump(smiles_to_conformation_dict,
             open('../data/data_0/intermediate/smiles_to_conformation_dict.pkl', 'wb'))
    print('Valid smiles count:', len(smiles_to_conformation_dict))

def construct_data_list():
    data_df = pd.read_csv("../data/data_0/raw/train_0.csv")
    smiles_to_conformation_dict = pkl.load(
        open('../data/data_0/intermediate/smiles_to_conformation_dict.pkl', 'rb'))
    data_list = []
    for index, row in data_df.iterrows():
        smiles = row["smiles"]
        if smiles in smiles_to_conformation_dict:
            data_item = {
                "atoms": smiles_to_conformation_dict[smiles]["atoms"],
                "coordinates": smiles_to_conformation_dict[smiles]["coordinates"],
                "smiles": smiles,
                "sequence": row["sequence"],
                "label": row["label"],
                "dataset_type": row["dataset_type"],
            }
            data_list.append(data_item)
    pkl.dump(data_list, open('../data/data_0/intermediate/data_list.pkl', 'wb'))


def convert_data_list_to_data_loader():
    def convert_data_list_to_dataset_(data_list):
        dictionary = Dictionary.load('../data/data_0/raw/token_list.txt')
        dictionary.add_symbol("[MASK]", is_special=True)
        smiles_dataset = KeyDataset(data_list, "smiles")
        sequence_dataset = KeyDataset(data_list, "sequence")
        label_dataset = KeyDataset(data_list, "label")
        dataset = ConformerSampleDataset(data_list, 1024, "atoms", "coordinates")
        dataset = AtomTypeDataset(data_list, dataset)
        dataset = RemoveHydrogenDataset(dataset, "atoms", "coordinates", True, False)
        dataset = CroppingDataset(dataset, 1, "atoms", "coordinates", 256)
        dataset = NormalizeDataset(dataset, "coordinates", normalize_coord=True)
        token_dataset = KeyDataset(dataset, "atoms")
        token_dataset = TokenizeDataset(token_dataset, dictionary, max_seq_len=512)
        coord_dataset = KeyDataset(dataset, "coordinates")
        src_dataset = AppendTokenDataset(PrependTokenDataset(token_dataset, dictionary.bos()), dictionary.eos())
        edge_type = EdgeTypeDataset(src_dataset, len(dictionary))
        coord_dataset = FromNumpyDataset(coord_dataset)
        coord_dataset = AppendTokenDataset(PrependTokenDataset(coord_dataset, 0.0), 0.0)
        distance_dataset = DistanceDataset(coord_dataset)
        return NestedDictionaryDataset({
            "input": {
                "src_tokens": RightPadDataset(src_dataset, pad_idx=dictionary.pad(), ),
                "src_coord": RightPadDatasetCoord(coord_dataset, pad_idx=0, ),
                "src_distance": RightPadDataset2D(distance_dataset, pad_idx=0, ),
                "src_edge_type": RightPadDataset2D(edge_type, pad_idx=0, ),
                "smiles": RawArrayDataset(smiles_dataset),
                "sequence": RawArrayDataset(sequence_dataset),
            },
            "target": {
                "label": RawLabelDataset(label_dataset),
            }
        })

    batch_size = 8
    data_list = pkl.load(open('../data/data_0/intermediate/data_list.pkl', 'rb'))
    data_list_train = [data_item for data_item in data_list if data_item["dataset_type"] == "train"]
    data_list_validate = [data_item for data_item in data_list if data_item["dataset_type"] == "validate"]
    data_list_test = [data_item for data_item in data_list if data_item["dataset_type"] == "test"]
    dataset_train = convert_data_list_to_dataset_(data_list_train)
    dataset_validate = convert_data_list_to_dataset_(data_list_validate)
    dataset_test = convert_data_list_to_dataset_(data_list_test)
    data_loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True,
                                   collate_fn=dataset_train.collater)
    data_loader_valid = DataLoader(dataset_validate, batch_size=batch_size, shuffle=True,
                                   collate_fn=dataset_validate.collater)
    data_loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, collate_fn=dataset_test.collater)
    return data_loader_train, data_loader_valid, data_loader_test


class UniMolModel(nn.Module):
    def __init__(self):
        super().__init__()
        dictionary = Dictionary.load('../data/data_0/raw/token_list.txt')
        dictionary.add_symbol("[MASK]", is_special=True)
        self.padding_idx = dictionary.pad()
        self.embed_tokens = nn.Embedding(
            len(dictionary), 512, self.padding_idx)
        self._num_updates = None
        self.encoder = TransformerEncoderWithPair(
            encoder_layers=15,
            embed_dim=512,
            ffn_embed_dim=2048,
            attention_heads=64,
            emb_dropout=0.1,
            dropout=0.1,
            attention_dropout=0.1,
            activation_dropout=0.0,
            max_seq_len=512,
            activation_fn='gelu',
            no_final_head_layer_norm=True,
        )

        K = 128
        n_edge_type = len(dictionary) * len(dictionary)
        self.gbf_proj = NonLinearHead(
            K, 64, 'gelu'
        )
        self.gbf = GaussianLayer(K, n_edge_type)

        self.apply(init_bert_params)

    def forward(self, sample,):
        net_input = sample['input']
        src_tokens, src_distance, src_coord, src_edge_type = net_input['src_tokens'], net_input['src_distance'], \
                                                             net_input['src_coord'], net_input['src_edge_type']
        padding_mask = src_tokens.eq(self.padding_idx)
        if not padding_mask.any():
            padding_mask = None
        x = self.embed_tokens(src_tokens)

        def get_dist_features(dist, et):
            n_node = dist.size(-1)
            gbf_feature = self.gbf(dist, et)
            gbf_result = self.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            return graph_attn_bias

        graph_attn_bias = get_dist_features(src_distance, src_edge_type)
        (
            encoder_rep,
            encoder_pair_rep,
            delta_encoder_pair_rep,
            x_norm,
            delta_encoder_pair_rep_norm,
        ) = self.encoder(x, padding_mask=padding_mask, attn_mask=graph_attn_bias)
        output = {
            "molecule_embedding": encoder_rep,
            "molecule_representation": encoder_rep[:, 0, :],  # get cls token
            "smiles": sample['input']["smiles"],
        }
        return output


class TransformerDecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_normalization_cross_attention_1 = nn.LayerNorm(1280) 
        self.cross_attention = nn.MultiheadAttention(embed_dim=512, num_heads=8, kdim=1280, vdim=1280, batch_first=True)
        self.layer_normalization_cross_attention_2 = nn.LayerNorm(512)
        self.feed_forward_cross_attention = nn.Sequential(
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, 512),
        )

    def forward(self, x, y, padding_mask):
        y = self.layer_normalization_cross_attention_1(y)
        y, _ = self.cross_attention(x, y, y, key_padding_mask=padding_mask)
        y_old = y
        y = self.layer_normalization_cross_attention_2(y)
        y = self.feed_forward_cross_attention(y)
        y = y + y_old
        return y


class TransformerEncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_normalization_self_attention_1 = nn.LayerNorm(512) 
        self.self_attention = nn.MultiheadAttention(embed_dim=512, num_heads=8, kdim=512, vdim=512, batch_first=True)
        self.layer_normalization_self_attention_2 = nn.LayerNorm(512)
        self.feed_forward_self_attention = nn.Sequential(
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, 512),
        )

    def forward(self, x):
        x_old = x
        x = self.layer_normalization_self_attention_1(x)
        x, _ = self.self_attention(x, x, x)
        x = x + x_old
        x_old = x
        x = self.layer_normalization_self_attention_2(x)
        x = self.feed_forward_self_attention(x)
        x = x + x_old
        return x


class EsmUnimolClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.molecule_encoder = UniMolModel()
        self.molecule_encoder.load_state_dict(th.load('mol_pre_no_h_220816.pt')['model'], strict=False) 
        self.protein_encoder, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D() 
        self.protein_encoder.load_state_dict(th.load("p450_eukaryota_esm_2.pth")) 
        self.batch_converter = self.alphabet.get_batch_converter(truncation_seq_length=2048)

        self.transformer_layer_cross_attention = TransformerDecoderLayer()
        self.transformer_layer_self_attention = TransformerEncoderLayer()
        self.mlp = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def move_data_batch_to_cuda(self, data_batch):
        data_batch['input'] = {k: v.cuda() if isinstance(v, th.Tensor) else v for k, v in data_batch['input'].items()}
        data_batch['target'] = {k: v.cuda() if isinstance(v, th.Tensor) else v for k, v in data_batch['target'].items()}
        return data_batch

    def forward(self, data_batch):
        data_batch = self.move_data_batch_to_cuda(data_batch)

       
        molecule_encoder_output = self.molecule_encoder(data_batch)
        molecule_embedding = molecule_encoder_output['molecule_embedding']

        
        sequence_batch = data_batch['input']['sequence']
        sequence_batch = [('', sequence) for sequence in sequence_batch]
        _, sequence_batch, token_batch = self.batch_converter(sequence_batch)
        token_batch = token_batch.cuda()

        protein_encoder_output = self.protein_encoder(token_batch, repr_layers=[33], return_contacts=False)
        protein_embedding = protein_encoder_output["representations"][33]

       
        x = self.transformer_layer_cross_attention(molecule_embedding, protein_embedding, None)
        x1 = self.transformer_layer_self_attention(x)
        x2 = x1[:, 0, :] 
        x3 = self.mlp(x2)
        return x3


def evaluate(model, data_loader, csv_save):
    model.eval()
    label_predict = th.tensor([], dtype=th.float32).cuda()
    label_true = th.tensor([], dtype=th.long).cuda()
    with th.no_grad():
        for data_batch in data_loader:
            # for data_batch in tqdm(data_loader):
            label_predict_batch = model(data_batch)

            label_true_batch = data_batch['target']['label'].to(th.long)
            label_predict = th.cat((label_predict, label_predict_batch.detach()), dim=0)
            label_true = th.cat((label_true, label_true_batch.detach()), dim=0)

    label_predict = th.softmax(label_predict, dim=1)
    label_predict = label_predict.cpu().numpy()
    predict_label = np.argmax(label_predict, axis=1)
    label_true = label_true.cpu().numpy()

   
    if csv_save == True:
        df = pd.DataFrame({'label_true': label_true, 'predict_label': predict_label, 'label_predict': label_predict[:, 1]})
        df.to_csv('label_predict_test_0.csv', index=False)

    auc_roc = round(roc_auc_score(label_true, label_predict[:, 1]), 3)
    auc_prc = round(average_precision_score(label_true, label_predict[:, 1]), 3)
    accuracy = round(accuracy_score(label_true, np.argmax(label_predict, axis=1)), 3)
    precision = round(precision_score(label_true, np.argmax(label_predict, axis=1)), 3)
    recall = round(recall_score(label_true, np.argmax(label_predict, axis=1)), 3)
    f1_score = round(2 * precision * recall / (precision + recall), 3)
    jaccard = round(jaccard_score(label_true, np.argmax(label_predict, axis=1)), 3)
    balanced_accuracy = round(balanced_accuracy_score(label_true, np.argmax(label_predict, axis=1)), 3)
    metric = {'auc_roc': auc_roc, 'auc_prc': auc_prc, 'accuracy': accuracy, "balanced_accuracy": balanced_accuracy,
              'precision': precision, 'recall': recall, 'f1_score': f1_score, "jaccard": jaccard}
    return metric


def train(trial_version):
    data_loader_train, data_loader_validate, data_loader_test = convert_data_list_to_data_loader()

    model = EsmUnimolClassifier()
    model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.000005, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 15)

    current_best_metric = -1e10
    max_bearable_epoch = 100
    current_best_epoch = 0

    for epoch in range(2000):
        model.train()
        for step, data_batch in enumerate(data_loader_train):
            label_predict_batch = model(data_batch)
            label_true_batch = data_batch['target']['label'].to(th.long)

            loss = criterion(label_predict_batch, label_true_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (step + 1) % 19 == 0:
                print('Epoch: {}, Step: {}, Loss: {}'.format(epoch, step, round(loss.item(), 3)))

        scheduler.step()

        metric_train = evaluate(model, data_loader_train, csv_save=False)
        metric_validate = evaluate(model, data_loader_validate, csv_save=False)
        metric_test = evaluate(model, data_loader_test, csv_save=False)

        if metric_validate['accuracy'] > current_best_metric:
            current_best_metric = metric_validate['accuracy']
            current_best_epoch = epoch
            th.save(model.state_dict(), f"../weight/{trial_version}.pt")
        print("==================================================================================")
        print('Epoch', epoch)
        print('Train', metric_train)
        print('validate', metric_validate)
        print('Test', metric_test)
        print('current_best_epoch', current_best_epoch, 'current_best_metric', current_best_metric)
        print("==================================================================================")
        if epoch > current_best_epoch + max_bearable_epoch:
            break


def test(trial_version):
    data_loader_train, data_loader_validate, data_loader_test = convert_data_list_to_data_loader()

    model = EsmUnimolClassifier()
    model.cuda()
    model.load_state_dict(th.load(f"../weight/{trial_version}.pt"))

    metric_train = evaluate(model, data_loader_train, csv_save=False)
    metric_validate = evaluate(model, data_loader_validate, csv_save=False)
    metric_test = evaluate(model, data_loader_test, csv_save=True)
    print("Train", metric_train)
    print("validate", metric_validate)
    print("Test", metric_test)

if __name__ == "__main__":
    set_random_seed(1024)
    print("data_process start!") 
    calculate_molecule_3D_structure()
    construct_data_list()

    print("train start!") 
    train(trial_version='1')  

    print("test start!")
    test(trial_version='1') 

    print('All is well!')
