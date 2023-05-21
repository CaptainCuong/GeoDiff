import os
import argparse
import pickle
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from utils.datasets import PackedConformationDataset
from utils.evaluation.covmat import CovMatEvaluator, print_covmat_results
from utils.misc import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default=r'logs/qm9_default_2023_03_16__20_52_33/sample_2023_03_17__01_45_34/samples_all.pkl')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--ratio', type=int, default=2)
    parser.add_argument('--start_idx', type=int, default=0)
    args = parser.parse_args()
    assert os.path.isfile(args.path)

    # Logging
    tag = args.path.split('/')[-1].split('.')[0]
    logger = get_logger('eval', os.path.dirname(args.path), 'log_eval_'+tag+'.txt')
    
    # Load results
    logger.info('Loading results: %s' % args.path)
    with open(args.path, 'rb') as f:
        packed_dataset = pickle.load(f)
    logger.info('Total: %d' % len(packed_dataset))
    '''
    packed_dataset: list(Data)
    e.g.
    [Data(edge_index=[2, 156], pos=[18, 3], atom_type=[18], edge_type=[156], 
            rdmol=<rdkit.Chem.rdchem.Mol object at 0x000002B731435580>, smiles='CCOCCC(=O)C#N', 
            idx=[1], pos_ref=[954, 3], num_pos_ref=[1], num_nodes_per_graph=[1], bond_edge_index=[2, 34], 
            edge_order=[156], is_bond=[156], pos_gen=[1908, 3]),...]
    '''
    '''
    # Uncomment if want to generate conformers with RDKit
    for data in packed_dataset:
        data.rdmol = Chem.MolFromSmiles(data.smiles)
        data.rdmol = Chem.AddHs(data.rdmol)
        N = data.rdmol.GetNumAtoms()
        rd_pos = []
        numConfs = int(data.pos_gen.shape[0]/N)
        AllChem.EmbedMultipleConfs(data.rdmol, numConfs=numConfs)
        for i in range(numConfs):
            for atom_num in range(N):
                atom_pos = data.rdmol.GetConformer(i).GetAtomPosition(atom_num)
                rd_pos.append([atom_pos.x,atom_pos.y,atom_pos.z])
        data['pos_gen'] = torch.tensor(rd_pos)
        

        print('-'*30)
    '''
    # Evaluator
    evaluator = CovMatEvaluator(
        num_workers = args.num_workers,
        ratio = args.ratio,
        print_fn=logger.info,
    )
    results = evaluator(
        packed_data_list = list(packed_dataset),
        start_idx = args.start_idx,
    )
    df = print_covmat_results(results, print_fn=logger.info)

    # Save results
    csv_fn = args.path[:-4] + '_covmat.csv'
    results_fn = args.path[:-4] + '_covmat.pkl'
    df.to_csv(csv_fn)
    print(type(results_fn))
    print(results_fn)
    print(type(results))
    print(results)
    with open(results_fn, 'wb') as f:
        pickle.dump(results, f)

