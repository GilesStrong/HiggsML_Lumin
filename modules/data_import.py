import pandas as pd
import pickle
import numpy as np
import optparse
import os
import h5py
from pathlib import Path
from collections import OrderedDict
from typing import Union, Tuple, Dict, Optional, List
from functools import partial

from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.cluster import k_means

from lumin.data_processing.hep_proc import calc_pair_mass, proc_event
from lumin.data_processing.pre_proc import fit_input_pipe, proc_cats
from lumin.data_processing.file_proc import df2foldfile
from lumin.utils.misc import ids2unique, str2bool


def calc_pair_transverse_mass(df:pd.DataFrame, masses:Union[Tuple[float,float],Tuple[np.ndarray,np.ndarray]], feat_map:Dict[str,str]) -> np.ndarray:
    r'''
    Vectorised computation of invarient transverse mass of pair of particles with given masses, using transverse components of 3-momenta.
    Only works for vectors defined in Cartesian coordinates.

    Arguments:
        df: DataFrame vector components
        masses: tuple of masses of particles (either constant or different pair of masses per pair of particles)
        feat_map: dictionary mapping of requested momentum components to the features in df

    Returns:
        np.array of invarient masses
    '''

    # TODO: rewrite to not use a DataFrame for holding parent vector
    # TODO: add inplace option
    # TODO: extend to work on pT, eta, phi coordinates

    tmp = pd.DataFrame()
    tmp['0_E'] = np.sqrt((masses[0]**2)+np.square(df.loc[:, feat_map['0_px']])+np.square(df.loc[:, feat_map['0_py']]))
    tmp['1_E'] = np.sqrt((masses[1]**2)+np.square(df.loc[:, feat_map['1_px']])+np.square(df.loc[:, feat_map['1_py']]))
    tmp['p_px'] = df.loc[:, feat_map['0_px']]+df.loc[:, feat_map['1_px']]
    tmp['p_py'] = df.loc[:, feat_map['0_py']]+df.loc[:, feat_map['1_py']]
    tmp['p_E'] = tmp.loc[:, '0_E']+tmp.loc[:, '1_E']
    tmp['p_p2'] = np.square(tmp.loc[:, 'p_px'])+np.square(tmp.loc[:, 'p_py'])
    tmp['p_mass'] = np.sqrt(np.square(tmp.loc[:, 'p_E'])-tmp.loc[:, 'p_p2'])
    return tmp.p_mass.values


def add_mass_feats(df:pd.DataFrame) -> None:
    '''Add extra mass features used by Melis https://pdfs.semanticscholar.org/01e7/aee90cb61178fcb09d3fa813294a216116a2.pdf'''
    # ln(1 + m_inv(tau, jet_0))
    m = calc_pair_mass(df, (0,0), feat_map={'0_px':'PRI_tau_px', '0_py':'PRI_tau_py', '0_pz':'PRI_tau_pz',
                                            '1_px':'PRI_jet_leading_px', '1_py':'PRI_jet_leading_py', '1_pz':'PRI_jet_leading_pz'})
    df['EXT_m_tj0'] = np.log(1+m)

    # ln(1 + m_inv(tau, jet_1))
    m = calc_pair_mass(df, (0,0), feat_map={'0_px':'PRI_tau_px', '0_py':'PRI_tau_py', '0_pz':'PRI_tau_pz',
                                            '1_px':'PRI_jet_subleading_px', '1_py':'PRI_jet_subleading_py', '1_pz':'PRI_jet_subleading_pz'})
    df['EXT_m_tj1'] = np.log(1+m)

    # ln(1 + m_inv(tau, lep))
    m = calc_pair_mass(df, (0,0), feat_map={'0_px':'PRI_tau_px', '0_py':'PRI_tau_py', '0_pz':'PRI_tau_pz',
                                            '1_px':'PRI_lep_px', '1_py':'PRI_lep_py', '1_pz':'PRI_lep_pz'})
    df['EXT_m_tl'] = np.log(1+m)

    # ln(1 + mt_inv(tau, jet_0))
    m = calc_pair_transverse_mass(df, (0,0), feat_map={'0_px':'PRI_tau_px', '0_py':'PRI_tau_py',
                                                       '1_px':'PRI_jet_leading_px', '1_py':'PRI_jet_leading_py'})
    df['EXT_mt_tj0'] = np.log(1+m)

    # ln(1 + mt_inv(tau, jet_1))
    m = calc_pair_transverse_mass(df, (0,0), feat_map={'0_px':'PRI_tau_px', '0_py':'PRI_tau_py',
                                                       '1_px':'PRI_jet_subleading_px', '1_py':'PRI_jet_subleading_py'})
    df['EXT_mt_tj1'] = np.log(1+m)


def import_data(data_path:Path=Path("../data/"),
                rotate:bool=False, flip_y:bool=False, flip_z:bool=False, cartesian:bool=True,
                mode:str='OpenData',
                val_size:float=0.2, seed:Optional[int]=None, cat_feats:Optional[List[str]]=None, extra:bool=False):
    '''Import and split data from CSV(s)'''
    if cat_feats is None: cat_feats = []
    if mode == 'OpenData':  # If using data from CERN Open Access
        data = pd.read_csv(data_path/'atlas-higgs-challenge-2014-v2.csv')
        data.rename(index=str, columns={"KaggleWeight": "gen_weight", 'PRI_met': 'PRI_met_pt'}, inplace=True)
        data.drop(columns=['Weight'], inplace=True)
        training_data = pd.DataFrame(data.loc[data.KaggleSet == 't'])
        training_data.drop(columns=['KaggleSet'], inplace=True)
        
        test = pd.DataFrame(data.loc[(data.KaggleSet == 'b') | (data.KaggleSet == 'v')])
        test['private'] = 0
        test.loc[(data.KaggleSet == 'v'), 'private'] = 1
        test['gen_target'] = 0
        test.loc[test.Label == 's', 'gen_target'] = 1
        test.drop(columns=['KaggleSet', 'Label'], inplace=True)

    else:  # If using data from Kaggle
        training_data = pd.read_csv(data_path/'training.csv')
        training_data.rename(index=str, columns={"Weight": "gen_weight", 'PRI_met': 'PRI_met_pt'}, inplace=True)
        test = pd.read_csv(data_path/'test.csv')
        test.rename(index=str, columns={'PRI_met': 'PRI_met_pt'}, inplace=True)

    proc_event(training_data, fix_phi=rotate, fix_y=flip_y, fix_z=flip_z, use_cartesian=cartesian, ref_vec_0='PRI_lep', ref_vec_1='PRI_tau', default_vals=[-999.0], keep_feats=['PRI_met_pt'])
    proc_event(test, fix_phi=rotate, fix_y=flip_y, fix_z=flip_z, use_cartesian=cartesian, ref_vec_0='PRI_lep', ref_vec_1='PRI_tau', default_vals=[-999.0], keep_feats=['PRI_met_pt'])

    if extra:
        print('Computing extra features')
        add_mass_feats(training_data)
        add_mass_feats(test)
    
    training_data['gen_target'] = 0
    training_data.loc[training_data.Label == 's', 'gen_target'] = 1
    training_data.drop(columns=['Label'], inplace=True)
    training_data['gen_weight_original'] = training_data['gen_weight']  # gen_weight might be renormalised

    training_data['gen_strat_key'] = training_data['gen_target'] if len(cat_feats) == 0 else ids2unique(training_data[['gen_target'] + cat_feats].values)
    
    train_feats = [x for x in training_data.columns if 'gen' not in x and x != 'EventId' and 'kaggle' not in x.lower()]
    train, val = train_test_split(training_data, test_size=val_size, random_state=seed, stratify=training_data.gen_strat_key)

    print('Training on {} datapoints and validating on {}, using {} feats:\n{}'.format(len(train), len(val), len(train_feats), [x for x in train_feats]))

    return {'train': train[train_feats + ['gen_target', 'gen_weight', 'gen_weight_original', 'gen_strat_key']], 
            'val': val[train_feats + ['gen_target', 'gen_weight', 'gen_weight_original', 'gen_strat_key']],
            'test': test,
            'feats': train_feats}


def proc_targets(data:Dict[str,pd.DataFrame]):
    cluster = k_means(data['train'].loc[data['train'].gen_target == 0, 'gen_weight'].values[:, None], 3)
    data['train'].loc[data['train'].gen_target == 0, 'gen_sample'] = cluster[1]
    data['train'].loc[data['train'].gen_target == 1, 'gen_sample'] = 3
    data['val'].loc[data['val'].gen_target == 0, 'gen_sample'] = abs(data['val'].loc[data['val'].gen_target == 0, 'gen_weight'][None, :] - cluster[0][:, None]).argmin(axis=0)[0]
    data['val'].loc[data['val'].gen_target == 1, 'gen_sample'] = 3


def run_data_import(data_path:Path, rotate:bool, flip_y:bool, flip_z:bool, cartesian:bool, mode:str, val_size:float, seed:Optional[int], n_folds:int,
                    cat_feats:List, multi:bool, extra:bool, matrix:Optional[str]):
    '''Run through all the stages to save the data into files for training, validation, and testing'''
    # Get Data
    data = import_data(data_path, rotate, flip_y, flip_z, cartesian, mode, val_size, seed, cat_feats, extra)

    cont_feats = [x for x in data['feats'] if x not in cat_feats]
    input_pipe = fit_input_pipe(data['train'], cont_feats, data_path/'input_pipe')
    data['train'][cont_feats] = input_pipe.transform(data['train'][cont_feats])
    data['val'][cont_feats]   = input_pipe.transform(data['val'][cont_feats])
    data['test'][cont_feats]  = input_pipe.transform(data['test'][cont_feats])
    cat_maps, cat_szs = proc_cats(data['train'], cat_feats, data['val'], data['test'])

    misc_feats = ['gen_weight_original']
    if multi:
        proc_targets(data)
        targ_feat = 'gen_sample'
        misc_feats.append('gen_target')
        for c in set(data['train'].gen_sample):
            data['train'].loc[data['train'].gen_sample == c, 'gen_weight'] /= np.sum(data['train'].loc[data['train'].gen_sample == c, 'gen_weight'])
    else:
        targ_feat = 'gen_target'
        data['train'].loc[data['train'].gen_target == 0, 'gen_weight'] /= np.sum(data['train'].loc[data['train'].gen_target == 0, 'gen_weight'])
        data['train'].loc[data['train'].gen_target == 1, 'gen_weight'] /= np.sum(data['train'].loc[data['train'].gen_target == 1, 'gen_weight'])    

    fold_func = partial(df2foldfile, n_folds=10, cont_feats=cont_feats, cat_feats=cat_feats, cat_maps=cat_maps, targ_feats=targ_feat, wgt_feat='gen_weight',
                        targ_type='int', strat_key='gen_strat_key')
    
    if matrix is not None:
        fold_func.keyworks['matrix_vecs'] = ['PRI_lep', 'PRI_tau', 'PRI_met', 'PRI_jet_leading', 'PRI_jet_subleading']
        fold_func.keyworks['matrix_feats_per_vec'] = ['px', 'py', 'pz']
        fold_func.keyworks['matrix_row_wise'] = 'row' in matrix.lower()

    fold_func(df=data['train'], savename=data_path/'train', misc_feats=misc_feats)
    fold_func(df=data['val'], savename=data_path/'val', misc_feats=misc_feats)
    fold_func(df=data['test'], savename=data_path/'test', misc_feats=['private', 'EventId'])


def parse_cats(string): return [x.strip() for x in string.split(',')] if string is not None else []
        

if __name__ == '__main__':
    parser = optparse.OptionParser(usage=__doc__)
    parser.add_option("-d", "--data_path", dest="data_path", action="store", default="./data/", help="Data folder location")
    parser.add_option("-r", "--rotate", dest="rotate", action="store", default=False, help="Rotate events in phi to have common alignment")
    parser.add_option("-y", "--flipy", dest="flip_y", action="store", default=False, help="Flip events in y to have common alignment")
    parser.add_option("-z", "--flipz", dest="flip_z", action="store", default=False, help="Flip events in z to have common alignment")
    parser.add_option("-c", "--cartesian", dest="cartesian", action="store", default=True, help="Convert to Cartesian system")
    parser.add_option("-m", "--mode", dest="mode", action="store", default="OpenData", help="Using open data or Kaggle data")
    parser.add_option("-v", "--val_size", dest="val_size", action="store", default=0.2, help="Fraction of data to use for validation")
    parser.add_option("-s", "--seed", dest="seed", action="store", default=1337, help="Seed for train/val split")
    parser.add_option("-n", "--n_folds", dest="n_folds", action="store", default=10, help="Number of folds to split data")
    parser.add_option("-f", "--cat_feats", dest="cat_feats", action="store", default=None, help="Comma-separated list of features to be treated as categorical")
    parser.add_option("--multi", dest="multi", action="store", default=False, help="Use multiclass classification") 
    parser.add_option("-e", "--extra", dest="extra", action="store", default=False, help="Compute extra Melis features")   
    parser.add_option("--matrix", dest="matrix", action="store", default=None, help="'row' of 'column' to store vectors as matrices, otherwise leave as None")    
    opts, args = parser.parse_args()

    run_data_import(Path(opts.data_path),
                    str2bool(opts.rotate), str2bool(opts.flip_y), str2bool(opts.flip_z), str2bool(opts.cartesian),
                    opts.mode, opts.val_size, int(opts.seed), opts.n_folds, parse_cats(opts.cat_feats), str2bool(opts.multi), str2bool(opts.extra),
                    opts.matrix)
    