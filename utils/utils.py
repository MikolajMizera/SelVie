"""
Utility functions for SelVie
"""
from uuid import uuid4
from ast import literal_eval
from os import remove, makedirs
from os.path import join, split, exists, isdir
from glob import glob
from time import time
from base64 import b64encode, b64decode
import numpy as np
import pandas as pd

from tqdm import tqdm
from sklearn.metrics import pairwise_distances
from rdkit.Chem import Draw, SDMolSupplier, rdFMCS, MolFromSmarts, RemoveHs
from rdkit.Chem import SDWriter
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol



def speed_tests(mols, fps_radius, fps_ns_bits):
    """ This function aims at comparison between accuracy of NNs search and
    size of fingerprints. It will compare correlation between NNs computed 
    based with each size of fingerprint specified in fps_ns_bits list and NNs 
    computed with the largest fps_nbits from fps_ns_bits."""
    
    D_mats, timings = [], []
    for fps_nbits in fps_ns_bits:
        get_fps = lambda m: GetMorganFingerprintAsBitVect(m, 
                                                  int(fps_radius), 
                                                  int(fps_nbits))            
        t0 = time()
        mols_fps = np.array([get_fps(m) for m in mols])
        D_mats.append(pairwise_distances(mols_fps, mols_fps))
        timings.append(time()-t0)
    
    D_max = D_mats[np.argmax(fps_ns_bits)]
    
    corrs = [np.mean([np.corrcoef(r1[:10], r2[:10])[0,1] for r1, r2 in zip(D, D_max)])
            for D in D_mats]
    
    return corrs, timings

def get_mcs(m1, m2, f):
    return rdFMCS.FindMCS([RemoveHs(f(m1)), RemoveHs(f(m2))], 
                          bondCompare=rdFMCS.BondCompare.CompareOrderExact, 
                          ringMatchesRingOnly=True)
        
def get_MCSs(test_mols, known_mols, nns_indices=None, murcko_scaff=False):
    
    if nns_indices is None:
        nns_indices = [np.arange(len(known_mols))]*len(test_mols)
    
    if murcko_scaff:
        f = lambda x: GetScaffoldForMol(x)
    else:
        f = lambda x: x
        
    known_mols = np.array(known_mols)
    
    MCSs, MCS_matches, NN_mols, NN_MCS_matches  = [], [], [], []
    
    for query_mol, nn_i in list(zip(test_mols, nns_indices)):
        
        known_subset = known_mols[nn_i]
        
        query_MCS = [get_mcs(query_mol, m, f) for m in tqdm(known_subset)]
        query_MCS_sim = [m.numAtoms for m in query_MCS]
        NN_mol = known_subset[np.argmax(query_MCS_sim)]
        mcs = query_MCS[np.argmax(query_MCS_sim)]        
        mcs_mol = MolFromSmarts(mcs.smartsString)
        NN_mol_match = NN_mol.GetSubstructMatch(mcs_mol)
        query_mol_match = query_mol.GetSubstructMatch(mcs_mol)
        
        MCSs.append(mcs)
        MCS_matches.append(query_mol_match)
        NN_mols.append(NN_mol)
        NN_MCS_matches.append(NN_mol_match)
        
    return MCSs, MCS_matches, NN_mols, NN_MCS_matches

def preprocess_mols(mols, session_id):
    
    session_dir = join('uploads', session_id)
    mols = np.array(mols)
    df = pd.DataFrame([m.GetPropsAsDict() for m in mols])
    df['NN'] = np.nan
    
    exp_cols = [c for c in df.columns if 'experimental' in c]
    # if any experiemntal value is known for a given molecule, the molecule is 
    # assumed to be known
    experiemntal_mask = np.any(~pd.isna(df[exp_cols]), 1)
    
    # TO DO - check if there is no issues with preserving order
    test_mols = mols[~experiemntal_mask]
    test_mols_ids = df.index[~experiemntal_mask]
    known_mols = mols[experiemntal_mask]
    known_mols_ids = df.index[experiemntal_mask]
        
    # get indices of NNs for molecules that don't have experiemntal data
    # ie. that were not previously tested molecules
    if len(test_mols):
        test_nns_idx, similarity = get_Tanimoto_NNs(test_mols, known_mols, 3, 
                                                    nns=50, return_sim=True)
        test_nns_ids = test_mols_ids[test_nns_idx]
        formatted = list(map(repr, test_nns_ids.tolist()))
        df.at[~experiemntal_mask.values, 'NN'] = formatted
        df.at[~experiemntal_mask.values, 'Similarity_Tanimoto'] = similarity[:,0]
    
    # get indices of NNs for molecules that have experiemntal data
    if len(known_mols):
        known_nns_idx, similarity = get_Tanimoto_NNs(known_mols, known_mols, 3, 
                                                     order=1, nns=50,
                                                     return_sim=True)
        known_nns_ids = known_mols_ids[known_nns_idx]
        formatted = list(map(repr, known_nns_ids.tolist()))
        df.at[experiemntal_mask.values, 'NN'] = formatted
        df.at[experiemntal_mask.values, 'Similarity_Tanimoto'] = similarity[:,0]
    
    # Save molecules as dataset
    ## Check if the dir already exists
    if exists(session_dir):
        raise RuntimeError('The session directory %s already exists!'%session_dir)
    else:
        makedirs(session_dir)
    
    ## Write id of Nearest Neighbour to SDF properties and save each mol to 
    ## separate file, the filenames are equal to index in dataset
    for idx, mol in zip(df.index, mols):
       writer = SDWriter(join(session_dir, '%d.sdf'%idx))
       mol.SetProp('NN', '%s'%df.loc[idx]['NN'])
       mol.SetProp('Similarity_Tanimoto', '%s'%df.loc[idx]['Similarity_Tanimoto'])
       writer.write(mol)
       writer.close()
       
    return session_dir

def load_props(mols_dir):
    
    props = []
    for f in glob(join(mols_dir, '*.sdf')):
        p = SDMolSupplier(f)[0].GetPropsAsDict()
        p['id'] = split(f)[-1].replace('.sdf', '')
        props.append(p)
    df = pd.DataFrame(props)
    
    # Limit dataframe only to necessary columns    
    sorted_cols = []
    for r in sorted(set([c.split('_')[0] for c in df.columns if 'prediction' in c])):
        sorted_cols += ['%s_experimental'%r, '%s_prediction'%r, '%s_error'%r]
    sorted_cols = ['molId']+sorted_cols+['Similarity_Tanimoto', 'NN', 'id']
    
    return df[sorted_cols]

def get_Tanimoto_NNs(test_mols, known_mols, fps_radius, fps_nbits=512, order=0,
                     nns=1, return_sim=False):
    """Looks for nearest neighbour in known_mols for each molecule from 
    test_mols. The molecules are represented by Morgan fingerprints."""
    
    get_fps = lambda m: GetMorganFingerprintAsBitVect(m, 
                                                      int(fps_radius), 
                                                      int(fps_nbits))

    test_mols_fps = np.array([get_fps(m) for m in test_mols])
    known_mols_fps = np.array([get_fps(m) for m in known_mols])

    D = pairwise_distances(test_mols_fps, known_mols_fps, 'jaccard')
    sorted_ids = np.argsort(D)[:, order:order+nns]
    sorted_similarities = 1-np.sort(D)[:, order:order+nns]
    
    if return_sim:
        return sorted_ids, sorted_similarities
    else:
        return sorted_ids

def MCS_NN_search(sdf_file):
    
    sess_dir = split(sdf_file)[0]
    mol = SDMolSupplier(sdf_file, removeHs=True)[0]
    candidate_ids = literal_eval(mol.GetPropsAsDict()['NN'])
    candidate_mols = [SDMolSupplier(join(sess_dir, '%d.sdf'%id), removeHs=True)[0]
                        for id in candidate_ids]
    MCSs, MCS_matches, NN_mols, NN_MCS_matches = get_MCSs([mol], candidate_mols)
    mol_img = draw_base64(mol, highlightAtoms=MCS_matches[0])
    nn_img = draw_base64(NN_mols[0], highlightAtoms=NN_MCS_matches[0])
    return mol_img, nn_img
    

def get_NN_id(sdf_file):
    
    mol = SDMolSupplier(sdf_file, removeHs=True)[0]
    return literal_eval(mol.GetPropsAsDict()['NN'])[0]

def get_properties(sdf_file):
    mol = SDMolSupplier(sdf_file, removeHs=True)[0]
    return mol.GetPropsAsDict()

def draw_structure(sdf_file, **kwargs):
    
    mol = SDMolSupplier(sdf_file, removeHs=True)[0]
    return draw_base64(mol, **kwargs)

def draw_base64(mol, legend='', highlightAtoms=[], molSize=(200, 200)):
    """A function to depict a moelcular structure and encode it as a
    base64 string."""

    data = Draw._moltoimg(mol,
                          molSize,
                          highlightAtoms,
                          legend,
                          returnPNG=True,
                          kekulize=True)
    return b64encode(data).decode('ascii')

def parse_sdf(contents, filename):
    """Loads contents of an uploaded file and tries to parse as a SDF. Returns
    list of RDKit molecules and status message. Returns empty list and error
    meassage in case of failure."""
    
    content_type, content_string = contents.split(',')
    decoded = b64decode(content_string)
    session_id = str(uuid4())
    
    try:
        if filename[-4:].lower()=='.sdf':
            
            # Generate random file name and save contents to a file
            unique_fname = join('uploads', '%s.sdf'%session_id)
            with open(unique_fname, 'w') as fh:
                fh.write(decoded.decode('utf-8'))
                
            mols = SDMolSupplier(unique_fname, removeHs=False)
            n_mols = len(mols)
            mols = [m for m in mols if m]
            n_sucess = len(mols)
            
            try:
                remove(unique_fname)
            except Exception as e:
                #This is not critical
                print(e)
                
            return mols, 'Loaded %d/%d mols'%(n_mols, n_sucess), session_id
        else:
            return [], 'The file has a wrong format.', session_id
        
    except Exception as e:
        print(e)
        return [], 'Error occured during processing of a file.', session_id

    