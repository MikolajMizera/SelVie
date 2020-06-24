"""
These are tests for SelVie.

"""

from shutil import rmtree
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit.Chem import SDMolSupplier

from utils import speed_tests, get_MCSs, get_Tanimoto_NNs, preprocess_mols


def time_fps(sdf_file, png_file, radius):
    
    mols = SDMolSupplier(sdf_file, removeHs=False)
    mols = [m for m in mols if m]
    
    fps_sizes = np.arange(5,12)
    corrs, timings = speed_tests(mols, radius, 2**fps_sizes)
    
    sns.set(font_scale=1.5)
    f, ax  = plt.subplots(1, 1, figsize=(7, 5), dpi=300)
    ax.plot(fps_sizes, corrs, '-o')
    
    ax.set_xlabel('size of fingerprint ($log_2$ scale)')
    ax.set_ylabel('Correlation')
    ax.set_xticks(fps_sizes)
    ax.set_xticklabels(['%d'%2**p for p in fps_sizes])        
    
    for i, t in enumerate(timings):
        ax.annotate('%.1fs'%t, (fps_sizes[i], corrs[i]-0.05), 
                    fontsize='small')
            
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.2, right=0.95)
    f.savefig(png_file)
    
def test_MCS(sdf_file):
    
    mols = SDMolSupplier(sdf_file, removeHs=False)
    mols = np.array([m for m in mols if m])
    nns_ids = get_Tanimoto_NNs(mols, mols, 3, fps_nbits=512, order=1, nns=10)
     
    MCSs, MCS_matches, NN_mols, NN_MCS_matches = get_MCSs(mols, mols, 
                                                          nns_indices=nns_ids)
    
def test_preprocess_mols(sdf_file, session_id):
    
    try:
        rmtree(join('uploads', session_id))
    except Exception as e:
        print(e)
        
    mols = SDMolSupplier(sdf_file, removeHs=False)
    mols = np.array([m for m in mols if m])
    df = preprocess_mols(mols, session_id)
    
    try:
        rmtree(join('uploads', session_id))
    except Exception as e:
        print(e)
        
    return df
                    
if __name__ == '__main__':
#    time_fps('SelVie_ChEMBL.sdf', 'fps_timing.png', 3)
#    test_MCS('SelVie_ChEMBL_test.sdf')
    df = test_preprocess_mols('SelVie_ChEMBL_test.sdf', 'test')
    