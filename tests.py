"""
These are tests for SelVie.

"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit.Chem import SDMolSupplier

from utils import speed_tests, get_MCSs, get_NNs


def time_fps(sdf_file, png_file):
    
    mols = SDMolSupplier(sdf_file, removeHs=False)
    mols = [m for m in mols if m]
    
    fps_sizes = np.arange(5,12)
    fps_radiuses = np.arange(2, 5)
    
    sns.set(font_scale=1.25)
    f, axs  = plt.subplots(1, 3, figsize=(15, 5), dpi=300, sharey=True)
    axs[0].set_ylabel('Correlation')
    for ax, radius in zip(axs, fps_radiuses):
        corrs, timings = speed_tests(mols, radius, 2**fps_sizes)
        ax.plot(fps_sizes, corrs, '-o')
        ax.set_xticks(fps_sizes)
        ax.set_xlabel('size of fingerprint ($log_2$ scale)')
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
    nns_ids = get_NNs(mols, mols, 3, fps_nbits=512, order=1, nns=10)
     
    MCSs, MCS_matches, NN_mols, NN_MCS_matches = get_MCSs(mols, mols, 
                                                          nns_indices=nns_ids)
    
if __name__ == '__main__':
    #time_fps('SelVie_ChEMBL_test.sdf', 'fps_timing.png')
    test_MCS('SelVie_ChEMBL_test.sdf')
    