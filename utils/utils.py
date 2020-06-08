"""
Utility functions for SelVie
"""
from uuid import uuid4
from os import remove
from os.path import join
from base64 import b64encode, b64decode
from rdkit.Chem import Draw, SDMolSupplier


def DrawAsBase64PNG(mol, legend='', highlightAtoms=[], molSize=(200, 200)):
    """A helper function to depict a moelcular structure and encode it as a
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
    
    try:
        if filename[-4:].lower()=='.sdf':
            
            # Generate random file name and save contents to a file
            unique_fname = join('uploads', '%s.sdf'%str(uuid4()))
            with open(unique_fname, 'w') as fh:
                fh.write(decoded.decode('utf-8'))
                
            mols = SDMolSupplier(join('uploads', 'file.sdf'))
            n_mols = len(mols)
            mols = [m for m in mols if m]
            n_sucess = len(mols)
            
            try:
                remove(unique_fname)
            except Exception as e:
                #This is not critical
                print(e)
                
            return mols, 'Loaded %d/%d molecules'%(n_mols, n_sucess)
        else:
            return [], 'The file has a wrong format.'
        
    except Exception as e:
        print(e)
        return [], 'Error occured during processing of a file.'

    