from doctest import master
import pandas as pd
from rdkit import Chem
import numpy as np
import pickle

def convert_SMILES_to_canon(df, token):
    df['SMILES'] = df.apply(lambda x: '<'+token+'>|'+Chem.CanonSmiles(x['SMILES']), axis=1)
    return df

class Normalizer():

    def __init__(self, array) -> None:
        self.array = array
        self.ymax = np.max(array)
        self.ymin = np.min(array)

    def fit_transform(self, array, ymax, ymin):
        '''Called when initialized for the first time on train'''
        normalized_y = [(yi-ymin)/(ymax-ymin) for yi in array]
        return normalized_y[0]

    def transform(self, array, ymax, ymin):
        '''Called after loading Normalizer pickle file on valid/test'''
        normalized_y = [(yi-ymin)/(ymax-ymin) for yi in array]
        return normalized_y[0]


if __name__ == "__main__": 
    lipo = pd.read_csv('D:/Jupyter/Chemformer/original_datasets/Lipophilicity.csv')
    esol = pd.read_csv('D:/Jupyter/Chemformer/original_datasets/delaney-processed.csv')
    frees = pd.read_csv('D:/Jupyter/Chemformer/original_datasets/FreeSolv.txt', sep=';')

    lipo = convert_SMILES_to_canon(lipo, token='Lipo')
    esol = convert_SMILES_to_canon(esol, token='ESOL')
    frees = convert_SMILES_to_canon(frees, token='FreeS')

    lipo.to_csv('D:/Jupyter/Chemformer/canonical_datasets/Lipo.csv')
    esol.to_csv('D:/Jupyter/Chemformer/canonical_datasets/ESOL.csv')
    frees.to_csv('D:/Jupyter/Chemformer/canonical_datasets/FreeSolv.csv')

    lipo = pd.read_csv('D:/Jupyter/Chemformer/canonical_datasets/Lipo.csv')[['exp', 'SMILES']]
    esol = pd.read_csv('D:/Jupyter/Chemformer/canonical_datasets/ESOL.csv')[['logS', 'SMILES']]
    frees = pd.read_csv('D:/Jupyter/Chemformer/canonical_datasets/FreeSolv.csv')[[' Mobley group calculated value (GAFF) (kcal/mol)', 'SMILES']]
    lipo.rename(columns={'exp':'pXC50'}, inplace=True)
    esol.rename(columns={'logS':'pXC50'}, inplace=True)
    frees.rename(columns={' Mobley group calculated value (GAFF) (kcal/mol)':'pXC50'}, inplace=True)

    # 75%, 10%, 15% split - train, valid, test
    trainlipo, validatelipo, testlipo = np.split(lipo.sample(frac=1, random_state=42), [int(.75*len(lipo)), int(.85*len(lipo))])
    trainesol, validateesol, testesol = np.split(esol.sample(frac=1, random_state=42), [int(.75*len(esol)), int(.85*len(esol))])
    trainfrees, validatefrees, testfrees = np.split(frees.sample(frac=1, random_state=42), [int(.75*len(frees)), int(.85*len(frees))])

    trainlipo['SET'] = 'train'
    validatelipo['SET'] = 'valid'
    testlipo['SET'] = 'test'

    trainesol['SET'] = 'train'
    validateesol['SET'] = 'valid'
    testesol['SET'] = 'test'

    trainfrees['SET'] = 'train'
    validatefrees['SET'] = 'valid'
    testfrees['SET'] = 'test'

    ################ Normalize target feature from 0-1
    ######### Lipo
    lipotrain_y = np.array(trainlipo['pXC50']).reshape(1,-1)
    norm = Normalizer(lipotrain_y)
    trainlipo['pXC50'] = norm.fit_transform(lipotrain_y, norm.ymin, norm.ymax)

    with open('D:/Jupyter/Chemformer/normalizers/LipoNormalizer.pickle', 'wb') as filehandler:
        pickle.dump(norm, filehandler)

    test_y = np.array(testlipo['pXC50']).reshape(1,-1)
    testlipo['pXC50'] = norm.transform(test_y, norm.ymax, norm.ymin)
    validate_y = np.array(validatelipo['pXC50']).reshape(1,-1)
    validatelipo['pXC50'] = norm.transform(validate_y, norm.ymax, norm.ymin)

    ######## ESOL
    esoltrain_y = np.array(trainesol['pXC50']).reshape(1,-1)
    norm = Normalizer(esoltrain_y)
    trainesol['pXC50'] = norm.fit_transform(esoltrain_y, norm.ymin, norm.ymax)

    with open('D:/Jupyter/Chemformer/normalizers/ESOLNormalizer.pickle', 'wb') as filehandler:
        pickle.dump(norm, filehandler)

    test_y = np.array(testesol['pXC50']).reshape(1,-1)
    testesol['pXC50'] = norm.transform(test_y, norm.ymax, norm.ymin)
    validate_y = np.array(validateesol['pXC50']).reshape(1,-1)
    validateesol['pXC50'] = norm.transform(validate_y, norm.ymax, norm.ymin)

    ####### FreeS
    freestrain_y = np.array(trainfrees['pXC50']).reshape(1,-1)
    norm = Normalizer(freestrain_y)
    trainfrees['pXC50'] = norm.fit_transform(freestrain_y, norm.ymin, norm.ymax)

    with open('D:/Jupyter/Chemformer/normalizers/FreeSNormalizer.pickle', 'wb') as filehandler:
        pickle.dump(norm, filehandler)

    test_y = np.array(testfrees['pXC50']).reshape(1,-1)
    testfrees['pXC50'] = norm.transform(test_y, norm.ymax, norm.ymin)
    validate_y = np.array(validatefrees['pXC50']).reshape(1,-1)
    validatefrees['pXC50'] = norm.transform(validate_y, norm.ymax, norm.ymin)

    
    trainlipo.to_csv('D:/Jupyter/Chemformer/normalized_split_datasets/Lipo_train.csv')
    validatelipo.to_csv('D:/Jupyter/Chemformer/normalized_split_datasets/Lipo_valid.csv')
    testlipo.to_csv('D:/Jupyter/Chemformer/normalized_split_datasets/Lipo_test.csv')

    validateesol.to_csv('D:/Jupyter/Chemformer/normalized_split_datasets/ESOL_valid.csv')
    testesol.to_csv('D:/Jupyter/Chemformer/normalized_split_datasets/ESOL_test.csv')

    validatefrees.to_csv('D:/Jupyter/Chemformer/normalized_split_datasets/FreeSolv_valid.csv')
    testfrees.to_csv('D:/Jupyter/Chemformer/normalized_split_datasets/FreeSolv_test.csv')

    print(len(trainlipo), len(trainfrees), len(trainesol), 'are lengths of LIPO | FREES | ESOL training datasets')
    print(len(validatelipo), len(validatefrees), len(validateesol), 'are lengths of LIPO | FREES | ESOL validation datasets')
    print(len(testlipo), len(testfrees), len(testesol), 'are lengths of LIPO | FREES | ESOL test datasets')

    #### Upsample freesolve and ESOL

    trainfrees_upsampled3 = pd.concat([trainfrees, trainfrees, trainfrees])
    trainesol_upsampled2 = pd.concat([trainesol, trainesol])

    trainesol_upsampled2.to_csv('D:/Jupyter/Chemformer/normalized_split_datasets/ESOL_train.csv')
    trainfrees_upsampled3.to_csv('D:/Jupyter/Chemformer/normalized_split_datasets/FreeSolv_train.csv')
    print('After upsampling..')
    print(len(trainfrees_upsampled3), len(trainesol_upsampled2), 'are lengths of FREES | ESOL training datasets')
    #### Combine all 3 datasets to 1 main dataset with train/valid/test sets

    master_dataset = pd.concat([trainlipo, trainesol_upsampled2, trainfrees_upsampled3, 
                                validatelipo, validateesol, validatefrees,
                                testlipo, testesol, testfrees
                                ])

    master_dataset.to_csv('D:/Jupyter/Chemformer/Regr_molecule_property_dataset.csv')


    predict_esol = pd.concat([trainesol_upsampled2, testesol])
    predict_lipo = pd.concat([trainlipo, testlipo])
    predict_frees = pd.concat([trainfrees_upsampled3, testfrees])

    predict_esol.to_csv('D:/Jupyter/Chemformer/ESOL.csv')
    predict_lipo.to_csv('D:/Jupyter/Chemformer/Lipo.csv')
    predict_frees.to_csv('D:/Jupyter/Chemformer/FreeS.csv') 