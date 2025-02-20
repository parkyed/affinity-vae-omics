import os
import numpy as np
import pandas as pd


# function to split counts matrix into a set of 1D np arrays, append class label and save as individual files

def matrixsplitsave(omic_mat, class_vector, path_out):
    sample_names_vec = omic_mat.columns

    gene_array_lst = list(range(omic_mat.shape[1]))

    for idx in range(omic_mat.shape[1]):
        gene_array_lst[idx] = omic_mat.iloc[:, idx].values

    # Make sure that the user has a directory named input_arrays.
    # if not, create it.
    input_arrays_path = os.path.join(path_out, 'input_arrays')
    if not os.path.exists(input_arrays_path):
        os.makedirs(input_arrays_path)
        print(f"Directory '{input_arrays_path}' created.")

# save each array with a file name prefixed with its class label

    for sid, sample, label in zip(sample_names_vec, gene_array_lst, class_vector):
        sample_name = str(label) + '_' + str(sid) + '.npy'
        sample = np.array(sample) #.reshape(1, -1)  # edited out while testing the impact of 1d vectors

        # Build save path
        save_path = os.path.join(path_out, 'input_arrays', sample_name)

        # save the sample arrays
        np.save(save_path, sample)


# parse arguments to extract file paths for saving down the data

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()

    # -db DATABASE -u USERNAME -p PASSWORD -size 20 # MTN is this needed?
    parser.add_argument("--matrix_file", help="Path to rna-seq counts matrix csv file")
    parser.add_argument("--labels_file", help="Path to class labels csv file")
    parser.add_argument("--output_path", help="Output path to save data")

    args = parser.parse_args()

    matrix_path = args.matrix_file
    labels_path = args.labels_file
    output_path = args.output_path

    # read in counts matrix and classes
    data_matrix = pd.read_csv(matrix_path, header=0)
    sample_classes = pd.read_csv(labels_path, header=None).values.flatten()
    sample_classes_unique = np.unique(sample_classes).reshape(1, -1)  # -1 allows for variable number of classes

    # split matrix and save as individual files
    matrixsplitsave(omic_mat=data_matrix, class_vector=sample_classes, path_out=output_path)

    # save the unique list of classes as a csv file
    np.savetxt(os.path.join(output_path,'class_lst.csv'), sample_classes_unique, fmt="%s", delimiter=",")

# next steps to add
# randomised train test split
# build the matrix splitter as a class to handle train / test versions
