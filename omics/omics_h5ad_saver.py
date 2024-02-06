import anndata
import os
import numpy as np

# function to split a h5ad file into a set of 1D np arrays, append class (cell_type) label and save as individual files.


def matrixsplitsave(omic_mat, sample_classes, path_out, adata):
    sample_names_vec = adata.obs_names

    gene_array_lst = list(range(omic_mat.shape[1]))

    for idx in range(omic_mat.shape[1]):
        gene_array_lst[idx] = omic_mat[:, idx].A

    # Make sure that the user has a directory named input_arrays.
    # if not, create it.
    input_arrays_path = os.path.join(path_out, 'input_arrays')
    if not os.path.exists(input_arrays_path):
        os.makedirs(input_arrays_path)
        print(f"Directory '{input_arrays_path}' created.")

# save each array with a file name prefixed with its class label

    for sid, sample, label in zip(sample_names_vec, gene_array_lst, sample_classes):
        sample_name = str(label) + '_' + str(sid) + '.npy'
        sample = np.array(sample)

        # Build save path
        save_path = os.path.join(path_out, 'input_arrays', sample_name)

        # save the sample arrays
        np.save(save_path, sample)

# parse arguments to extract file paths for saving down the data


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()

    #
    parser.add_argument(
        "--h5ad_file",
        help="Path to h5ad file (includes counts matrix and cell_type column in the metadata)"
    )

    parser.add_argument(
        "--output_path",
        help="Output path to save data"
    )

    #
    args = parser.parse_args()

    #
    h5ad_file = args.h5ad_file
    output_path = args.output_path

    # read in h5ad file
    anndata = anndata.read_h5ad(h5ad_file)

    # Extract the counts matrix
    sc_matrix = anndata.X

    # Extract the cell_type column from the metadata
    cell_types = anndata.obs['cell_type']
    # Modify cell_types to remove spaces
    cell_types = cell_types.str.replace(' ', '')

    # Identify the unique classes
    cell_types_unique = np.unique(cell_types).reshape(1, -1)

    # save the unique list of classes as a csv file
    np.savetxt(os.path.join(output_path, 'class_lst.csv'), cell_types_unique, fmt='%s',  delimiter=",")
    print(f"Unique classes saved to {output_path + 'class_lst.csv'}")

    # split matrix and save as individual files
    matrixsplitsave(omic_mat=sc_matrix, sample_classes=cell_types, path_out=output_path, adata=anndata)
