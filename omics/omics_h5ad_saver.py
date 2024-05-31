# Importing the neccecary modules
import argparse
import anndata
import scanpy
import os
import numpy as np

# Script to split an .h5ad file into a set of 1D numpy arrays,
# append class (cell_type) label and save as individual files.


def matrix_split_save(sc_matrix, cell_types, output_path):
    """
        Split obs (cells) from a sparce matrix (extracted from a .h5ad file)
        and save into individual .npy files.

        Parameters:
        -----------
        sc_matrix : scipy.sparse.csr_matrix
            Sparce matrix data (from a .h5ad file) to be split and saved.
        cell_types : list
            A List containing class labels (cell type) for each sample (cell).
        output_path : str
            Path to the directory where the data will be saved.

        Returns:
        --------
        None (files are saved to output_path directory)

        Notes:
        ------
        This function splits the input matrix `sc_matrix` into individual numpy arrays based on sample classes
        provided in `cell_types` and saves them in separate files in the directory specified by `output_path`.
        Each saved file is named using the sample's class label and sample ID.

        Example:
        --------
        sc_matrix = ...  # Load your omic matrix data
        cell_types = ['Class1', 'Class2', 'Class1', ...]  # Class labels for each sample
        output_directory = '/path/to/output'  # Directory where the data will be saved
        matrix_split_save(sc_matrix, cell_types, output_directory)
        """

    cell_array_lst = list(range(sc_matrix.shape[0]))

    for idx in range(sc_matrix.shape[0]):
        cell_array_lst[idx] = sc_matrix[idx, :].A.reshape(-1,)  # added reshape to make sure that my arrays are 1D
        print(f"Processed cell {idx + 1}/{sc_matrix.shape[0]}")

    # cell_array_lst = sc_matrix.A.tolist()  # Optimized version of above, encounters issues with lack of memory.

    # Make sure that the user has a directory named input_arrays.
    # if not, create it.
    input_arrays_path = os.path.join(output_path, 'input_arrays')
    if os.path.exists(input_arrays_path):
        print(f"Warning '{input_arrays_path}' directory already exists")
        warning_printed = True
    else:
        os.makedirs(input_arrays_path)
        print(f"Directory '{input_arrays_path}' created.")
        warning_printed = False

    # save each array with a file name prefixed with its class label
    #idx = 0
    cell_names = cell_types.index
    for idx, (sid, sample, label) in enumerate(zip(cell_names, cell_array_lst, cell_types)):
        sample_name = str(label) + '_' + str(sid) + '.npy'
        # sample = np.array(sample) #.reshape(1, -1)
        # print(sample.shape)

        # Build save path
        save_path = os.path.join(output_path, 'input_arrays', sample_name)

        # save the sample arrays
        np.save(save_path, sample)
        idx = idx + 1
        print(f"Saved sample {idx}/{len(cell_names)}")

    # check that the number of files in input_arrays is equal to the number of saved samples
    if idx != len(os.listdir(input_arrays_path)):
        print("Warning: Number of files saved does not match the number of samples in input_arrays")

    # Warning message at the end to indicate data may have been overwritten or appended to input_arrays,
    # (only if the first one was printed).
    if warning_printed:
        print(f"Warning: Samples saved to previously generated directory '{input_arrays_path}'.")

# parse arguments to extract file paths for saving down the data


if __name__ == '__main__':

    # Creating an ArgumentParser object
    parser = argparse.ArgumentParser()

    # Adding arguments for the script
    # Argument for the path to the .h5ad file
    parser.add_argument(
        "--h5ad_file",
        help="Path to h5ad file (includes counts matrix and cell_type column in the metadata)"
    )

    # Argument for the output path to save data
    parser.add_argument(
        "--output_path",
        help="Output path to save data"
    )

    # Argument for the column name in the metadata that contains the cell type information
    parser.add_argument(
        "--cell_type_column_name",
        default="cell_type",
        help="The column name in the metadata that contains the cell type information. Default is 'cell_type'"
    )

    # Argument for the number of most variable genes to include in the analysis
    parser.add_argument(
        "--n_most_variable_genes",
        default=2000,
        help="The number of the most vaiable genes to be included, Default is '2000'"
    )

    # Parsing the command-line arguments
    args = parser.parse_args()

    # Assigning values from parsed arguments to variables
    h5ad_file = args.h5ad_file
    output_path = args.output_path
    cell_type_column_name = args.cell_type_column_name
    ngenes = args.n_most_variable_genes

    # read in the .h5ad file
    adata = anndata.read_h5ad(h5ad_file)
    print(r"read in the .h5ad file")

    #identify the most variable genes
    scanpy.pp.highly_variable_genes(adata, n_top_genes = ngenes)

    # downsample to only inlude the most variable genes
    adata = adata[:, adata.var.highly_variable]
    print(f"Subsetted the data to the top {ngenes} highly variable genes")

    # Extract the counts matrix
    sc_matrix = adata.X
    print(r"Extracted the counts matrix")

    # Extract the cell_type column from the metadata
    cell_types = adata.obs[cell_type_column_name]
    # Modify cell_types to remove commas
    cell_types = cell_types.str.replace(', ', '_')
    # Modify cell_types to remove spaces
    cell_types = cell_types.str.replace(' ', '-')

    # Identify the unique classes
    cell_types_unique = np.unique(cell_types).reshape(1, -1)

    # save the unique list of classes as a csv file
    np.savetxt(os.path.join(output_path, 'class_lst.csv'), cell_types_unique, fmt='%s',  delimiter=",")
    print(f"Unique classes saved to {output_path}\\class_lst.csv")

    # split matrix and save as individual files
    matrix_split_save(sc_matrix, cell_types, output_path)
