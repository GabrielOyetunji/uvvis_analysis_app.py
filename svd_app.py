import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd

# Load data
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    return df

# Upload CSV
st.title("UV-Vis Kinetics Analysis Tool")
uploaded_file = st.file_uploader("Upload CSV data", type="csv")

if uploaded_file is not None:
    assay_df = load_data(uploaded_file)
    assay_df = assay_df.drop(index=0).reset_index(drop=True)

    # Build test_matrix
    absorbance_data = []
    for i in range(1, assay_df.shape[1], 2):
        col_values = pd.to_numeric(assay_df.iloc[:, i], errors='coerce')
        absorbance_data.append(col_values.values)
    test_matrix = np.array(absorbance_data).T
    wavelengths = pd.to_numeric(assay_df.iloc[:, 0], errors='coerce').values

    # Sidebar controls
    st.sidebar.header("Parameters")
    min_wl = st.sidebar.number_input("Min Wavelength", value=400)
    max_wl = st.sidebar.number_input("Max Wavelength", value=700)
    index0 = st.sidebar.number_input("Start Scan Index", value=100, step=1)
    index1 = st.sidebar.number_input("Number of Scans", value=15, step=1)
    index2 = st.sidebar.number_input("Reference Range Length", value=10, step=1)
    k = st.sidebar.slider("Number of SVD Components", 1, 20, 5)

    # Reference subtraction
    reference_spectrum = np.mean(test_matrix[:index2, :], axis=0)
    test_matrix_ref = test_matrix - reference_spectrum

    # SVD
    U, s, Vt = svd(test_matrix_ref, full_matrices=False)

    def reconstruct_svd(U, s, Vt, k):
        S_k = np.diag(s[:k])
        return U[:, :k] @ S_k @ Vt[:k, :]

    filtered_matrix = reconstruct_svd(U, s, Vt, k)

    # 📉 Scree Plot
    st.subheader("Scree Plot (Singular Values)")
    fig_scree, ax_scree = plt.subplots()
    ax_scree.plot(range(1, len(s)+1), s, marker='o', linestyle='-')
    ax_scree.axvline(k, color='r', linestyle='--', label=f'Selected k = {k}')
    ax_scree.set_xlabel("Component Number")
    ax_scree.set_ylabel("Singular Value")
    ax_scree.set_title("Scree Plot")
    ax_scree.grid(True)
    ax_scree.legend()
    st.pyplot(fig_scree)

    # Wavelength filtering
    rows_in_range = np.where((wavelengths >= min_wl) & (wavelengths <= max_wl))[0]
    wl_selected = wavelengths[rows_in_range]
    test_matrix_sliced = test_matrix_ref[rows_in_range, :]
    filtered_matrix_sliced = filtered_matrix[rows_in_range, :]

    # Time slicing
    index_end = index0 + index1
    slice_original = test_matrix_sliced[index0:index_end, :]
    slice_filtered = filtered_matrix_sliced[index0:index_end, :]
    residual = slice_original - slice_filtered

    # Plot comparison
    st.subheader("Matrix Comparison")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(slice_original, aspect='auto', cmap='viridis')
    axes[0].set_title('Original Slice')

    axes[1].imshow(slice_filtered, aspect='auto', cmap='viridis')
    axes[1].set_title('Filtered Slice')

    axes[2].imshow(residual, aspect='auto', cmap='bwr')
    axes[2].set_title('Residual')

    st.pyplot(fig)

    # Autocorrelation
    def autocorr(vec):
        return np.sum(vec[:-1] * vec[1:])

    u_autocorr = [autocorr(U[:, i]) for i in range(k)]
    v_autocorr = [autocorr(Vt[i, :]) for i in range(k)]

    st.subheader("Autocorrelation")
    st.write("U Components:", u_autocorr)
    st.write("V Components:", v_autocorr)

    # Prepare labeled exports
    scan_columns = [f"Scan_{i}" for i in range(test_matrix.shape[1])]
    scan_index_labels = [f"Scan_{i}" for i in range(index0, index0 + index1)]

    df_filtered = pd.DataFrame(filtered_matrix_sliced, index=wl_selected, columns=scan_columns)
    df_residual = pd.DataFrame(residual, index=scan_index_labels, columns=scan_columns)
    df_reference_subtracted = pd.DataFrame(test_matrix_ref, index=range(test_matrix_ref.shape[0]), columns=scan_columns)

    # Download buttons
    st.subheader("Download Results")
    st.download_button("Filtered Matrix", data=df_filtered.to_csv(index=True), file_name="filtered_matrix_labeled.csv")
    st.download_button("Residual Slice", data=df_residual.to_csv(index=True), file_name="residual_matrix_labeled.csv")
    st.download_button("Reference Subtracted Matrix", data=df_reference_subtracted.to_csv(index=True), file_name="reference_subtracted.csv")
    st.download_button("U Matrix", data=pd.DataFrame(U).to_csv(index=False, header=False), file_name="U_matrix.csv")
    st.download_button("Singular Values", data=pd.DataFrame(s).to_csv(index=False, header=False), file_name="singular_values.csv")
    st.download_button("Vt Matrix", data=pd.DataFrame(Vt).to_csv(index=False, header=False), file_name="Vt_matrix.csv")
