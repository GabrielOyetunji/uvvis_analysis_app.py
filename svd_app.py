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

# File upload
st.title("UV-Vis Kinetics Analysis Tool")
uploaded_file = st.file_uploader("Upload CSV data", type="csv")

if uploaded_file is not None:
    assay_df = load_data(uploaded_file)
    assay_df = assay_df.drop(index=0).reset_index(drop=True)

    # Build absorbance matrix
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
    index2 = st.sidebar.number_input("Reference Length (index2)", value=10, step=1)
    k = st.sidebar.slider("SVD Components (k)", 1, 20, 5)
    show_scree = st.sidebar.checkbox("Show Singular Value Plot")

    # Reference subtraction
    reference_spectrum = np.mean(test_matrix[:index2, :], axis=0)
    test_matrix_ref = test_matrix - reference_spectrum

    # SVD
    U, s, Vt = svd(test_matrix_ref, full_matrices=False)

    def reconstruct_svd(U, s, Vt, k):
        S_k = np.diag(s[:k])
        return U[:, :k] @ S_k @ Vt[:k, :]

    filtered_matrix = reconstruct_svd(U, s, Vt, k)

    # Scree plot
    if show_scree:
        st.subheader("Singular Value Plot (Scree)")
        fig_scree, ax = plt.subplots()
        ax.plot(range(1, len(s) + 1), s, 'o-', label='Singular Values')
        ax.set_xlabel("Component Index")
        ax.set_ylabel("Singular Value")
        ax.set_title("Scree Plot")
        st.pyplot(fig_scree)

    # Slice by wavelength
    rows_in_range = np.where((wavelengths >= min_wl) & (wavelengths <= max_wl))[0]
    wl_selected = wavelengths[rows_in_range]
    test_sliced = test_matrix_ref[rows_in_range, :]
    filtered_sliced = filtered_matrix[rows_in_range, :]

    # Time slice
    index_end = index0 + index1
    slice_original = test_sliced[index0:index_end, :]
    slice_filtered = filtered_sliced[index0:index_end, :]
    residual = slice_original - slice_filtered

    # Plots
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

    # Export section
    st.subheader("Download Results")

    df_filtered = pd.DataFrame(filtered_sliced, index=wl_selected)
    df_residual = pd.DataFrame(residual, index=wl_selected)
    df_U = pd.DataFrame(U)
    df_s = pd.DataFrame(s)
    df_Vt = pd.DataFrame(Vt)

    st.download_button("Download Filtered Matrix", data=df_filtered.to_csv(), file_name="filtered_matrix.csv")
    st.download_button("Download Residual", data=df_residual.to_csv(), file_name="residual.csv")
    st.download_button("Download U Matrix", data=df_U.to_csv(index=False), file_name="U_matrix.csv")
    st.download_button("Download Singular Values", data=df_s.to_csv(index=False), file_name="singular_values.csv")
    st.download_button("Download Vt Matrix", data=df_Vt.to_csv(index=False), file_name="Vt_matrix.csv")
