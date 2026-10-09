# SVM results (CIFAR-10, airplane vs automobile)

Settings: {'classes': (0, 1), 'hog_orientations': 9, 'hog_pixels_per_cell': 8, 'hog_cells_per_block': 2, 'pca_components': 64, 'C': 1.0, 'encrypt_model': True}, encrypted model: True, test images: 2000

## Table IV

| Plaintext ACC | Training Time (s) | Encrypted ACC | Inference Latency (s) |
|---|---|---|---|
| 88.95% | 3.244 | 88.95% | 2.43 |

## Additional measurements

| test feature encryption (s) | decryption (s) | prediction agreement | max |score diff| |
|---|---|---|---|
| 108.33 | 0.58 | 100.00% | 1.27e-04 |
