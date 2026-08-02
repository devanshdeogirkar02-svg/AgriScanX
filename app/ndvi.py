rgb_resized = cv2.resize(original_np, (512, 512))
        R = rgb_resized[:, :, 0].astype(np.float32) / 255.0
        nir = fake_nir_np

        ndvi = (nir - R) / (nir + R + 1e-6)
        ndvi = np.clip(ndvi, -1.0, 1.0)