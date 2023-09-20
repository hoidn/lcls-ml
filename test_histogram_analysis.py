
    def test_generate_null_distribution(self):
        null_dist = generate_null_distribution(self.histograms, self.average_histogram, self.roi_x_start, self.roi_x_end, self.roi_y_start, self.roi_y_end)
        self.assertEqual(len(null_dist), 1000)
        
    def test_calculate_p_values(self):
        emd_vals = calculate_emd_values(self.histograms, self.average_histogram)
        null_dist = generate_null_distribution(self.histograms, self.average_histogram, self.roi_x_start, self.roi_x_end, self.roi_y_start, self.roi_y_end)
        p_vals = calculate_p_values(emd_vals, null_dist)
        self.assertEqual(p_vals.shape, (2, 2))
        
    def test_identify_roi_connected_cluster(self):
        emd_vals = calculate_emd_values(self.histograms, self.average_histogram)
        null_dist = generate_null_distribution(self.histograms, self.average_histogram, self.roi_x_start, self.roi_x_end, self.roi_y_start, self.roi_y_end)
        p_vals = calculate_p_values(emd_vals, null_dist)
        cluster = identify_roi_connected_cluster(p_vals, 0.2, self.roi_x_start, self.roi_x_end, self.roi_y_start, self.roi_y_end)
        self.assertEqual(cluster.shape, (2, 2))

if __name__ == "__main__":
    unittest.main()
