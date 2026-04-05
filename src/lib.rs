// iron-librosa: Rust extension module for librosa
// This module is compiled to `librosa._rust` and provides accelerated
// implementations of librosa functions.  Python code checks for the
// presence of this module at runtime and falls back to pure-Python
// implementations when a Rust version is not available.

use pyo3::prelude::*;


mod convert;
mod dct;
mod mel;
mod nn_filter;
mod onset;
mod stft;
mod istft;
mod spectrum_utils;
mod chroma;
mod tuning;
mod phase_vocoder;
mod cqt_vqt;
mod beat;
mod rhythm;

/// The iron-librosa Rust extension module (`librosa._rust`).
#[pymodule]
#[pyo3(name = "_rust")]
fn iron_librosa_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Expose the module version so Python can sanity-check compatibility.
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    // ----- convert -----
    m.add_function(wrap_pyfunction!(convert::hz_to_mel, m)?)?;
    m.add_function(wrap_pyfunction!(convert::mel_to_hz, m)?)?;

    // ----- decompose -----
    // NMF is handled by Python/scikit-learn; no Rust implementation exposed.

    // ----- nn_filter -----
    m.add_function(wrap_pyfunction!(nn_filter::nn_filter, m)?)?;

    // ----- mel -----
    m.add_function(wrap_pyfunction!(mel::mel_project_f64, m)?)?;
    m.add_function(wrap_pyfunction!(mel::mel_project_f32, m)?)?;
    m.add_function(wrap_pyfunction!(mel::mel_filter_f32, m)?)?;

    // ----- dct -----
    m.add_function(wrap_pyfunction!(dct::dct2_ortho_f32, m)?)?;
    m.add_function(wrap_pyfunction!(dct::dct2_ortho_f64, m)?)?;

    // ----- stft -----
    m.add_function(wrap_pyfunction!(stft::stft_power, m)?)?;
    m.add_function(wrap_pyfunction!(stft::stft_complex, m)?)?;
    m.add_function(wrap_pyfunction!(stft::stft_power_f64, m)?)?;
    m.add_function(wrap_pyfunction!(stft::stft_complex_f64, m)?)?;
    m.add_function(wrap_pyfunction!(stft::stft_complex_batch, m)?)?;
    m.add_function(wrap_pyfunction!(stft::stft_complex_f64_batch, m)?)?;

    // ----- onset -----
    m.add_function(wrap_pyfunction!(onset::onset_flux_mean_f32, m)?)?;
    m.add_function(wrap_pyfunction!(onset::onset_flux_mean_ref_f32, m)?)?;
    m.add_function(wrap_pyfunction!(onset::onset_flux_mean_maxfilter_f32, m)?)?;
    m.add_function(wrap_pyfunction!(onset::onset_flux_mean_f64, m)?)?;
    m.add_function(wrap_pyfunction!(onset::onset_flux_mean_ref_f64, m)?)?;
    m.add_function(wrap_pyfunction!(onset::onset_flux_mean_maxfilter_f64, m)?)?;
    m.add_function(wrap_pyfunction!(onset::onset_flux_median_ref_f32, m)?)?;
    m.add_function(wrap_pyfunction!(onset::onset_flux_median_ref_f64, m)?)?;

    // ----- istft (Inverse STFT) -----
    m.add_function(wrap_pyfunction!(istft::istft_f32, m)?)?;
    m.add_function(wrap_pyfunction!(istft::istft_f64, m)?)?;

    // ----- spectrum_utils (dB conversions and spectral reductions) -----
    m.add_function(wrap_pyfunction!(spectrum_utils::rms_spectrogram_f32, m)?)?;
    m.add_function(wrap_pyfunction!(spectrum_utils::rms_spectrogram_f64, m)?)?;
    m.add_function(wrap_pyfunction!(spectrum_utils::rms_time_f32, m)?)?;
    m.add_function(wrap_pyfunction!(spectrum_utils::rms_time_f64, m)?)?;
    m.add_function(wrap_pyfunction!(spectrum_utils::spectral_centroid_f32, m)?)?;
    m.add_function(wrap_pyfunction!(spectrum_utils::spectral_centroid_f64, m)?)?;
    m.add_function(wrap_pyfunction!(spectrum_utils::spectral_centroid_variable_freq_f32, m)?)?;
    m.add_function(wrap_pyfunction!(spectrum_utils::spectral_centroid_variable_freq_f64, m)?)?;
    m.add_function(wrap_pyfunction!(spectrum_utils::spectral_rolloff_f32, m)?)?;
    m.add_function(wrap_pyfunction!(spectrum_utils::spectral_rolloff_f64, m)?)?;
    m.add_function(wrap_pyfunction!(spectrum_utils::spectral_rolloff_variable_freq_f32, m)?)?;
    m.add_function(wrap_pyfunction!(spectrum_utils::spectral_rolloff_variable_freq_f64, m)?)?;
    m.add_function(wrap_pyfunction!(spectrum_utils::spectral_bandwidth_f32, m)?)?;
    m.add_function(wrap_pyfunction!(spectrum_utils::spectral_bandwidth_f64, m)?)?;
    m.add_function(wrap_pyfunction!(spectrum_utils::spectral_bandwidth_auto_centroid_f32, m)?)?;
    m.add_function(wrap_pyfunction!(spectrum_utils::spectral_bandwidth_auto_centroid_f64, m)?)?;
    m.add_function(wrap_pyfunction!(spectrum_utils::spectral_flatness_f32, m)?)?;
    m.add_function(wrap_pyfunction!(spectrum_utils::spectral_flatness_f64, m)?)?;
    m.add_function(wrap_pyfunction!(spectrum_utils::spectral_contrast_band_f32, m)?)?;
    m.add_function(wrap_pyfunction!(spectrum_utils::spectral_contrast_band_f64, m)?)?;
    m.add_function(wrap_pyfunction!(spectrum_utils::spectral_contrast_fused_f32, m)?)?;
    m.add_function(wrap_pyfunction!(spectrum_utils::spectral_contrast_fused_f64, m)?)?;
    m.add_function(wrap_pyfunction!(spectrum_utils::power_to_db_f32, m)?)?;
    m.add_function(wrap_pyfunction!(spectrum_utils::power_to_db_f64, m)?)?;
    m.add_function(wrap_pyfunction!(spectrum_utils::amplitude_to_db_f32, m)?)?;
    m.add_function(wrap_pyfunction!(spectrum_utils::amplitude_to_db_f64, m)?)?;
    m.add_function(wrap_pyfunction!(spectrum_utils::db_to_power_f32, m)?)?;
    m.add_function(wrap_pyfunction!(spectrum_utils::db_to_power_f64, m)?)?;
    m.add_function(wrap_pyfunction!(spectrum_utils::db_to_amplitude_f32, m)?)?;
    m.add_function(wrap_pyfunction!(spectrum_utils::db_to_amplitude_f64, m)?)?;
    m.add_function(wrap_pyfunction!(spectrum_utils::median_filter_harmonic_f32, m)?)?;
    m.add_function(wrap_pyfunction!(spectrum_utils::median_filter_harmonic_f64, m)?)?;
    m.add_function(wrap_pyfunction!(spectrum_utils::median_filter_percussive_f32, m)?)?;
    m.add_function(wrap_pyfunction!(spectrum_utils::median_filter_percussive_f64, m)?)?;
    m.add_function(wrap_pyfunction!(spectrum_utils::hpss_fused_f32, m)?)?;
    m.add_function(wrap_pyfunction!(spectrum_utils::hpss_fused_f64, m)?)?;
    m.add_function(wrap_pyfunction!(spectrum_utils::hpss_fused_batch_f32, m)?)?;
    m.add_function(wrap_pyfunction!(spectrum_utils::hpss_fused_batch_f64, m)?)?;

    // ----- chroma -----
    m.add_function(wrap_pyfunction!(chroma::chroma_filter_f32, m)?)?;
    m.add_function(wrap_pyfunction!(chroma::chroma_filter_f64, m)?)?;
    m.add_function(wrap_pyfunction!(chroma::chroma_project_f32, m)?)?;
    m.add_function(wrap_pyfunction!(chroma::chroma_project_f64, m)?)?;

    // ----- tuning -----
    m.add_function(wrap_pyfunction!(tuning::piptrack_from_spectrogram_f32, m)?)?;
    m.add_function(wrap_pyfunction!(tuning::piptrack_from_spectrogram_f64, m)?)?;
    m.add_function(wrap_pyfunction!(tuning::estimate_tuning_from_piptrack_f32, m)?)?;
    m.add_function(wrap_pyfunction!(tuning::estimate_tuning_from_piptrack_f64, m)?)?;

    // ----- phase_vocoder -----
    m.add_function(wrap_pyfunction!(phase_vocoder::phase_vocoder_f32, m)?)?;
    m.add_function(wrap_pyfunction!(phase_vocoder::phase_vocoder_f64, m)?)?;

    // ----- cqt_vqt -----
    m.add_function(wrap_pyfunction!(cqt_vqt::cqt_project_f32, m)?)?;
    m.add_function(wrap_pyfunction!(cqt_vqt::cqt_project_f64, m)?)?;
    m.add_function(wrap_pyfunction!(cqt_vqt::vqt_project_f32, m)?)?;
    m.add_function(wrap_pyfunction!(cqt_vqt::vqt_project_f64, m)?)?;

    // ----- beat (phase 14 seam) -----
    m.add_function(wrap_pyfunction!(beat::beat_track_dp_f32, m)?)?;
    m.add_function(wrap_pyfunction!(beat::beat_track_dp_f64, m)?)?;

    // ----- rhythm (phase 15 seam) -----
    m.add_function(wrap_pyfunction!(rhythm::tempogram_ac_f32, m)?)?;
    m.add_function(wrap_pyfunction!(rhythm::tempogram_ac_f64, m)?)?;

    Ok(())
}
