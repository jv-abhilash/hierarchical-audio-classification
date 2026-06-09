import numpy as np
import scipy.io.wavfile as wav

def create_synthetic_test_audio(filename="test_siren_simulation.wav", duration=5, sample_rate=22050):
    print(f"Generating synthetic acoustic test file ({duration} seconds)...")
    
    # 1. Timeline array: 110,250 continuous step points
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # 2. Synthesize a modulating Siren wave (pitch rises and falls over time)
    # The frequency sweeps up and down smoothly between 600Hz and 1100Hz
    siren_frequency_modulation = 850 + 250 * np.sin(2 * np.pi * 0.5 * t)
    siren_wave = np.sin(2 * np.pi * siren_frequency_modulation * t)
    
    # 3. Synthesize chaotic background noise (simulating rain/wind textures)
    background_noise = np.random.normal(0, 1, size=t.shape)
    
    # 4. Blend the audio tracks together (60% Siren focus, 40% Background Ambient noise)
    blended_audio = (0.6 * siren_wave) + (0.4 * background_noise)
    
    # 5. Peak amplitude scaling cushion to prevent digital clipping distortion
    blended_audio = blended_audio / np.max(np.abs(blended_audio))
    audio_signal_int16 = (blended_audio * 32767).astype(np.int16)
    
    # 6. Save binary WAV file format to disk
    wav.write(filename, sample_rate, audio_signal_int16)
    print(f"✅ Success! Test sample written to current directory: {filename}")

if __name__ == "__main__":
    create_synthetic_test_audio()