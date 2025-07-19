import torch
import os
import torchaudio
import torchaudio.transforms as T

class AudioPipeline(torch.nn.Module):
    def __init__(
        self,
        freq=16000,
        n_fft=1024,
        n_mel=128,
        win_length=1024,
        hop_length=256,
    ):
        super().__init__()
        self.freq=freq
        pad = int((n_fft-hop_length)/2)
        self.spec = T.Spectrogram(n_fft=n_fft, win_length=win_length, hop_length=hop_length,
            pad=pad, power=None,center=False, pad_mode='reflect', normalized=False, onesided=True)

        self.mel_scale = T.MelScale(n_mels=n_mel, sample_rate=freq, n_stft=n_fft // 2 + 1)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        shift_waveform = waveform
        # Convert to power spectrogram
        spec = self.spec(shift_waveform)
        spec = torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + 1e-6)
        # Convert to mel-scale
        mel = self.mel_scale(spec)
        return mel

device = "cpu"

hifigan = torch.hub.load("./hifi-gan-0.3.1", "hifigan_48k",source='local', force_reload=False).to(device)

# Load audio
SR = 48000
wav, sr = torchaudio.load("test.wav")
if sr != SR:
    wav = torchaudio.functional.resample(wav, sr, SR)
    sr = SR

audio_pipeline = AudioPipeline(freq=sr,
                                n_fft=1024,
                                n_mel=80,
                                win_length=1024,
                                hop_length=256)
mel = audio_pipeline(wav)
out = hifigan(mel)

wav_out = out.unsqueeze(0).detach().cpu()

torchaudio.save("test_out_48k.wav", wav_out, sr)