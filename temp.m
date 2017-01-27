% Y = fft(sample,length(sample));
% Pyy = Y.*conj(Y)/length(Y);
% f = Fs/length(Y)*(0:length(Y)-1);
% plot(f,Pyy(1:length(Y)))
% title('Power spectral density')
% xlabel('Frequency (Hz)')

sample = fdata;
Y = fft(sample,length(sample));
Pyy = Y.*conj(Y)/length(Y)*2;
f = Fs/length(Y)*(0:length(Y)/2-1);
title('Power spectral density')
xlabel('Frequency (Hz)')
figure(); plot(f,Pyy(1:length(Y)/2))
