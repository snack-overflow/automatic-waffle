
%%% Plot FFT of signal data 'fdata'

% sample = fdata;
% Y = fft(sample,length(sample));
% Pyy = Y.*conj(Y)/length(Y)*2;
% f = Fs/length(Y)*(0:length(Y)/2-1);
% title('Power spectral density')
% xlabel('Frequency (Hz)')
% figure(); plot(f,Pyy(1:length(Y)/2))


% % %  For converting the 90750x 132 matrix to a 1375 x 132 x 66 matrix

% left_signals = permute(reshape(left_signals, 1375,66, 132), [1 3 2]);
% right_signals = permute(reshape(right_signals, 1375,63, 132), [1 3 2]);

% % % For power spectrum density
% power = (norm(temp)^2)/length(temp);
% psd1=spectrum(temp,1024);
% hpsd = dspdata.psd(psd1,'Fs',Fs);
% figure();plot(hpsd);


% % % For plotting graphs
%figure(); plot([right_signals(1:1000,16,1) left_signals(1:1000,16,1)])
%figure(); subplot(2,1,1); plot(right_signals(1:1000,16,1)); subplot(2,1,2); plot(right_signals(1:1000,16,2))

% % % For csp channel ranks

matrix=csvread('csp_matrix.csv');

selection_matrix = [];
selection_matrix= [selection_matrix matrix(:,1:10)];
selection_matrix= [selection_matrix matrix(:,123:end)];
ranks = [];
for i =1:1:20
    [rank_coeff,rank_index] = sort(abs(selection_matrix(:,i)),'descend');
    ranks = [ranks; (rank_index.')];
end

csvwrite('ranks.csv',ranks);
csvwrite('transpose_ranks.csv',ranks');

