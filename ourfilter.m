function [output, n]  = ourfilter(channel_data,buffer, Rp, Rs, Fs)

 
output=[];
freqs=[8 12;13 30; 8 16; 17 30; 8 20; 21 30];

for i = 1:1:6
    Wp = [freqs(i,1) freqs(i,2)]/(Fs/2); 
    Ws = [freqs(i,1)-buffer freqs(i,2)+buffer]/(Fs/2);
    [n,Wn] = buttord(Wp,Ws,Rp,Rs); 
    [z, p, k] = butter(n,Wn,'bandpass');
    [sos,g] = zp2sos(z,p,k);
    filt = dfilt.df2sos(sos,g);
    fdata = filter(filt,channel_data);
    output=[output fdata];
end
    