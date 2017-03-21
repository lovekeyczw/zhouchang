load('Positive_AC.mat')
[nrows,ncols]=size(Positive_AC)
filename='Positive_AC.txt'
fid=fopen(filename,'w');
for i=1:nrows
    fprintf(fid,'%s\n',Positive_AC{i,1})
end
fclose(fid)