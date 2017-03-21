function P=PYH_6(PR,Ma)
alfabeto=['A' 'C' 'D' 'E' 'F' 'G' 'H' 'I' 'K' 'L' 'M' 'N' 'P' 'Q' 'R' 'S' 'T' 'V' 'W' 'Y'];
for j=1:length(PR)
    if find(PR(j)==alfabeto)
        P(j,:)=Ma(find(PR(j)==alfabeto),:);
    else
        P(j,1:6)=0;
    end
end