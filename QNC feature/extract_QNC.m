load('phy_6.mat');

A_feature_QNC=[];

for i=1:size(P_protein_a)
    SEQ=P_protein_a(i);
	SEQ=cell2mat(SEQ);
    P=PYH_6(SEQ,phy_6);
	FF=AutoCov(P,30);
    A_feature_QNC(i,:)=FF;
	kd = mod(i,100);
	if kd==0
		prin = i;
		prin
	end
end






