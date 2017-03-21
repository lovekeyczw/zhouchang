
feature_QLC=[];



for i=1:size(P_protein_a)
    SEQ=P_protein_a(i);
	FF=mctd(SEQ);
    feature_QLC(i,:)=FF;
end
