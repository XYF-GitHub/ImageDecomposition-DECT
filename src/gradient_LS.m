function gf = gradient_LS(x,weight)


%%
% g1_d2 = dummy-[dummy(2:end,:); zeros(1, size(dummy,1))];
% g1_d1 = dummy-[zeros(1, size(dummy,1)); dummy(1:end-1,:)];
% g1_d0 = dummy-[zeros(size(dummy,1),1) dummy(:,1:end-1)];
% g1_d3 = dummy-[dummy(:,2:end) zeros(size(dummy,1),1)];
%%
[row,col,~]=size(weight);
xf = reshape(x,row,col);

xf_d0 = xf-[zeros(size(xf,1),1) xf(:,1:end-1)];
xf_d1 = xf-[zeros(1, size(xf,1)); xf(1:end-1,:)];
xf_d2 = xf-[xf(2:end,:); zeros(1, size(xf,1))];
xf_d3 = xf-[xf(:,2:end) zeros(size(xf,1),1)];

gf = 4*(xf_d0.*weight(:,:,1)+xf_d1.*weight(:,:,2)+xf_d2.*weight(:,:,3)+xf_d3.*weight(:,:,4));