function w = get_weight(img, h, w)
    w = ones([[h, w] 4]);
    std_1 = std(img(1:end/2));
    std_2 = std(img(end/2 + 1:end));
    std_ratio = std_1/std_2;
    
    BW = edge(img(:,1:end/2) + img(:,end/2 + 1:end)*std_ratio,'canny',0.1,1.5);
    BW_2 = edge(img(:,1:end/2) + img(:,end/2 + 1:end)*std_ratio,'prewitt',0.1,'both');
    BW = BW + BW_2;
    
    dummy = BW;
    g1_d2 = dummy - [dummy(2:end,:); zeros(1, size(dummy,1))];
    g1_d1 = dummy - [zeros(1, size(dummy,1)); dummy(1:end - 1,:)];
    g1_d0 = dummy - [zeros(size(dummy,1),1) dummy(:,1:end - 1)];
    g1_d3 = dummy - [dummy(:,2:end) zeros(size(dummy,1),1)];
    
    thresh = 0.0001;
    Lvalue = 0.1;
    dummy = ones([512 512]);
    dummy(abs(g1_d0) > thresh) = Lvalue;
    w(:,:,1) = dummy;
    dummy = ones([512 512]);
    dummy(abs(g1_d1) > thresh) = Lvalue;
    w(:,:,2) = dummy;
    dummy = ones([512 512]);
    dummy(abs(g1_d2) > thresh) = Lvalue;
    w(:,:,3) = dummy;
    dummy = ones([512 512]);
    dummy(abs(g1_d3) > thresh) = Lvalue;
    w(:,:,4) = dummy;
end