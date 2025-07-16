clc
clear
r101=importdata('data_stations.txt');
r102=importdata('data_Demand forecasting.txt');
vertexs=r101(:,2:3);
node_size=r102(:,1);
figure;
c = hot;
c = flipud(c);
plot(vertexs(1,1),vertexs(1,2),'s','linewidth',1,'MarkerEdgeColor','b','MarkerFaceColor','b','MarkerSize',12);hold on;
scatter(vertexs(2:79,1), vertexs(2:79,2), node_size*4, node_size, 'filled');
colormap(c); % ������ɫӳ��"hot"��"cool"��"spring"
colorbar;
box on; % ��ӱ߿�
for i = 1:size(r101,1)
    text(vertexs(i,1),vertexs(i,2),['   ' num2str(i-1)]);
end
% ����x��y��ķ�Χ
xlim([118.0979, 118.1102]);
