clc;
clear;
X=readmatrix('data_dd.xlsx');
[idx,C] = kmeans(X,4);
figure;
plot(X(:,1),X(:,2),'k.');
figure;
plot(X(idx==1,1),X(idx==1,2),'ro','MarkerSize',3)
hold on
plot(X(idx==2,1),X(idx==2,2),'bo','MarkerSize',3)
hold on
plot(X(idx==3,1),X(idx==3,2),'go','MarkerSize',3)
hold on
plot(X(idx==4,1),X(idx==4,2),'yo','MarkerSize',3)
plot(C(:,1),C(:,2),'kx',...
     'MarkerSize',10,'LineWidth',3) 
legend('Clustering 1','Clustering 2','Clustering 3','Clustering center',...
       'Location','NW')
hold off
C