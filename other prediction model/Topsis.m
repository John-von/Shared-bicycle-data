data=[rmse;error2];%中间型指标,0为最优
[m,n]=size(data);
k=size(error1,2);
%正向化
sum=0;
for j=1:n
    M=max(abs(data(:,j)));
    for i=1:m
        data(i,j)=1-abs(data(i,j)-M)/M;
        sum=sum+data(i,j)^2;
    end
end
%标准化
data=data/sqrt(sum);

%这里每一个评价指标等价，无需引入权重矩阵
score=zeros(m,1);
for i=1:m
    sum1=0;
    sum2=0;
    for j=1:n
        z1=max(data(:,j));%最大值
        z2=min(data(:,j));%最小值      
        sum1=sum1+(data(i,j)-z1)^2;%与最大值的距离
        sum2=sum2+(data(i,j)-z2)^2;%与最小值的距离
    s1=sum1^0.5;
    s2=sum2^0.5;
    score(i)=s2/(s1+s2);
    end
end

%排序
[a,b]=sort(score);
[c,d]=sort(b);
disp(d)

k=size(error1,2);
x=linspace(1,10,k);
plot(x,error1,x,error2);