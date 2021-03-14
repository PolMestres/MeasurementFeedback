%%Measurement-based time-varying distributed optimization

clear all;

global a;
a=[0.01,0.01,0.01,0.01,0.01];
global aa;
aa=[1,2,3,4,5];
global b;
b=10;
global bb;
bb=-2;
global hx;
hx=1e-6;
global ht;
ht=1e-6;
global beta;
beta=300; %Tune it according to Lemma4
global a1;
a1=50;
global a2;
a2=1;

%state constraints
% global x1low; global x1high;
% x1low=-1500; x1high=1500;
% global x2low; global x2high;
% x2low=-2000; x2high=2000;
% global x3low; global x3high;
% x3low=-1500; x3high=1500;
% global x4low; global x4high;
% x4low=-2000; x4high=2000;
% global x5low; global x5high;
% x5low=-2000; x5high=2000;

x10=0;x20=0;x30=0; x40=0;x50=0;
x0=[x10,x20,x30,x40,x50];
tspan=[0 10];

% [t1,x1sol]=ode45(@(t,x)u1(x(1),x(2),x(3),dot(a,x)+w(t),t), tspan, x0);
% [t2,x2sol]=ode45(@(t,x)u2(x(2),x(1),x(3),dot(a,x)+w(t),t), tspan, x0);
% [t3,x3sol]=ode45(@(t,x)u3(x(3),x(2),x(4),dot(a,x)+w(t),t), tspan, x0);
% [t4,x4sol]=ode45(@(t,x)u4(x(4),x(1),x(3),dot(a,x).*x+w(t),t), tspan, x0);
% [t5,x5sol]=ode45(@(t,x)u5(x(5),x(4),dot(a,x)+w(t),t), tspan, x0);

[t,xsol]=ode45(@(t,x)u(x,dot(a,x)+w(t),dot(aa,x)+w2(t),t),tspan,x0);
hold on;
title('$\beta=20$','Interpreter','Latex','Fontsize',14);
plot(t,xsol(:,1),'Linewidth',2);
plot(t,xsol(:,2),'Linewidth',2);
plot(t,xsol(:,3),'Linewidth',2);
plot(t,xsol(:,4),'Linewidth',2);
plot(t,xsol(:,5),'Linewidth',2);
legend({'$x_1(t)$','$x_2(t)$','$x_3(t)$','$x_4(t)$','$x_5(t)$'},'Interpreter','Latex','Fontsize',16,'Location','northeastoutside');
xlabel('t','Interpreter','Latex','Fontsize',14);
hold off;

%evolution of constraints
cons=[];
for i=1:length(t)
    cons=[cons dot(a,xsol(i,:))+w(t(i))];
end
figure()
plot(t,cons)

cons2=[];
for i=1:length(t)
    cons2=[cons2 dot(aa,xsol(i,:))+w2(t(i))];
end
figure()
plot(t,cons2)

%simple projection function in R
function projection=projection(x,low,high)
if(x>low && x<high)
    projection=x;
else
    projection=0;
end
end

function w=w(t)
w=0.1*sin(t);
end

function w2=w2(t)
w2=0.1*sin(2*t);
end

function u=u(x,y,y2,t)
u=[u1(x(1),x(2),x(3),y,t),u2(x(2),x(1),x(3),t),u3(x(3),x(2),x(4),y,y2,t),u4(x(4),x(1),x(3),x(5),t),u5(x(5),x(4),y2,t)]';
end

%%%
function u1=u1(x1,x2,x3,y,t)
global beta;
global x1low;
global x1high;
phi1=-inv(hessL1(x1,y,t))*(gradL1(x1,y,t)+ddtgradL1(x1,y,t));
u1=-beta*inv(hessL1(x1,y,t))*(2*x1-x2-x3)+phi1;
%u1=projection(u1,x1low,x1high);
end

function u2=u2(x2,x1,x3,t)
global beta;
global x2low;
global x2high;
phi2=-inv(hessL2(x2,t))*(gradL2(x2,t)+ddtgradL2(x2,t));
u2=-beta*inv(hessL2(x2,t))*(2*x2-x1-x3)+phi2;
%u2=projection(u2,x2low,x2high);
end

function u3=u3(x3,x2,x4,y,y2,t)
global beta;
global x3low;
global x3high;
phi3=-inv(hessL3(x3,y,y2,t))*(gradL3(x3,y,y2,t)+ddtgradL3(x3,y,y2,t));
u3=-beta*inv(hessL3(x3,y,y2,t))*(2*x3-x2-x4)+phi3;
%u3=projection(u3,x3low,x3high);
end

function u4=u4(x4,x1,x3,x5,t)
global beta;
global x4low;
global x4high;
phi4=-inv(hessL4(x4,t))*(gradL4(x4,t)+ddtgradL4(x4,t));
u4=-beta*inv(hessL4(x4,t))*(3*x4-x1-x3-x5)+phi4;
%u4=projection(u4,x4low,x4high);
end

function u5=u5(x5,x4,y2,t)
global beta;
global x5low;
global x5high;
phi5=-inv(hessL5(x5,y2,t))*(gradL5(x5,y2,t)+ddtgradL5(x5,y2,t));
u5=-beta*inv(hessL5(x5,y2,t))*(x5-x4)+phi5;
%u5=projection(u5,x5low,x5high);
end

%%%
function hessL1=hessL1(x1,y,t)
global hx;
global a;
global b;
hessf1=(f1(x1+hx,t)-2*f1(x1,t)+f1(x1-hx,t))/hx^2;
hessL1=hessf1+(rho(t)*a(1)^2)/(1-rho(t)*(y-b))^2;
end

function hessL2=hessL2(x2,t)
global hx;
hessf2=(f2(x2+hx,t)-2*f2(x2,t)+f2(x2-hx,t))/hx^2;
hessL2=hessf2;
end

function hessL3=hessL3(x3,y,y2,t)
global hx;
global a;
global b;
global aa;
global bb;
hessf3=(f3(x3+hx,t)-2*f3(x3,t)+f3(x3-hx,t))/hx^2;
hessL3=hessf3+(rho(t)*a(3)^2)/(1-rho(t)*(y-b))^2;
hessL3=hessL3+(rho(t)*aa(3)^2)/(1-rho(t)*(y2-bb))^2;
end

function hessL4=hessL4(x4,t)
global hx;
hessf4=(f4(x4+hx,t)-2*f4(x4,t)+f4(x4-hx,t))/hx^2;
hessL4=hessf4;
end

function hessL5=hessL5(x5,y2,t)
global hx;
global aa;
global bb;
hessf5=(f5(x5+hx,t)-2*f5(x5,t)+f5(x5-hx,t))/hx^2;
hessL5=hessf5+(rho(t)*aa(3)^2)/(1-rho(t)*(y2-bb))^2;
end

%%%
function ddtgradL1=ddtgradL1(x1,y,t)
global hx, 
global ht;
global a;
global b;
global a1;
global a2;
ddtgradf1=(f1(x1+hx,t+ht)-f1(x1,t+ht)-f1(x1+hx,t)+f1(x1,t))/(hx*ht);
ddtrho=a2*a1*exp(a2*t);
ddtgradL1=ddtgradf1+(ddtrho*(y-b)*a(1))/(1-rho(t)*(y-b))^2+(rho(t)*a(1))/(1-rho(t)*(y-b))^2;
%ddtgradf1
%pause;
end

function ddtgradL2=ddtgradL2(x2,t)
global hx, 
global ht;
ddtgradf2=(f2(x2+hx,t+ht)-f2(x2,t+ht)-f2(x2+hx,t)+f2(x2,t))/(hx*ht);
ddtgradL2=ddtgradf2;
end

function ddtgradL3=ddtgradL3(x3,y,y2,t)
global hx, 
global ht;
global a;
global b;
global aa;
global bb;
global a1;
global a2;
ddtgradf3=(f3(x3+hx,t+ht)-f3(x3,t+ht)-f3(x3+hx,t)+f3(x3,t))/(hx*ht);
ddtrho=a1*exp(a2*t);
ddtgradL3=ddtgradf3+(ddtrho*(y-b)*a(3))/(1-rho(t)*(y-b))^2+(rho(t)*a(3))/(1-rho(t)*(y-b))^2;
ddtgradL3=ddtgradL3+(ddtrho*(y2-bb)*aa(3))/(1-rho(t)*(y2-bb))^2+(rho(t)*aa(3))/(1-rho(t)*(y2-bb))^2;
end

function ddtgradL4=ddtgradL4(x4,t)
global hx, 
global ht;
ddtgradf4=(f4(x4+hx,t+ht)-f4(x4,t+ht)-f4(x4+hx,t)+f4(x4,t))/(hx*ht);
ddtgradL4=ddtgradf4;
end

function ddtgradL5=ddtgradL5(x5,y2,t)
global hx, 
global ht;
global aa;
global bb;
global a2;
global a1;
ddtgradf5=(f5(x5+hx,t+ht)-f5(x5,t+ht)-f5(x5+hx,t)+f5(x5,t))/(hx*ht);
ddtrho=a1*exp(a2*t);
ddtgradL5=ddtgradf5+(ddtrho*(y2-bb)*aa(3))/(1-rho(t)*(y2-bb))^2+(rho(t)*aa(3))/(1-rho(t)*(y2-bb))^2;
end

%%%
function gradL1=gradL1(x1,y,t)
global hx;
global a;
global b;
gradf1=(f1(x1+hx,t)-f1(x1,t))/hx;
gradL1=gradf1+a(1)/(1-rho(t)*(y-b)); %has sensor
end

function gradL2=gradL2(x2,t)
global hx;
gradf2=(f2(x2+hx,t)-f2(x2,t))/hx;
%gradL2=gradf2+a(2)/(1-rho(t)*(y-b));
gradL2=gradf2;
end

function gradL3=gradL3(x3,y,y2,t)
global hx;
global a;
global b;
global aa;
global bb;
gradf3=(f3(x3+hx,t)-f3(x3,t))/hx;
gradL3=gradf3+a(3)/(1-rho(t)*(y-b))+aa(3)/(1-rho(t)*(y2-bb)); %has sensor
end

function gradL4=gradL4(x4,t)
global hx;
gradf4=(f4(x4+hx,t)-f4(x4,t))/hx;
%gradL4=gradf4+a(4)/(1-rho(t)*(y-b));
gradL4=gradf4;
end

function gradL5=gradL5(x5,y2,t)
global hx;
global aa;
global bb;
gradf5=(f5(x5+hx,t)-f5(x5,t))/hx;
%gradL5=gradf5+a(5)/(1-rho(t)*(y-b));
gradL5=gradf5+aa(5)/(1-rho(t)*(y2-bb));
end

%%%
function L1=L1(x1,y,t)
global b;
L1=f1(x1,t)-1/rho(t)*log(1-rho(t)*(y-b));
end

function L2=L2(x2,t)
L2=f2(x2,t);
end

function L3=L3(x3,y,y2,t)
global b;
global bb;
L3=f3(x3,t)-1/rho(t)*log(1-rho(t)*(y-b))-1/rho(t)*log(1-rho(t)*(y2-bb));
end

function L4=L4(x4,t)
L4=f4(x4,t);
end

function L5=L5(x5,y2,t)
global bb;
L5=f5(x5,t)-1/rho(t)*log(1-rho(t)*(y2-bb));
end

%%%
function rho=rho(t)
%Not sure how to choose a1 and a2
global a1;
global a2;
rho=a1*exp(a2*t);
end

%%%
function f1=f1(x1,t)
f1=(x1+sin(t)+5)^2;
end

function f2=f2(x2,t)
f2=(x2+2*sin(2*t)+5)^2;
end

function f3=f3(x3,t)
f3=(x3+3*sin(3*t)+5)^2;
end

function f4=f4(x4,t)
f4=(x4+4*sin(4*t)+5)^2;
end

function f5=f5(x5,t)
f5=(x5+5*sin(5*t)+5)^2;
end