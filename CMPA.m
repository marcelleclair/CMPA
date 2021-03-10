set(0, 'DefaultFigureWindowStyle', 'docked');
close all

Is = 0.01e-12; % FB saturation current (A)
Ib = 0.1e-12; % Breakdown saturation current (A)
Vb = 1.3; % Breakdown voltage (V)
Gp = 0.1; % Parasitic parallel conductance (S)

Id = @(v) (Is.*(exp(1.2 .* v ./ 0.025) - 1)) + (Gp .* v) - (Ib.*(exp(-(1.2/0.025) .* (v + Vb)) - 1));

V = transpose(linspace(-1.92, 0.7, 200));
I = Id(V);
disturb =  1 + 0.2.*(rand(200,1) - 0.5);
I_dist = I .* disturb;

% Polynomial Fits
P4 = polyfit(V,I,4);
P4D = polyfit(V,I_dist,4);
P8 = polyfit(V,I,8);
P8D = polyfit(V,I_dist,8);

% Nonlinear Fit 1
fo1 = fittype('A.*(exp(1.2*x/25e-3)-1) + 0.1.*x - C*(exp(1.2*(-(x+1.3))/25e-3)-1)');
ff1 = fit(V,I,fo1);
If1 = ff1(V);
ffd1 = fit(V,I_dist,fo1);
Ifd1 = ffd1(V);

% Nonlinear Fit 2
fo2 = fittype('A.*(exp(1.2*x/25e-3)-1) + B.*x - C*(exp(1.2*(-(x+1.3))/25e-3)-1)');
ff2 = fit(V,I,fo2);
If2 = ff2(V);
ffd2 = fit(V,I_dist,fo2);
Ifd2 = ffd2(V);

% Nonlinear Fit 3
fo3 = fittype('A.*(exp(1.2*x/25e-3)-1) + B.*x - C*(exp(1.2*(-(x+D))/25e-3)-1)');
ff3 = fit(V,I,fo3);
If3 = ff3(V);
ffd3 = fit(V,I_dist,fo3);
Ifd3 = ffd3(V);

% Neural-Network Fitting
inputs = V.';
targets = I_dist.';
hiddenLayerSize = 10;
net = fitnet(hiddenLayerSize);
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;
[net,tr] = train(net,inputs,targets);
outputs = net(inputs);
errors = gsubtract(outputs,targets);
performance = perform(net,targets,outputs);
view(net);
Inn = outputs;

% Plot Results

figure('Name','Polynomial Fit (Lin Scale)');
plot(V,I_dist);
hold on
plot(V,polyval(P4D,V),'--');
plot(V,polyval(P8D,V),'-.');
xlabel('V_D (V)');
ylabel('I_D (A)');
legend('Disturbed Data','4th Order Polynomial Fit','8th Order Polynomial Fit');
grid on

figure('Name','Nonlinear Fit (Lin Scale)');
plot(V,I_dist);
hold on
plot(V,Ifd1,'--');
plot(V,Ifd2,'-.');
plot(V,Ifd3,':');
xlabel('V_D (V)');
ylabel('I_D (A)');
legend('Disturbed Data','Fit 2','Fit 3','Fit 4');
grid on

figure('Name','Neural Network Fit (Lin Scale)');
plot(V,I_dist);
hold on
plot(V,Inn,'--');
xlabel('V_D (V)');
ylabel('I_D (A)');
legend('Disturbed Data','NN Fit');
grid on

figure('Name','Polynomial Fit (Log Scale)');
semilogy(V,abs(I_dist));
hold on
semilogy(V,abs(polyval(P4D,V)),'--');
semilogy(V,abs(polyval(P8D,V)),'-.');
xlabel('V_D (V)');
ylabel('I_D (A)');
legend('Disturbed Data','4th Order Polynomial Fit','8th Order Polynomial Fit');
grid on

figure('Name','Nonlinear Fit (Log Scale)');
semilogy(V,abs(I_dist));
hold on
semilogy(V,abs(Ifd1),'--');
semilogy(V,abs(Ifd2),'-.');
semilogy(V,abs(Ifd3),':');
xlabel('V_D (V)');
ylabel('I_D (A)');
legend('Disturbed Data','Fit 2','Fit 3','Fit 4');
grid on

figure('Name','Neural Network Fit (Log Scale)');
semilogy(V,abs(I_dist));
hold on
semilogy(V,abs(Inn),'--');
xlabel('V_D (V)');
ylabel('I_D (A)');
legend('Disturbed Data','NN Fit');
grid on