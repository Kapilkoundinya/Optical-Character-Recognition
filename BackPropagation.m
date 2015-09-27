%% Intialization
clc;
close all;
clear all;


%% Training and Testing Set 

 K = [1,1,0,0,0,0,1,1;
      1,1,0,0,0,1,1,0;
      1,1,0,0,1,1,0,0;
      1,1,1,1,1,0,0,0;
      1,1,1,1,1,0,0,0;
      1,1,0,0,1,1,0,0;
      1,1,0,0,0,1,1,0;
      1,1,0,0,0,0,1,1];
 
 C = [1,1,1,1,1,1,1,1;
      1,1,1,1,1,1,1,1;
      1,1,0,0,0,0,0,0;
      1,1,0,0,0,0,0,0;
      1,1,0,0,0,0,0,0;
      1,1,0,0,0,0,0,0;
      1,1,1,1,1,1,1,1;
      1,1,1,1,1,1,1,1];
 
 M = [1,1,0,0,0,0,1,1;
      1,1,1,0,0,1,1,1;
      1,1,1,1,1,1,1,1;
      1,1,0,1,1,0,1,1;
      1,1,0,0,0,0,1,1;
      1,1,0,0,0,0,1,1;
      1,1,0,0,0,0,1,1;
      1,1,0,0,0,0,1,1];
 
 G = [1,1,1,1,1,1,1,1;
      1,1,0,0,0,0,0,0;
      1,1,0,0,0,0,0,0;
      1,1,1,1,1,1,1,1;
      1,1,1,1,1,1,1,1;
      1,1,0,0,0,0,1,1;
      1,1,0,0,0,0,1,1;
      1,1,1,1,1,1,1,1];
  
 H = [1,1,0,0,0,0,1,1;
      1,1,0,0,0,0,1,1;
      1,1,0,0,0,0,1,1;
      1,1,1,1,1,1,1,1;
      1,1,1,1,1,1,1,1;
      1,1,0,0,0,0,1,1;
      1,1,0,0,0,0,1,1;
      1,1,0,0,0,0,1,1];

 I = [1,1,1,1,1,1,1,1;
      1,1,1,1,1,1,1,1;
      0,0,0,1,1,0,0,0;
      0,0,0,1,1,0,0,0;
      0,0,0,1,1,0,0,0;
      0,0,0,1,1,0,0,0;
      1,1,1,1,1,1,1,1;
      1,1,1,1,1,1,1,1];
 
 U = [1,1,0,0,0,0,1,1;
      1,1,0,0,0,0,1,1;
      1,1,0,0,0,0,1,1;
      1,1,0,0,0,0,1,1;
      1,1,0,0,0,0,1,1;
      1,1,0,0,0,0,1,1;
      1,1,1,1,1,1,1,1;
      1,1,1,1,1,1,1,1];
  

%Validation Set
 X = [1,0,0,0,0,0,0,1;
      0,1,0,0,0,0,1,0;
      0,0,1,0,0,1,0,0;
      0,0,0,1,1,0,0,0;
      0,0,0,1,1,0,0,0;
      0,0,1,0,0,1,0,0;
      0,1,0,0,0,0,1,0;
      1,0,0,0,0,0,0,1];
  
 A = [0,0,0,1,1,0,0,0;
      0,0,1,0,0,1,0,0;
      0,1,0,0,0,0,1,0;
      1,1,1,1,1,1,1,1;
      1,0,0,0,0,0,0,1;
      1,0,0,0,0,0,0,1;
      1,0,0,0,0,0,0,1;
      1,0,0,0,0,0,0,1];
  
  
%Testing Set

 Z = [1,1,1,1,1,1,1,1;
      1,1,1,1,1,1,1,1;
      0,0,0,0,1,1,0,0;
      0,0,0,1,1,0,0,0;
      0,0,1,1,0,0,0,0;
      0,1,1,0,0,0,0,0;
      1,1,1,1,1,1,1,1;
      1,1,1,1,1,1,1,1];
  
 Y = [1,1,0,0,0,0,1,1;
      1,1,0,0,0,0,1,1;
      1,1,0,0,0,0,1,1;
      1,1,1,1,1,1,1,1;
      1,1,1,1,1,1,1,1;
      0,0,0,1,1,0,0,0;
      0,0,0,1,1,0,0,0;
      0,0,0,1,1,0,0,0]; 
  
 O = [1,1,1,1,1,1,1,1;
      1,1,1,1,1,1,1,1;
      1,1,0,0,0,0,1,1;
      1,1,0,0,0,0,1,1;
      1,1,0,0,0,0,1,1;
      1,1,0,0,0,0,1,1;
      1,1,1,1,1,1,1,1;
      1,1,1,1,1,1,1,1];

  
  
%% Converting raw input to input vectors (Store the information as row vectors)

%Training Vectors
K = reshape(K,[1,64]); % O/P maps to {0,1}
M = reshape(M,[1,64]); % O/P maps to {1,0}
C = reshape(C,[1,64]); % O/P maps to {1,1}

% O/P maps to {0,0}
G = reshape(G,[1,64]); 
H = reshape(H,[1,64]);
I = reshape(I,[1,64]);
U = reshape(U,[1,64]);

%Training Vector Matrix
trainingMatrix = [K;M;C;G;H;I;U];

%Desired Output Matrix
desiredOutput = [0,1;1,0;1,1;0,0;0,0;0,0;0,0];

%Validation Vectors
X = reshape(X,[1,64]);
A = reshape(A,[1,64]);

%Validation Vector Matrix
validationMatrix = [X;A];

%Test Vectors
Z = reshape(Z,[1,64]);
Y = reshape(Y,[1,64]);
O = reshape(O,[1,64]);

%Validation Vector Matrix
testMatrix = [Z;Y;O];

%check whether each inputvector has an associated output
if size(trainingMatrix,1) ~= size(desiredOutput)
    disp('ERROR: Data Mismatch');
    return
end

patterns = size(trainingMatrix,1);

%% Back Propagation Implementation
%No.of Hidden Layers = 2
%No.of neurons in the output layer = 2
%No.of Input neurons = 64
%Learning Rate = 1.25
%Epochs = 10,000
%Initial Weights = [-0.5 0.5]

%Neurons in the network
inputNeurons = size(trainingMatrix,2);
layer1Neurons = inputNeurons/2;
layer2Neurons = layer1Neurons/2;
outputNeurons = size(desiredOutput,2);


%Network Architecture 
hiddenLayers = 2;
epochs = 1000;
loopIterations = hiddenLayers+1;
weightMatrix = cell(1,loopIterations);
baisMatrix = cell(1,loopIterations);
calcultedOutput_Y = cell(1,loopIterations);
intermediateOutput_Z = cell(1,loopIterations);


%Initialize Weight and bias Matrices
weightMatrix{1} = rand(inputNeurons,layer1Neurons);
weightMatrix{2} = rand(layer1Neurons,layer2Neurons);
weightMatrix{3} = rand(layer2Neurons,outputNeurons);
baisMatrix{1} = rand(inputNeurons,layer1Neurons);
baisMatrix{2} = rand(layer1Neurons,layer2Neurons);
baisMatrix{3} = rand(layer2Neurons,outputNeurons);

for i=1:epochs
while loopIterations<1
    

loopIterations = loopIterations-1;
end

%update the weights

%find the error
end

