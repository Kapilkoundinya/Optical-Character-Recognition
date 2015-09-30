%% Intialization
clc;
close all;
clear all;


%% Training and Testing Set 

ChK = [1,1,0,0,0,0,1,1;
      1,1,0,0,0,1,1,0;
      1,1,0,0,1,1,0,0;
      1,1,1,1,1,0,0,0;
      1,1,1,1,1,0,0,0;
      1,1,0,0,1,1,0,0;
      1,1,0,0,0,1,1,0;
      1,1,0,0,0,0,1,1];
 
ChC = [1,1,1,1,1,1,1,1;
      1,1,1,1,1,1,1,1;
      1,1,0,0,0,0,0,0;
      1,1,0,0,0,0,0,0;
      1,1,0,0,0,0,0,0;
      1,1,0,0,0,0,0,0;
      1,1,1,1,1,1,1,1;
      1,1,1,1,1,1,1,1];
 
ChM = [1,1,0,0,0,0,1,1;
      1,1,1,0,0,1,1,1;
      1,1,1,1,1,1,1,1;
      1,1,0,1,1,0,1,1;
      1,1,0,0,0,0,1,1;
      1,1,0,0,0,0,1,1;
      1,1,0,0,0,0,1,1;
      1,1,0,0,0,0,1,1];
 
ChG = [1,1,1,1,1,1,1,1;
      1,1,0,0,0,0,0,0;
      1,1,0,0,0,0,0,0;
      1,1,1,1,1,1,1,1;
      1,1,1,1,1,1,1,1;
      1,1,0,0,0,0,1,1;
      1,1,0,0,0,0,1,1;
      1,1,1,1,1,1,1,1];
  
ChH = [1,1,0,0,0,0,1,1;
      1,1,0,0,0,0,1,1;
      1,1,0,0,0,0,1,1;
      1,1,1,1,1,1,1,1;
      1,1,1,1,1,1,1,1;
      1,1,0,0,0,0,1,1;
      1,1,0,0,0,0,1,1;
      1,1,0,0,0,0,1,1];

ChI = [1,1,1,1,1,1,1,1;
      1,1,1,1,1,1,1,1;
      0,0,0,1,1,0,0,0;
      0,0,0,1,1,0,0,0;
      0,0,0,1,1,0,0,0;
      0,0,0,1,1,0,0,0;
      1,1,1,1,1,1,1,1;
      1,1,1,1,1,1,1,1];
 
ChU = [1,1,0,0,0,0,1,1;
      1,1,0,0,0,0,1,1;
      1,1,0,0,0,0,1,1;
      1,1,0,0,0,0,1,1;
      1,1,0,0,0,0,1,1;
      1,1,0,0,0,0,1,1;
      1,1,1,1,1,1,1,1;
      1,1,1,1,1,1,1,1];
  

%Validation Set
ChX = [1,0,0,0,0,0,0,1;
      0,1,0,0,0,0,1,0;
      0,0,1,0,0,1,0,0;
      0,0,0,1,1,0,0,0;
      0,0,0,1,1,0,0,0;
      0,0,1,0,0,1,0,0;
      0,1,0,0,0,0,1,0;
      1,0,0,0,0,0,0,1];
  
ChA = [0,0,0,1,1,0,0,0;
      0,0,1,0,0,1,0,0;
      0,1,0,0,0,0,1,0;
      1,1,1,1,1,1,1,1;
      1,0,0,0,0,0,0,1;
      1,0,0,0,0,0,0,1;
      1,0,0,0,0,0,0,1;
      1,0,0,0,0,0,0,1];
  
  
%Testing Set

ChZ = [1,1,1,1,1,1,1,1;
      1,1,1,1,1,1,1,1;
      0,0,0,0,1,1,0,0;
      0,0,0,1,1,0,0,0;
      0,0,1,1,0,0,0,0;
      0,1,1,0,0,0,0,0;
      1,1,1,1,1,1,1,1;
      1,1,1,1,1,1,1,1];
  
ChY = [1,1,0,0,0,0,1,1;
      1,1,0,0,0,0,1,1;
      1,1,0,0,0,0,1,1;
      1,1,1,1,1,1,1,1;
      1,1,1,1,1,1,1,1;
      0,0,0,1,1,0,0,0;
      0,0,0,1,1,0,0,0;
      0,0,0,1,1,0,0,0]; 
  
ChO = [1,1,1,1,1,1,1,1;
      1,1,1,1,1,1,1,1;
      1,1,0,0,0,0,1,1;
      1,1,0,0,0,0,1,1;
      1,1,0,0,0,0,1,1;
      1,1,0,0,0,0,1,1;
      1,1,1,1,1,1,1,1;
      1,1,1,1,1,1,1,1];

%1 Bit Noise   
ChK1 = [1,1,0,0,0,1,1,1;
      1,1,0,0,0,1,1,0;
      1,1,0,0,1,1,0,0;
      1,1,1,1,1,0,0,0;
      1,1,1,1,1,0,0,0;
      1,1,0,0,1,1,0,0;
      1,1,0,0,0,1,1,0;
      1,1,0,0,0,0,1,1];
 
ChC1 = [1,1,1,1,1,1,1,1;
      1,1,1,1,1,1,1,1;
      1,1,0,0,0,1,0,0;
      1,1,0,0,0,0,0,0;
      1,1,0,0,0,0,0,0;
      1,1,0,0,0,0,0,0;
      1,1,1,1,1,1,1,1;
      1,1,1,1,1,1,1,1];
 
ChM1 = [1,1,0,0,0,0,1,1;
      1,1,1,0,0,1,1,1;
      1,1,1,1,1,1,1,1;
      1,1,0,1,1,0,1,1;
      1,1,0,0,0,1,1,1;
      1,1,0,0,0,0,1,1;
      1,1,0,0,0,0,1,1;
      1,1,0,0,0,0,1,1];

%2 Bit Noise 
ChK2 = [1,1,0,0,0,0,1,1;
      1,1,0,0,0,1,1,0;
      1,1,1,1,1,1,0,0;
      1,1,1,1,1,0,0,0;
      1,1,1,1,1,0,0,0;
      1,1,0,0,1,1,0,0;
      1,1,0,0,0,1,1,0;
      1,1,0,0,0,0,1,1];
 
ChC2 = [1,1,1,1,1,1,1,1;
      1,1,1,1,1,1,1,1;
      1,1,0,0,0,0,0,0;
      1,1,0,0,0,0,0,0;
      1,1,0,0,0,0,0,0;
      1,1,1,1,0,0,0,0;
      1,1,1,1,1,1,1,1;
      1,1,1,1,1,1,1,1];
 
ChM2 = [1,1,0,0,0,0,1,1;
      1,1,1,1,1,1,1,1;
      1,1,1,1,1,1,1,1;
      1,1,0,1,1,0,1,1;
      1,1,0,0,0,0,1,1;
      1,1,0,0,0,0,1,1;
      1,1,0,0,0,0,1,1;
      1,1,0,0,0,0,1,1];

%3 Bit Noise
ChK3 = [1,1,1,1,1,0,1,1;
      1,1,0,0,0,1,1,0;
      1,1,0,0,1,1,0,0;
      1,1,1,1,1,0,0,0;
      1,1,1,1,1,0,0,0;
      1,1,0,0,1,1,0,0;
      1,1,0,0,0,1,1,0;
      1,1,0,0,0,0,1,1];
 
ChC3 = [1,1,1,1,1,1,1,1;
      1,1,1,1,1,1,1,1;
      1,1,0,0,0,0,0,0;
      1,1,1,1,1,0,0,0;
      1,1,0,0,0,0,0,0;
      1,1,0,0,0,0,0,0;
      1,1,1,1,1,1,1,1;
      1,1,1,1,1,1,1,1];
 
ChM3 = [1,1,0,0,0,0,1,1;
      1,1,1,0,0,1,1,1;
      1,1,1,1,1,1,1,1;
      1,1,0,1,1,0,1,1;
      1,1,0,0,0,0,1,1;
      1,1,1,1,1,0,1,1;
      1,1,0,0,0,0,1,1;
      1,1,0,0,0,0,1,1];
  
%% Converting raw input to input vectors (Store the information as row vectors)

%Training Vectors
ChK = reshape(ChK,[1,64]); % O/P maps to {0,1}
ChM = reshape(ChM,[1,64]); % O/P maps to {1,0}
ChC = reshape(ChC,[1,64]); % O/P maps to {1,1}

% O/P maps to {0,0}
ChG = reshape(ChG,[1,64]); 
ChH = reshape(ChH,[1,64]);
ChI = reshape(ChI,[1,64]);
ChU = reshape(ChU,[1,64]);

%Training Vector Matrix
trainingMatrix = [ChK;ChM;ChC;ChG;ChH;ChI;ChU];

%Desired Output Matrix
desiredOutput = [0,1;1,0;1,1;0,0;0,0;0,0;0,0];

%Validation Vectors
ChX = reshape(ChX,[1,64]);
ChA = reshape(ChA,[1,64]);

%Validation Vector Matrix
validationMatrix = [ChX;ChA];

%Test Vectors
ChZ = reshape(ChZ,[1,64]);
ChY = reshape(ChY,[1,64]);
ChO = reshape(ChO,[1,64]);
ChK1 = reshape(ChO,[1,64]);
ChM1 = reshape(ChO,[1,64]);
ChC1 = reshape(ChO,[1,64]);
ChK2 = reshape(ChO,[1,64]);
ChM2 = reshape(ChO,[1,64]);
ChC2 = reshape(ChO,[1,64]);
ChK3 = reshape(ChO,[1,64]);
ChM3 = reshape(ChO,[1,64]);
ChC3 = reshape(ChO,[1,64]);

%Validation Vector Matrix
testMatrix = [ChZ;ChY;ChO;ChK1;ChM1;ChC1;ChK2;ChM2;ChC2;ChK3;ChM3;ChC3];

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
learningRate = 1.75;
neurons = [inputNeurons,layer1Neurons,layer2Neurons,outputNeurons];

%Network Architecture 
hiddenLayers = 2;
epochs = 100000;
loopIterations = hiddenLayers+1;
weightMatrix = cell(1,loopIterations);
%baisMatrix = cell(1,loopIterations);
calcultedOutput_Y = cell(1,loopIterations);
intermediateOutput_Z = cell(1,loopIterations);
del = cell(1,loopIterations);
delWeightMatrix = cell(1,loopIterations);
gradientWeightMatrix = cell(1,loopIterations);

errorForTrainingVector = zeros(patterns,outputNeurons);
calculatedOutput = zeros(patterns,outputNeurons);

%Initialize Weight and bias Matrices
weightMatrix{1} = rand(inputNeurons,layer1Neurons);
weightMatrix{2} = rand(layer1Neurons,layer2Neurons);
weightMatrix{3} = rand(layer2Neurons,outputNeurons);
%baisMatrix{1} = rand(inputNeurons,layer1Neurons);
%baisMatrix{2} = rand(   layer1Neurons,layer2Neurons);
%baisMatrix{3} = rand(layer2Neurons,outputNeurons);
delWeightMatrix{1} = zeros(inputNeurons,layer1Neurons);
delWeightMatrix{2} = zeros(layer1Neurons,layer2Neurons);
delWeightMatrix{3} = zeros(layer2Neurons,outputNeurons);
gradientWeightMatrix{1} = zeros(inputNeurons,layer1Neurons);
gradientWeightMatrix{2} = zeros(layer1Neurons,layer2Neurons);
gradientWeightMatrix{3} = zeros(layer2Neurons,outputNeurons);
del{1} = zeros(1,layer1Neurons);
del{2} = zeros(1,layer2Neurons);
del{3} = zeros(1,outputNeurons);



%TRAINING THE NEURAL NETWORK
for i=1:epochs
    for j=1:patterns
        
        %Error
        %error = zeros(patterns,outputNeurons);
        
        %Forward Pass for each training example
        for k=1:loopIterations
            if k==1
                intermediateOutput_Z{k} = trainingMatrix(j,:)*weightMatrix{k};
                calcultedOutput_Y{k} = sigmoid(intermediateOutput_Z{k});
            else
                intermediateOutput_Z{k} = calcultedOutput_Y{k-1}*weightMatrix{k};
                calcultedOutput_Y{k} = sigmoid(intermediateOutput_Z{k});
            end   
        end
        
        calculatedOutput(j,:) = calcultedOutput_Y{k};
    
        %Back Propagation
        %Find the output Layer error and propagate the error to the inner layers
       
        %for output layer
        del{k} = (desiredOutput(j,:) - calcultedOutput_Y{k}).*(calcultedOutput_Y{k}.*( ones(size(calcultedOutput_Y{k})) - calcultedOutput_Y{k} ));
        delWeightMatrix{k} = (calcultedOutput_Y{k-1}'*del{k});
        k = k-1; %K refers to the last hidden layer not the output layer     
        
        %for hidden layers
        while k>1  %k = totalLayers - 1   k=2
           fdash  = (calcultedOutput_Y{k}.*( ones(size(calcultedOutput_Y{k})) - calcultedOutput_Y{k} ));
           del{k} = (del{k+1}*weightMatrix{k+1}').*fdash;
           delWeightMatrix{k} = (calcultedOutput_Y{k-1}'*del{k});
           k=k-1;
        end
        
        %for input layer
        fdash  = (calcultedOutput_Y{k}.*( ones(size(calcultedOutput_Y{k})) - calcultedOutput_Y{k} ));
        del{k} = (del{k+1}*weightMatrix{k+1}').*fdash;
        delWeightMatrix{k} = (trainingMatrix(j,:)'*del{k});
        
        updatingConstant = learningRate;
        for wm=1:loopIterations
             weightMatrix{wm} = weightMatrix{wm} - updatingConstant.*delWeightMatrix{wm};
        end
        %{
        for gm=1:loopIterations
            gradientWeightMatrix{gm} = gradientWeightMatrix{gm} + delWeightMatrix{gm};
        end
        %}     
        errorForTrainingVector(j,:) = desiredOutput(j,:)- calculatedOutput(j,:);
    end
    
    totalTrainingError=sum(sum(errorForTrainingVector.^2,2))
    errorForEpochs(i) = sqrt(totalTrainingError);
        
    %{
    %update the weights for each epoch 
    %updatingConstant = learningRate/patterns;
    for wm=1:loopIterations
        weightMatrix{wm} = weightMatrix{wm} - updatingConstant.*gradientWeightMatrix{wm};
    end
    %}
    
end
%}

    
%{
%TESTING THE NEURAL NETWORK
testOutput = zeros(size(testMatrix,1),outputNeurons);
for tp = 1:size(testMatrix,1)
    for tk=1:loopIterations
        if tk==1
            intermediateOutput_Z{tk} = testMatrix(tp,:)*weightMatrix{tk};
            calcultedOutput_Y{tk} = sigmoid(intermediateOutput_Z{tk});
        else
            intermediateOutput_Z{tk} = calcultedOutput_Y{tk-1}*weightMatrix{tk};
            calcultedOutput_Y{tk} = sigmoid(intermediateOutput_Z{tk});
        end   
    end
   testout = calcultedOutput_Y{tk};
   testOutput(tp,:) = testout; 
end
-
%}