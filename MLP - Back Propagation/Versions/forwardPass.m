function networkOutput{k} = forwardPass( inputData,weightMatrix{:},hiddenLayers, neurons)
    
    networkOutput = cell(1,loopIterations);
    intermediateOutput_Z = cell(1,loopIterations);
    for k=1:hiddenLayers+1
        if k==1
            intermediateOutput_Z{k} = inputData*weightMatrix{k};
            networkOutput{k} = sigmoid(intermediateOutput_Z{k});
        else
            intermediateOutput_Z{k} = networkOutput{k-1}*weightMatrix{k};
            networkOutput{k} = sigmoid(intermediateOutput_Z{k});
        end   
    end

end

