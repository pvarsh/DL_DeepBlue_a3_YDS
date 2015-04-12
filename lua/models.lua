function linear_baseline(opt)    
    model = nn.Sequential()
   
    model:add(nn.Reshape(opt.minibatchSize*opt.inputDim))
    model:add(nn.Linear(opt.minibatchSize*opt.inputDim, opt.minibatchSize*opt.inputDim*2))
    model:add(nn.ReLU())
    model:add(nn.Dropout(0.5)) 
    model:add(nn.Linear(opt.minibatchSize*opt.inputDim*2, 5))
    model:add(nn.LogSoftMax())

    criterion = nn.ClassNLLCriterion()

    return model, criterion
end

function linear_two_hidden(opt)    
    model = nn.Sequential()
   
    model:add(nn.Reshape(opt.minibatchSize*opt.inputDim))

    model:add(nn.Linear(opt.minibatchSize*opt.inputDim, opt.minibatchSize*opt.inputDim*2))
    model:add(nn.ReLU())

    model:add(nn.Linear(opt.minibatchSize*opt.inputDim*2, opt.minibatchSize*opt.inputDim*2))

    model:add(nn.ReLU())

    -- model:add(nn.Dropout(0.5))
    model:add(nn.Linear(opt.minibatchSize*opt.inputDim*2, 5))
    model:add(nn.LogSoftMax())

    criterion = nn.ClassNLLCriterion()

    return model, criterion
end

function conv_baseline(opt)
    model = nn.Sequential()
    model:add(nn.TemporalConvolution(1, 20, 10, 1))
    
    if opt.pooling == 'max' then
        model:add(nn.TemporalMaxPooling(3, 1))
    elseif opt.pooling == 'logexp' then
        model:add(nn.TemporalLogExpPooling(3, 1, opt.beta))
    else
        error("opt.pooling must be 'max' or 'logexp'")
    end
    
    -- subtracting 11 may break the code if
    -- step or window size (opt.kW, opt.dW) are changed
    model:add(nn.Reshape(20*(opt.inputDim-11), true))
    model:add(nn.Linear(20*(opt.inputDim-11), 5))
    model:add(nn.LogSoftMax())

    criterion = nn.ClassNLLCriterion()

    return model, criterion
end

function conv_concat(opt)
    model = nn.Sequential()
    model:add(nn.TemporalConvolution(opt.inputDim, opt.inputDim*5, 10, 1))
    model:add(nn.TemporalMaxPooling(3, 1))
    
    -- subtracting 11 may break the code if
    -- step or window size (opt.kW, opt.dW) are changed
    model:add(nn.Reshape(opt.inputDim*5*(opt.nWordsConcat-11), true))
    model:add(nn.Linear(opt.inputDim*5*(opt.nWordsConcat-11), 5))
    model:add(nn.LogSoftMax())

    criterion = nn.ClassNLLCriterion()

    return model, criterion
end