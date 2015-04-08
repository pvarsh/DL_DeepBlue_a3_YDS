function linear_baseline_model(opt)    

    model = nn.Sequential()
   
    -- if you decide to just adapt the baseline code for part 2, you'll probably want to make this linear and remove pooling
    -- model:add(nn.TemporalConvolution(1, 20, 10, 1))
    model:add(nn.Reshape(opt.minibatchSize*opt.inputDim))
    model:add(nn.Linear(opt.minibatchSize*opt.inputDim, opt.minibatchSize*opt.inputDim*2))
    model:add(nn.ReLU())
    model:add(nn.Dropout(0.5)) 
    model:add(nn.Linear(opt.minibatchSize*opt.inputDim*2, 5))
    model:add(nn.LogSoftMax())

    criterion = nn.ClassNLLCriterion()

    return model, criterion
end
