require 'torch'
require 'nn'
require 'optim'
require 'string'
require 'xlua'
require 'cunn'

ffi = require('ffi')

-- set seed for recreating tests
torch.manualSeed(8)

-- function to read in raw document data and convert to quantized vectors
function preprocess_train_data(raw_data, dictionary, opt)
    
    -- create empty tensors that will hold wordvector concatenations
    local data = torch.zeros(opt.nTrainDocs, opt.length, opt.frame)
    local labels = torch.zeros(opt.nTrainDocs)
    
    for j=1,opt.nTrainDocs do

        local index = raw_data.index[opt.idx][j]
        -- standardize to all lowercase
        local document = ffi.string(torch.data(raw_data.content:narrow(1, index, 1))):lower()

        -- will either scan the entire document or only go as far as length permits
        for c = 1,math.min(document:len(),opt.length) do
            if dictionary[document:sub(c,c)] then
                data[{ {j},{c},{dictionary[document:sub(c,c)]} }] = 1
            end
        end

        labels[j] = raw_data.labels[opt.idx][j]
    end

    return data, labels
end

function preprocess_test_data(raw_data, dictionary, opt)
    
    -- create empty tensors that will hold wordvector concatenations
    local data = torch.zeros(5*opt.nTestDocs, opt.length, opt.frame)
    local labels = torch.zeros(5*opt.nTestDocs)
    
    local counter = 1
    for i = 1,5 do
        for j=opt.nTrainDocs+1,opt.nTrainDocs+opt.nTestDocs do

            local index = raw_data.index[i][j]
            -- standardize to all lowercase
            local document = ffi.string(torch.data(raw_data.content:narrow(1, index, 1))):lower()

            -- will either scan the entire document or only go as far as length permits
            for c = 1,math.min(document:len(),opt.length) do
                if dictionary[document:sub(c,c)] then
                    data[{ {j},{c},{dictionary[document:sub(c,c)]} }] = 1
                end
            end

            labels[counter] = raw_data.labels[i][j]
            counter = counter + 1
        end
    end

    return data, labels
end

function train_model(model, criterion, training_data, training_labels, opt)

	-- classes
	classes = {'1','2','3','4','5'}

	-- This matrix records the current confusion across classes
	confusion = optim.ConfusionMatrix(classes)

    parameters,gradParameters = model:getParameters()

    -- configure optimizer
    optimState = {
    	learningRate = opt.learningRate,
    	weightDecay = opt.weightDecay,
    	momentum = opt.momentum,
    	learningRateDecay = opt.learningRateDecay
    }
    optimMethod = optim.sgd

    epoch = epoch or 1
	local time = sys.clock()

	model:training()

	inputs = torch.zeros(opt.batchSize,opt.length,opt.frame):cuda()
	targets = torch.zeros(opt.batchSize):cuda()

	-- do one epoch
	print("\n==> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
	for t = 1,training_data:size(1),opt.batchSize do
		-- disp progress
		-- xlua.progress(t, training_data:size(1))
		inputs:zero()
		targets:zero()

		-- create mini batch
		if t + opt.batchSize-1 <= training_data:size(1) then
			-- xx = opt.batchSize
--			print("in if stmt")
			inputs[{}] = training_data[{ {t,t+opt.batchSize-1},{},{} }]
			targets[{}] = training_labels[{ {t,t+opt.batchSize-1} }]
		
			-- create closure to evaluate f(X) and df/dX
		local feval = function(x)
                       -- get new parameters
                       if x ~= parameters then
                          parameters:copy(x)
                       end

                       -- reset gradients
                       gradParameters:zero()

                       -- f is the average of all criterions
                       local f = 0

                       -- evaluate function for complete mini batch
                       for i = 1,inputs:size(1) do
                          -- estimate f
                          local output = model:forward(inputs[{ {i} }])
--                          print("Output size")
--                          print(output:size())
                          local err = criterion:forward(output, targets[{ {i} }])
                          f = f + err

                          -- estimate df/dW
                          local df_do = criterion:backward(output, targets[{ {i} }])
                          model:backward(inputs[{ {i} }], df_do)
--                          print("Target")
--                          print(targets[{ {i} }])

                          -- update confusion
--                          for k=1,opt.batchSize do
--                    			confusion:add(model.output[k], targets[{ {k} }])
--                		  end

--                          confusion:add(output, targets[{ {i} }])
                       end

                       -- normalize gradients and f(X)
--                       gradParameters:div(inputs:len())
--                       f = f/#inputs

                       -- return f and df/dX
                       return f,gradParameters
                    end

			-- optimize on current mini-batch
			optimMethod(feval, parameters, optimState)
		end
	end

	-- time taken
	time = sys.clock() - time
	time = time / training_data:size(1)
	print("==> time to learn 1 sample = " .. (time*1000) .. 'ms')

	-- print confusion matrix
	-- print(confusion)
	confusion:updateValids()

	-- print accuracy
	print("==> training accuracy for epoch " .. epoch .. ':')
	accuracy = confusion.totalValid*100
    -- log accuracy for this epoch
    print(accuracy)

end

function test_model(model, data, labels, opt)

    model:evaluate()

    t_input = torch.zeros(opt.length, opt.frame):cuda()
    t_labels = torch.zeros(1):cuda()

    for t = 1,data:size(1) do
        t_input:zero()
        t_labels:zero()
        t_input[{}] = data[t]
        t_labels[{}] = labels[t]
        local pred = model:forward(t_input)
        confusion:add(pred, t_labels[1])
    end
    -- print(confusion)
    confusion:updateValids()

    -- print accuracy
    print("==> test accuracy for epoch " .. epoch .. ':')
    -- print(confusion)
    accuracy = confusion.totalValid*100
    print(accuracy)

    -- save/log current net
    if accuracy > accs['max'] then 
        local filename = paths.concat(opt.save, 'model_c.net')
        os.execute('mkdir -p ' .. sys.dirname(filename))
        print('==> saving model to '..filename)
        torch.save(filename, model)
    end

    -- if accuracy <= accs['max'] then
    --     opt.learningRate = opt.learningRate/10
    -- end

    accs['max'] = math.max(accuracy,accs['max'])
    accs[epoch] = accuracy

    confusion:zero()
end


function main()

    -- define the character dictionary
    local alphabet =  "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
    local dictionary = {}
    for i = 1,#alphabet do
        dictionary[alphabet:sub(i,i)] = i
    end
    -- Configuration parameters
    opt = {}
    -- table acting as a log of accuracies per epoch
    accs = {}
    accs['max'] = 0
    -- number of frames in the character vectors
    opt.frame = alphabet:len()
    opt.dataPath = "/scratch/courses/DSGA1008/A3/data/train.t7b"
    -- path to save model to
    opt.save = "results"
    -- maximum number of words per text document
    opt.length = 1014
    -- training/test sizes per class
    opt.nTrainDocs = 97500
    opt.nTestDocs = 32500

    -- training parameters
    opt.init_weight = 0.1 -- random weight initialization
    opt.nEpochs = 50
    opt.batchSize = 64
    opt.learningRate = 0.01
    opt.learningRateDecay = 1e-5
    opt.momentum = 0.9
    opt.weightDecay = 0
    
    print("Loading raw data...")
    local raw_data = torch.load(opt.dataPath)

    -- shuffle all the data in a way in which we can call on it five times for conserving memory
    print("Shuffling data...")
    s_raw_data = {}
    s_raw_data['index'] = torch.reshape(raw_data.index,1,5*130000)
    s_raw_data['labels'] = torch.zeros(650000)
    for i=1,5 do
        s_raw_data['labels'][{ {130000*(i-1)+1,i*130000} }] = i
    end
    order = torch.randperm(650000)

    index_table = {}
    for i=1,5 do
        index_table[i] = order[{ {130000*(i-1)+1,i*130000} }]
    end

    labels = torch.zeros(5,130000)
    index_n = torch.zeros(5,130000)
    for j=1,5 do
        for i=1,130000 do
            index_n[{ {j},{i} }] = s_raw_data.index[{ {},{index_table[j][i]} }]
            labels[{ {j},{i} }] = s_raw_data.labels[index_table[j][i]]
        end
    end

    n_raw_data = {}
    n_raw_data['index'] = index_n:clone()
    n_raw_data['labels'] = labels:clone()
    n_raw_data['content'] = raw_data.content:clone()
    -----------------------------------------------------------------------------------

    -- build model *****************************************************************************
    model = nn.Sequential()
    print("after seq")
    -- first layer (#inputDim x 204)
    model:add(nn.TemporalConvolution(opt.frame, 512, 7, 1))
 	model:add(nn.Threshold())
    model:add(nn.TemporalMaxPooling(2,2))

    -- second layer (147x512) 
--    model:add(nn.TemporalConvolution(512, 512, 7))
--    model:add(nn.Threshold())
--    model:add(nn.TemporalMaxPooling(3,3))
--
--    -- 1st fully connected layer (19x512)
    model:add(nn.Reshape(504*512))
    model:add(nn.Linear(504*512,1024))
    model:add(nn.Threshold())
    model:add(nn.Dropout(0.7))

    -- final layer for classification 1024
    model:add(nn.Linear(1024,5))
    model:add(nn.LogSoftMax())

	criterion = nn.ClassNLLCriterion()

	-- CUDA
	model:cuda()
	criterion:cuda()

    -- randomly initialize weights  
    -- model:getParameters():uniform(-opt.init_weight, opt.init_weight)
	print("\nTraining model...")
    for i=1,opt.nEpochs do
        epoch = i
        for j=1,5 do
            opt.idx = j
            print("Processing data batch " .. j .. " for epoch " .. i)
            local training_data, training_labels = preprocess_train_data(n_raw_data, dictionary, opt)
    		train_model(model, criterion, training_data, training_labels, opt)
        end
--        confusion:zero()
        local test_data, test_labels = preprocess_test_data(n_raw_data, dictionary, opt)
        test_model(model,test_data,test_labels,opt)
	end

end

main()