-------------------------------------------------
-- A linear neural net baseline without convolutions
-- or pooling layers.
-------------------------------------------------

require 'torch'
require 'nn'
require 'optim'
require 'models'
require 'pooling'

ffi = require('ffi')

--- Parses and loads the GloVe word vectors into a hash table:
-- glove_table['word'] = vector
function load_glove(path, inputDim)
    
    local glove_file = io.open(path)
    local glove_table = {}

    local line = glove_file:read("*l")
    while line do
        -- read the GloVe text file one line at a time, break at EOF
        local i = 1
        local word = ""
        for entry in line:gmatch("%S+") do -- split the line at each space
            if i == 1 then
                -- word comes first in each line, so grab it and create new table entry
                word = entry:gsub("%p+", ""):lower() -- remove all punctuation and change to lower case
                if string.len(word) > 0 then
                    glove_table[word] = torch.zeros(inputDim, 1) -- padded with an extra dimension for convolution
                else
                    break
                end
            else
                -- read off and store each word vector element
                glove_table[word][i-1] = tonumber(entry)
            end
            i = i+1
        end
        line = glove_file:read("*l")
    end
    
    return glove_table
end

function load_idf(path)
    print("Loading idf table...")
    local idf_file = io.open(path)
    local idf_table = {}

    local line = idf_file:read("*l")
    local break_count = 0
    while line do
        local i = 1
        local word = ""
        for entry in line:gmatch("[^,]+") do
            if i == 1 then 
                word = entry
            end
            if i == 2 then
                idf_table[word] = entry
            end
            i = i + 1
        end -- end: for

        break_count = break_count + 1
        line = idf_file:read("*l")
    end -- end: do
    return idf_table
end


--- Here we simply encode each document as a fixed-length vector 
-- by computing the unweighted average of its word vectors.
-- A slightly better approach would be to weight each word by its tf-idf value
-- before computing the bag-of-words average; this limits the effects of words like "the".
-- Still better would be to concatenate the word vectors into a variable-length
-- 2D tensor and train a more powerful convolutional or recurrent model on this directly.
function preprocess_data(raw_data, wordvector_table, opt)
    
    local data = torch.zeros(opt.nClasses*(opt.nTrainDocs+opt.nTestDocs), opt.inputDim, 1)
    local labels = torch.zeros(opt.nClasses*(opt.nTrainDocs + opt.nTestDocs))
    
    -- use torch.randperm to shuffle the data, since it's ordered by class in the file
    if opt.seed ~= 0 then
        torch.manualSeed(opt.seed)
    end

    local order = torch.randperm(opt.nClasses*(opt.nTrainDocs+opt.nTestDocs))
    
    -- Read first nTrainDocs + nTestDocs reviews from each of 5 classes
    -- Assign the computed vector representation to k'th element of data tensor
    for i=1,opt.nClasses do
        for j=1,opt.nTrainDocs+opt.nTestDocs do
            local k = order[(i-1)*(opt.nTrainDocs+opt.nTestDocs) + j]
            
            local doc_size = 1
            
            local index = raw_data.index[i][j]
            local document = ffi.string(torch.data(raw_data.content:narrow(1, index, 1))):lower()
            
            if opt.wordWeght == 'tfidf' then
                local tf_table = {}

                -- compute term frequency (tf)
                -- print("Computing term frequency...")
                for word in document:gmatch("%S+") do
                    if tf_table[word] then
                        tf_table[word] = tf_table[word] + 1
                    else
                        tf_table[word] = 1
                    end
                end

                print("Computing tf-idf weighted vector representation...")
                for word in document:gmatch("%S+") do
                    if wordvector_table[word:gsub("%p+", "")] then
                        doc_size = doc_size + 1
                        local tf_idf = tf_table[word]*opt.idf_table[word]

                        -- tf_idf is normalized by idf average: 12.175
                        data[k]:add(wordvector_table[word:gsub("%p+", "")]*tf_idf/12.175)

                    end
                end
            else
                -- print("Computing unweighted vector BOW representation...")
                -- break each review into words and compute the document average
                for word in document:gmatch("%S+") do
                    if wordvector_table[word:gsub("%p+", "")] then
                        doc_size = doc_size + 1
                        data[k]:add(wordvector_table[word:gsub("%p+", "")])                    
                    end
                end
            end
            data[k]:div(doc_size)
            labels[k] = i
        end
    end

    if opt.normalize == 1 then
        print(">> Normalizing observations...")
        local neg_feature_mean = data:mean(1):mul(-1)
        local feature_std = data:std(1)
        -- print("normalizing tensors size: ")
        -- print(neg_feature_mean:size())
        -- print(neg_feature_mean)
        -- print(feature_std)
        for i=1,opt.inputDim do
            data[{ {}, i, {} }]:add(neg_feature_mean[{1,i,1}])
            data[{ {}, i, {} }]:div(feature_std[{1,i,1}])
        end
    end

    return data, labels
end

function train_model(model, criterion, data, labels, test_data, test_labels, opt)

    parameters, grad_parameters = model:getParameters()
    
    -- optimization functional to train the model with torch's optim library
    local function feval(x) 
        local minibatch = data:sub(opt.idx, opt.idx + opt.minibatchSize - 1, 1, data:size(2)):clone()
        local minibatch_labels = labels:sub(opt.idx, opt.idx + opt.minibatchSize - 1):clone()
        
        model:training()
        -- print("minibatch tensor shape", minibatch[1]:size())
        -- TODO: Minibatches broken for batch size > 1
        local minibatch_loss = criterion:forward(model:forward(minibatch), minibatch_labels)
        model:zeroGradParameters()
        model:backward(minibatch, criterion:backward(model.output, minibatch_labels))
        
        return minibatch_loss, grad_parameters
    end
    
    for epoch=1,opt.nEpochs do
        local order = torch.randperm(opt.nBatches) -- not really good randomization
        for batch=1,opt.nBatches do
            opt.idx = (order[batch] - 1) * opt.minibatchSize + 1
            optim.sgd(feval, parameters, opt)
        end

        local accuracy = test_model(model, test_data, test_labels, opt)
        print("epoch ", epoch, " error: ", accuracy)
        print("Saving model to " .. opt.modelFileName .. "WARNING: This overwrites the file")
        torch.save(opt.modelFileName, model)

    end
end

function test_model(model, data, labels, opt)
    
    model:evaluate()

    local pred = model:forward(data)
    local _, argmax = pred:max(2)
    local err = torch.ne(argmax:double(), labels:double()):sum() / labels:size(1)

    --local debugger = require('fb.debugger')
    --debugger.enter()

    return err
end

function main(opt)

    opt.idx = 1

    if opt.wordWeight == 'tfidf' then
        print("Loading inverse document frequency table...")
        opt.idf_table = load_idf(opt.idfPath)
    end

    print("Loading word vectors...")
    local glove_table = load_glove(opt.glovePath, opt.inputDim)
    
    print("Loading raw data...")
    local raw_data = torch.load(opt.dataPath)


    
    print("Computing document input representations...")
    local processed_data, labels = preprocess_data(raw_data, glove_table, opt)
    
    -- split data into makeshift training and validation sets
    local training_data = processed_data:sub(1, 
                                             opt.nClasses*opt.nTrainDocs,
                                             1, 
                                             processed_data:size(2)):clone()
    local training_labels = labels:sub(1, opt.nClasses*opt.nTrainDocs):clone()
    
    -- make your own choices - here I have not created a separate test set
    local test_data = processed_data:sub(opt.nClasses*opt.nTrainDocs + 1,
                                         opt.nClasses*opt.nTrainDocs + opt.nClasses*opt.nTestDocs,
                                         1,
                                         processed_data:size(2)):clone()
    local test_labels = labels:sub(opt.nClasses*opt.nTrainDocs + 1,
                                         opt.nClasses*opt.nTrainDocs + opt.nClasses*opt.nTestDocs):clone()


    -- Build model
    if opt.model == 'linear_baseline' then
        opt.minibatchSize = 1
        print("WARNING: Resetting minibatchSize = 1. linear_baseline model breaks for larger minibatches.")
        model, criterion = linear_baseline(opt)
    elseif opt.model == 'linear_two_hidden' then
        opt.minibatchSize = 1
        print("WARNING: Resetting minibatchSize = 1. linear_baseline model breaks for larger minibatches.")
        model, criterion = linear_baseline(opt)
    elseif opt.model == 'conv_baseline' then
        model, criterion = conv_baseline(opt)
    end

    print(model) 
    print("Training...")
    train_model(model, criterion, training_data, training_labels, test_data, test_labels, opt)
    local results = test_model(model, test_data, test_labels)
    print(results)
end


--------------------------------------------------------------------------------------
-- Command line options
--------------------------------------------------------------------------------------
if not opt then
   print '==> processing options'
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text()
   cmd:text('Options:')
   cmd:option('-pooling', 'max', '[max | logexp] pooling')
   cmd:option('-beta', 20, 'LogExp pooling beta parameter')
   cmd:option('-inputDim', 50, 'word vector dimension: [50 | 100 | 200 | 300]')
   cmd:option('-glovePath', '/scratch/courses/DSGA1008/A3/glove/', 'path to GloVe files')
   cmd:option('-dataPath', '/scratch/courses/DSGA1008/A3/data/train.t7b', 'path to data')
   cmd:option('-idfPath', '../idf/idf.csv', 'path to idf.csv file')
   cmd:option('-nTrainDocs', 10000, 'number of training documents in each class')
   cmd:option('-nTestDocs', 1000, 'number of test documents in each class')
   cmd:option('-nClasses', 5, 'number of classes')
   cmd:option('-nEpochs', 50, 'number of training epochs')
   cmd:option('-minibatchSize', 128, 'minibatch size')
   cmd:option('-learningRate', 0.1, 'learning rate')
   cmd:option('-learningRateDecay', 0.001, 'learning rate decay')
   cmd:option('-momentum', 0.1, 'SGD momentum')
   cmd:option('-model', 'linear_baseline', 'model function to be used [linear_baseline | linear_two_hidden | conv_baseline | conv_concat]')
   cmd:option('-seed', 0, 'manual seed for initial data permutation')
   cmd:option('-modelFileName' , 'model.net', 'filename to save model')
   cmd:option('-wordWeight', 'none', 'word vector weights ["none" | "tfidf"]')
   cmd:option('-normalize', 0, 'normalize bag of words [true | false]')
   cmd:text()
   opt = cmd:parse(arg or {})
   opt.glovePath = opt.glovePath .. 'glove.6B.' .. opt.inputDim .. 'd.txt'
   opt.nBatches = math.floor(opt.nTrainDocs / opt.minibatchSize)
end
print(opt)
main(opt)
