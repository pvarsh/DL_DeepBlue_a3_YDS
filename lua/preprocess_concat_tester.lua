require('nn')
ffi = require('ffi')


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


function preprocess_data_concat(raw_data, wordvector_table, opt)

    local data = torch.zeros(
                    opt.nClasses*(opt.nTrainDocs+opt.nTestDocs), -- # samples
                    opt.inputDim, -- dimension of vector representation
                    opt.nWordsConcat, -- number of words to concatenate
                    1 -- ? dimension to be used for batching ? 
                    )
    local labels = torch.zeros(opt.nClasses*(opt.nTrainDocs + opt.nTestDocs))
    
    -- use torch.randperm to shuffle the data, since it's ordered by class in the file
    -- setting seed if given in options
    if opt.seed ~= 0 then
        torch.manualSeed(opt.seed)
    end
    -- local order = torch.randperm(opt.nClasses*(opt.nTrainDocs+opt.nTestDocs))
    local order = torch.range(1, opt.nClasses*(opt.nTrainDocs+opt.nTestDocs))
    
    for i=1,opt.nClasses do
        for j=1,opt.nTrainDocs+opt.nTestDocs do
            local k = order[(i-1)*(opt.nTrainDocs+opt.nTestDocs) + j]
            
            -- local doc_size = 1
            
            local index = raw_data.index[i][j]
            -- standardize to all lowercase
            local document = ffi.string(torch.data(raw_data.content:narrow(1, index, 1))):lower()
            
            -- break each review into words and compute the document average
            local word_count = 1
            for word in document:gmatch("%S+") do
                if wordvector_table[word:gsub("%p+", "")] then
                    -- doc_size = doc_size + 1
                    data[{ {k},{},{word_count},{1} }]:add(wordvector_table[word:gsub("%p+", "")])
                    word_count = word_count + 1
                    if word_count >= opt.nWordsConcat then
                        break
                    end
                end
            end

            -- data[k]:div(doc_size)
            labels[k] = i
        end
    end

    return data, labels
end

local opt = {}
opt.glovePath = "/Users/petervarshavsky/Dropbox/NYU/deeplearning/DL_DeepBlue_a3_YDS/glove/"
opt.dataPath = "/Users/petervarshavsky/Dropbox/NYU/deeplearning/DL_DeepBlue_a3_YDS/data/train.t7b"
opt.inputDim = 50
opt.glovePath = opt.glovePath .. 'glove.6B.' .. opt.inputDim .. 'd.txt'
opt.nClasses = 5
opt.nTrainDocs = 10
opt.nTestDocs = 5
opt.nWordsConcat = 100
opt.seed = 1


print("Loading word vectors...")
local glove_table = load_glove(opt.glovePath, opt.inputDim)

print("Loading raw data...")
local raw_data = torch.load(opt.dataPath)

print("Concatenating words...")
processed_data, labels = preprocess_data_concat(raw_data, glove_table, opt)
print(processed)