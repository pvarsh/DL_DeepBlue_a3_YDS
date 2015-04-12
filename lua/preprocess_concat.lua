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