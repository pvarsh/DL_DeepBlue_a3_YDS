function load_glove()
    
    path = '/scratch/courses/DSGA1008/A3/glove/glove.6B.300d.txt'
    inputDim = 300

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



function preprocess_data(raw_data, wordvector_table)
    
    inputDim = 300
    local data = torch.zeros(1, inputDim, 1)
    document = raw_data:lower()

    local doc_size = 1
    
    -- break each review into words and compute the document average
    for word in document:gmatch("%S+") do
        if wordvector_table[word:gsub("%p+", "")] then
            doc_size = doc_size + 1
            data:add(wordvector_table[word:gsub("%p+", "")])
        end
    end

    data:div(doc_size)
    return data
end



function main()
    -- Prepare for input
    local sentence
    local N
    N = io.read()

    -- Load glove vector and model
    glove_table = load_glove()
    model = torch.load("/scratch/pv629/baseline_49.2.net")
    model:evaluate()

    -- Run the model 
    for i=1,N do
        sentence = io.read()

        data = preprocess_data(sentence)

        pred = model:forward(data)
        _, rating = pred:max(2)
        print(rating)
    end
end

main()