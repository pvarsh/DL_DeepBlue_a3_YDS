require 'torch';
ffi = require 'ffi';
require 'string';

-- the dictionary of characters, 69
dictionary = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
train = torch.load("/scratch/courses/DSGA1008/A3/data/train.t7b")

function quantization(document,frame,length)
    document = document:lower()
    char_quant = torch.Tensor(frame,length):fill(0)
    for i = 1,math.min(document:len(), length) do
        character = document:sub(i,i)
        position = string.find(dictionary,character)
        if position ~= nil then
            char_quant[position][i] = 1 
        end
    end
    return char_quant
end

function gettensor(i, j, frame, length)
    index = train.index[i][j]
    document = ffi.string(torch.data(train.content:narrow(1, index, 1)))
    x = quantization(document, frame, length)
    return x
end
