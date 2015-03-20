require 'nn';

local TemporalLogExpPooling, parent = torch.class('nn.TemporalLogExpPooling', 'nn.Module')

function TemporalLogExpPooling:__init(kW, dW, beta)
   parent.__init(self)

   self.kW = kW
   self.dW = dW
   self.beta = beta

   self.indices = torch.Tensor()
end

function TemporalLogExpPooling:updateOutput(input)
   -----------------------------------------------
   -- your code here
   -----------------------------------------------
   
   local input_length = input:size()[1]
   local output_length = math.floor((input_length - self.kW)/self.dW + 1)
   print("Output length: ", output_length)
   self.output = torch.DoubleTensor(output_length)
   local y = torch.DoubleTensor(input:size())

   y:mul(input, self.beta) -- multiplication and exponentiation
   y:exp()                 -- only is done once
   
   local pos = 1
   local count = 1
   while pos + (self.kW - 1) <= input_length do
      print("Position", pos)
      self.output[count] = y[{ {pos,(pos + self.kW - 1)} }]:sum()
      pos = pos + self.dW
      count = count + 1
   end
   ((self.output:mul(1/self.kW)):log()):mul(1/self.beta)

   return self.output
end

function TemporalLogExpPooling:updateGradInput(input, gradOutput)
   -----------------------------------------------
   -- your code here
   -----------------------------------------------
   print("SIIIiZE: ", self.output:size())
   return self.gradInput
end

function TemporalLogExpPooling:empty()
   self.gradInput:resize()
   self.gradInput:storage():resize(0)
   self.output:resize()
   self.output:storage():resize(0)
   self.indices:resize()
   self.indices:storage():resize(0)
end

-- Script sets up a neural net that only pools using
-- provided pooling algorithm.

require 'nn';

ninputs = 10


x = torch.rand(ninputs,1)
print(x)

-- If TemporalLogExpPooling is added to a nn.Sequential() model, it breaks on :add() function
-- model = nn.Sequential()
-- model:add(TemporalLogExpPooling(3,1,.5))

-- If TemporalLogExpPooling is used without nn.Sequential container, it seems to work
model = nn.TemporalLogExpPooling(3, 2, .5)

print(model:forward(x))
-- print(model:backward())