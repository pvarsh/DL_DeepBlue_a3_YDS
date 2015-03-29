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
   
   local input_size = input:size()
   print("input_size:")
   print(input_size)
   local output_size = torch.LongStorage(2)
   output_size[1] = math.floor((input_size[1] - self.kW)/self.dW + 1)
   output_size[2] = input_size[2]

   self.output = torch.DoubleTensor(output_size)
   local exp_beta_x = torch.DoubleTensor(input_size)

   exp_beta_x:mul(input, self.beta) -- multiplication and exponentiation
   exp_beta_x:exp()                 -- only is done once
   
   local pos = 1
   local count = 1
   while pos + (self.kW - 1) <= input_size[1] do
      self.output[count] = exp_beta_x[{ {pos,(pos + self.kW - 1)} }]:sum()
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

   local input_length = input:size()[1]
   local exp_input = input:clone() -- check if this indeed defines a local variable
   exp_input[{}] = torch.exp(input)
   self.gradInput = input:clone():zero()


   local pos = 1
   local count = 1

   while pos + (self.kW - 1) <= input_length do
      
      -- compute derivative vector
      local deriv = exp_input[{ {pos, pos+self.kW - 1} }]:clone() -- make local?
      deriv = deriv:div(deriv:sum())

      self.gradInput[{ {pos, pos+self.kW-1} }]:add(deriv * gradOutput[count])

      count = count + 1
      pos = pos + self.dW
   end
   
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


-- TEST SCRIPT FOR THE ABOVE FUNCTIONS
ninputs = 10

x = torch.rand(ninputs,2)
gradOutput = torch.ones(4):div(2)
print("Input tensor: ")
print(x)

-- If TemporalLogExpPooling is added to a nn.Sequential() model, it breaks on :add() function
-- model = nn.Sequential()
-- model:add(TemporalLogExpPooling(3,1,.5))

-- If TemporalLogExpPooling is used without nn.Sequential container, it seems to work
model = nn.TemporalLogExpPooling(3, 2, .5)

model_out = model:forward(x)
-- gradInput = model:backward(x, gradOutput)
print("Model output: ")
print(model_out)
print("Model gradInput: ")
-- print(gradInput)


-- USING MAX POOLING
model_max_pooling = nn.TemporalMaxPooling(3, 2)

model_max_pooling_out = model_max_pooling:forward(x)
print("Max pooling output: ")
print(model_max_pooling_out)
grad_input_max_pooling = model_max_pooling:backward(x, model_max_pooling_out)