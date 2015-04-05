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
   
   ----  Note on dimensions:
   --    input:size()[1] is the size of minibatch
   --    input:size()[2] is the length of the feature vector
   --    input:size()[3] is the number of frames 

   local input_size = input:size()
   local output_size = torch.LongStorage(3)

   output_size[1] = input_size[1]
   output_size[2] = math.floor((input_size[1] - self.kW)/self.dW)
   output_size[3] = input_size[3]

   self.output = torch.DoubleTensor(output_size)
   -- print('self.output:size()')
   -- print(self.output)
   local exp_beta_x = torch.DoubleTensor(input_size)

   exp_beta_x:mul(input, self.beta) -- multiplication and exponentiation
   exp_beta_x:exp()                 -- only is done once
   
   for batch_idx = 1,input_size[1] do
      for frame_idx = 1,input_size[3] do
         for vec_idx = 1,output_size[2] do
            self.output[{ batch_idx, vec_idx, frame_idx }] = ((input[{ batch_idx, {vec_idx*self.dW, vec_idx*self.dW + self.kW}, frame_idx }]:sum()):div(self.kW):log()):div(self.beta)
         end -- end: feature vector loop
      end -- end: frame loop
   end -- end: minibatch loop
   -- local pos = 1
   -- local count = 1
   -- while pos + (self.kW - 1) <= input_size[1] do
   --    self.output[count] = exp_beta_x[{ {pos,(pos + self.kW - 1)} }]:sum(1)
   --    pos = pos + self.dW
   --    count = count + 1
   -- end
   -- ((self.output:mul(1/self.kW)):log()):mul(1/self.beta)
   return self.output
end

function TemporalLogExpPooling:updateGradInput(input, gradOutput)

   local input_size = input:size()
   
   -- Precompute exp(beta * input)
   local exp_beta_input = torch.exp(torch.mul(input, self.beta))

   -- Declare tensor for sliding window derivative
   local denom_sum = torch.Tensor(1, input_size[2])
   local dOut_dIn_window = torch.Tensor(self.kW, input_size[2])

   -- Reset self.gradInput to zeros
   -- TODO: check if self.gradInput is already of correct dimension and
   --       reset to zero. This might be better memory management.
   self.gradInput = torch.zeros(input_size)

   -- Set loop indexes
   local pos = 1   -- index of first row/element in pooling window
   local count = 1 -- index of row/element in gradOutput

   while pos + self.kW - 1 <= input_size[1] do

      -- Compute sliding window derivative
      dOut_dIn_window[{}] = exp_beta_input[{ {pos, pos + self.kW - 1} }]
      denom_sum[{}] = dOut_dIn_window:sum(1)

      -- Loop through columns of dOut_dIn_window
      for col_idx=1,input_size[2] do
         -- Divide exp(beta input) by denominator sum
         -- After this dOut_dIn_window contains dOut_dIn derivative
         dOut_dIn_window[{ {},{col_idx} }]:div(denom_sum[1][col_idx])         

         -- Add dE/dx_{i} * dx_{i}/dx_{i-1} to self.gradInput
         -- TODO: rewrite using several lines for readability
         self.gradInput[{ {pos,pos+self.kW-1},{col_idx} }]:add(dOut_dIn_window[{ {},{col_idx} }]:mul(gradOutput[count][col_idx]))
      end

      -- Update loop counters
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
batch_size = 2

x = torch.rand(ninputs, batch_size)
gradOutput = torch.ones(4):div(2)
print("Input tensor: ")
print(x)

model = nn.Sequential()
model:add(nn.TemporalLogExpPooling(3,1,.5))
-- model:add(nn.TemporalMaxPooling(3,2))

-- If TemporalLogExpPooling is used without nn.Sequential container, it seems to work
-- model = nn.TemporalLogExpPooling(3, 2, .5)

-- Feed forward
model_out = model:forward(x)
print("Model output: ")
print(model_out)

-- Define fake gradOutput to feed backward
gradOutput = torch.ones(model_out:size()):div(2)
print("gradOutput")
print(gradOutput)

-- Feed backward
gradInput = model:backward(x, gradOutput)
print("Model gradInput: ")
print(gradInput)


-- USING MAX POOLING
model_max_pooling = nn.TemporalMaxPooling(3, 2)

model_max_pooling_out = model_max_pooling:forward(x)
print("Max pooling output: ")
print(model_max_pooling_out)
grad_input_max_pooling = model_max_pooling:backward(x, model_max_pooling_out)


--------------------------------------------------------------------
-- SOME TESTS

print(">>>>>>>>>>SOME TESTS")
print(">> TEST 1")
x = torch.Tensor(3,1)
x[1] = 1
x[2] = 2
x[3] = 3
beta = 1
output = torch.Tensor(2,1)
gradOutput = torch.Tensor(2,1)
gradInput = torch.Tensor(3,1)
gradOutput[1] = 0.5
gradOutput[2] = 0.2

-- Compute output 'by hand'
exp_beta_x = x:clone():mul(beta):exp()
output[1] = torch.log((exp_beta_x[1] + exp_beta_x[2])*1/2) * 1/beta
output[2] = torch.log((exp_beta_x[2] + exp_beta_x[3])*1/2) * 1/beta
print("Manually computed output: ")
print(output)

-- Compute output using module
model = nn.TemporalLogExpPooling(2,1,beta)
print("TemporalLogExpPooling forward output: ")
print(model:forward(x))

gradInput[1] = exp_beta_x[1] / (exp_beta_x[1][1] + exp_beta_x[2][1]) * gradOutput[1]
gradInput[2] = exp_beta_x[2] / (exp_beta_x[1][1] + exp_beta_x[2][1]) * gradOutput[1] + exp_beta_x[2] / (exp_beta_x[2][1] + exp_beta_x[3][1]) * gradOutput[2]
gradInput[3] = exp_beta_x[3] / (exp_beta_x[2][1] + exp_beta_x[3][1]) * gradOutput[2]

print("Manually computed gradInput")
print(gradInput)
print("TemporalLogExpPooling backward output: ")
print(model:backward(x, gradOutput))

print(">> TEST 2")
x = torch.Tensor(4,2)
x[1][1] = 1
x[2][1] = 2
x[3][1] = 3
x[4][1] = 4
x[1][2] = 4
x[2][2] = 3
x[3][2] = 2
x[4][2] = 1
beta = .5
N = 2
step = 1
output = torch.Tensor(3,2)
gradOutput = torch.Tensor(3,2)
gradInput = torch.Tensor(4,2)
gradOutput[1][1] = 0.5
gradOutput[2][1] = 0.4
gradOutput[3][1] = 0.3
gradOutput[1][2] = 0.6
gradOutput[2][2] = 0.7
gradOutput[3][2] = 0.8


-- Compute output 'by hand'
exp_beta_x = x:clone():mul(beta):exp()
-- print("x:")
-- print(x)
-- print("exp_beta_x")
-- print(exp_beta_x)
output[1][1] = torch.log((exp_beta_x[1][1] + exp_beta_x[2][1])*1/N) * 1/beta
output[2][1] = torch.log((exp_beta_x[2][1] + exp_beta_x[3][1])*1/N) * 1/beta
output[3][1] = torch.log((exp_beta_x[3][1] + exp_beta_x[4][1])*1/N) * 1/beta
output[1][2] = torch.log((exp_beta_x[1][2] + exp_beta_x[2][2])*1/N) * 1/beta
output[2][2] = torch.log((exp_beta_x[2][2] + exp_beta_x[3][2])*1/N) * 1/beta
output[3][2] = torch.log((exp_beta_x[3][2] + exp_beta_x[4][2])*1/N) * 1/beta
print("Manually computed output: ")
print(output)

-- Compute output using module
model = nn.TemporalLogExpPooling(N,step,beta)
print("TemporalLogExpPooling forward output: ")
print(model:forward(x))

-- gradInput[1] = exp_beta_x[1] / (exp_beta_x[1][1] + exp_beta_x[2][1]) * gradOutput[1]
-- gradInput[2] = exp_beta_x[2] / (exp_beta_x[1][1] + exp_beta_x[2][1]) * gradOutput[1] + exp_beta_x[2] / (exp_beta_x[2][1] + exp_beta_x[3][1]) * gradOutput[2]
-- gradInput[3] = exp_beta_x[3] / (exp_beta_x[2][1] + exp_beta_x[3][1]) * gradOutput[2]

-- print("Manually computed gradInput")
-- print(gradInput)
-- print("TemporalLogExpPooling backward output: ")
-- print(model:backward(x, gradOutput))
