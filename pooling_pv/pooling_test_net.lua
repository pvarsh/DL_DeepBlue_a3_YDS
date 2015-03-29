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

   local input_size = input:size()
   
   -- Precompute exp(beta * input)
   local exp_beta_input = torch.exp(torch.mul(input, self.beta))

   -- Declare tensor for sliding window derivative
   local denom_sum = torch.DoubleTensor(1, input_size[2])
   local dOut_dIn_window = torch.DoubleTensor(self.kW, input_size[2])

   -- Reset self.gradInput to zeros
   self.gradInput = torch.zeros(input_size)

   -- if self.gradInput == nil then
   --    print("ha")
   --    self.gradInput = torch.zeros(input_size)
   -- else
   --    print("he")
   --    print(self.gradInput)
   --    self.gradInput:zero()
   -- end
   print("Declared gradInput")
   print(self.gradInput)

   -- Set loop indexes
   local pos = 1   -- index of first row/element in pooling window
   local count = 1 -- index of row/element in gradOutput

   while pos + self.kW - 1 <= input_size[1] do

      -- compute sliding window derivative
      dOut_dIn_window = exp_beta_input[{ {pos, pos + self.kW - 1} }]
      denom_sum[{}] = dOut_dIn_window:sum(1)
      print("denom_sum")
      print(denom_sum)

      -- Loop through columns of dOut_dIn_window
      for col_idx=1,input_size[2] do
         -- Divide exp(beta input) by denominator sum
         -- After this dOut_dIn_window contains dOut_dIn derivative
         dOut_dIn_window[{ {},{col_idx} }]:div(denom_sum[1][col_idx])         

         self.gradInput[{ {pos,pos+self.kW-1},{col_idx} }]:add(dOut_dIn_window[{ {},{col_idx} }]:mul(gradOutput[count][col_idx]))
      end

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

-- If TemporalLogExpPooling is added to a nn.Sequential() model, it breaks on :add() function
-- model = nn.Sequential()
-- model:add(TemporalLogExpPooling(3,1,.5))

-- If TemporalLogExpPooling is used without nn.Sequential container, it seems to work
model = nn.TemporalLogExpPooling(3, 2, .5)

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