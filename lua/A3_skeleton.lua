-------------------------------------------------------------------------
-- In this part of the assignment you will become more familiar with the
-- internal structure of torch modules and the torch documentation.
-- You must complete the definitions of updateOutput and updateGradInput
-- for a 1-d log-exponential pooling module as explained in the handout.
-- 
-- Refer to the torch.nn documentation of nn.TemporalMaxPooling for an
-- explanation of parameters kW and dW.
-- 
-- Refer to the torch.nn documentation overview for explanations of the 
-- structure of nn.Modules and what should be returned in self.output 
-- and self.gradInput.
-- 
-- Don't worry about trying to write code that runs on the GPU.
--
-- Please find submission instructions on the handout
------------------------------------------------------------------------

local TemporalLogExpPooling, parent = torch.class('nn.TemporalLogExpPooling', 'nn.Module')

function TemporalLogExpPooling:__init(kW, dW, beta)
   parent.__init(self)

   self.kW = kW
   self.dW = dW
   self.beta = beta

   self.indices = torch.Tensor()
end

function TemporalLogExpPooling:updateOutput(input)

   --------- OUR CODE
   
   local input_size = input:size()
   local output_size = torch.LongStorage(3)

   output_size[1] = input_size[1]
   output_size[2] = math.floor((input_size[2] + 1 - self.kW)/self.dW)
   output_size[3] = input_size[3]

   self.output = torch.DoubleTensor(output_size)
   -- print('self.output:size()')
   -- print(self.output)
   local exp_beta_x = torch.DoubleTensor(input_size)

   exp_beta_x:mul(input, self.beta) -- multiplication and exponentiation
   exp_beta_x:exp()                 -- only is done once
   
   for batch_idx = 1,input_size[1] do
      for frame_idx = 1,input_size[3] do
         for step_idx = 1,output_size[2] do
            -- print(step_idx*self.dW, step_idx*self.dW + self.kW - 1)
            local operand = input[{ batch_idx, {step_idx*self.dW, step_idx*self.dW + self.kW - 1}, frame_idx }]
            operand = operand:sum()
            operand = operand / self.kW
            operand = torch.log(operand)
            operand = operand / self.beta
           
            self.output[{ batch_idx, step_idx, frame_idx }] = operand
         end -- end: feature vector loop
      end -- end: frame loop
   end -- end: minibatch loop
   --------- END: OUR CODE

   return self.output
end

function TemporalLogExpPooling:updateGradInput(input, gradOutput)
   
   --------- OUR CODE

   -- print('Input size:: ')
   -- print(input:size())
   -- print('gradOutput size:: ')
   -- print(gradOutput:size())


   local in_size = input:size()
   local out_size = gradOutput:size()

   self.gradInput = torch.zeros(in_size)
   local exp_beta_x = input:clone():mul(self.beta):exp()

   for batch_idx=1,out_size[1] do
      for frame_idx=1,out_size[3] do
         for step_idx=1,out_size[2] do
            local gradInput_win_start = (step_idx - 1)*self.dW + 1
            local gradInput_win_end   = gradInput_win_start + self.kW - 1
            local denom_sum = exp_beta_x[{ batch_idx, {gradInput_win_start, gradInput_win_end}, frame_idx }]:sum()

            local dOut_dIn = exp_beta_x[{batch_idx, {gradInput_win_start, gradInput_win_end}, frame_idx}]:clone():div(denom_sum)
            self.gradInput[{ batch_idx, {gradInput_win_start, gradInput_win_end}, frame_idx }]:add(dOut_dIn:mul(gradOutput[{batch_idx, step_idx, frame_idx}]))
         end
      end
   end

   -- local input_size = input:size()
   
   -- -- Precompute exp(beta * input)
   -- local exp_beta_input = torch.exp(torch.mul(input, self.beta))

   -- -- Declare tensor for sliding window derivative
   -- local denom_sum = torch.Tensor(1, input_size[2])
   -- local dOut_dIn_window = torch.Tensor(self.kW, input_size[2])

   -- -- Reset self.gradInput to zeros
   -- -- TODO: check if self.gradInput is already of correct dimension and
   -- --       reset to zero. This might be better memory management.
   -- self.gradInput = torch.zeros(input_size)

   -- -- Set loop indexes
   -- local pos = 1   -- index of first row/element in pooling window
   -- local count = 1 -- index of row/element in gradOutput

   -- while pos + self.kW - 1 <= input_size[1] do

   --    -- Compute sliding window derivative
   --    dOut_dIn_window[{}] = exp_beta_input[{ {pos, pos + self.kW - 1} }]
   --    denom_sum[{}] = dOut_dIn_window:sum(1)

   --    -- Loop through columns of dOut_dIn_window
   --    for col_idx=1,input_size[2] do
   --       -- Divide exp(beta input) by denominator sum
   --       -- After this dOut_dIn_window contains dOut_dIn derivative
   --       dOut_dIn_window[{ {},{col_idx} }]:div(denom_sum[1][col_idx])         

   --       -- Add dE/dx_{i} * dx_{i}/dx_{i-1} to self.gradInput
   --       -- TODO: rewrite using several lines for readability
   --       self.gradInput[{ {pos,pos+self.kW-1},{col_idx} }]:add(dOut_dIn_window[{ {},{col_idx} }]:mul(gradOutput[count][col_idx]))
   --    end

   --    -- Update loop counters
   --    count = count + 1
   --    pos = pos + self.dW
   -- end
   --------- END: OUR CODE
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
