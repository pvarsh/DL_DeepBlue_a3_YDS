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

   -- Check if self.output exists (nonzero dimension)
   -- if self.output:dim() == 0 then 
   --   self.output = torch.Tensor(output_size)
   -- end
   self.output = torch.Tensor(output_size)

   local exp_beta_x = input:clone()
   exp_beta_x:mul(self.beta):exp()
   for step_idx=1,output_size[2] do
      local win_start = (step_idx - 1)*self.dW + 1
      local win_end   = win_start + self.kW - 1
      local sigma = exp_beta_x[{ {},{win_start,win_end},{} }]:sum(2)
      self.output[{ {},step_idx,{} }] = sigma:div(self.kW):log():div(self.beta)
   end

   --------- END: OUR CODE

   return self.output
end

function TemporalLogExpPooling:updateGradInput(input, gradOutput)
   
   --------- OUR CODE
   local in_size = input:size()
   local out_size = gradOutput:size()

   if self.gradInput:dim() == 0 then
      self.gradInput = torch.zeros(in_size)
   else
      self.gradInput:zero() -- Should this be done manually outside of updateGradInput?
   end

   local exp_beta_x = input:clone():mul(self.beta):exp()

   for step_idx=1,out_size[2] do
      local win_start = (step_idx - 1)*self.dW + 1
      local win_end = win_start + self.kW - 1
      local window = exp_beta_x[{ {}, {win_start,win_end}, {} }]:clone()
      local sigma = window:sum(2)
      -- print ("sigma:size()", sigma:size())
      -- print ("window:size()", window:size())


      for batch_idx=1,in_size[1] do
         for frame_idx=1,in_size[3] do
            window[{ {batch_idx},{},{frame_idx} }]:div(sigma[{ batch_idx,1,frame_idx }]):mul(gradOutput[{ batch_idx,step_idx,frame_idx }])
         end
      end

      -- print ("sigma:size()", sigma:size())
      -- window:cdiv(sigma):cmul(self.gradOutput[{ {},{step_idx},{} }])
      self.gradInput[{ {},{win_start,win_end},{} }]:add(window)
   end
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
