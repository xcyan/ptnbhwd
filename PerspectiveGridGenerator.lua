-- code adapted from github repo
-- implemented by Yijie Guo (guoyijie@umich.edu) and Xinchen Yan (skywalkeryxc@gmail.com)

local PGG, parent = torch.class('nn.PerspectiveGridGenerator', 'nn.Module')

--[[
   AffineGridGeneratorBHWD(depth,height,width) :
   AffineGridGeneratorBHWD:updateOutput(transformMatrix)
   AffineGridGeneratorBHWD:updateGradInput(transformMatrix, gradGrids)

   AffineGridGeneratorBHWD will take 4x4 an affine image transform matrix (homogeneous 
   coordinates) as input, and output a grid, in normalized coordinates* that, once used
   with the Bilinear Sampler, will result in an affine transform.

   AffineGridGenerator 
   - takes (B,4,4)-shaped transform matrices as input (B=batch).
   - outputs a grid in BDHWD layout, that can be used directly with BilinearSamplerPerspective
  
   *: normalized coordinates [-1,1] correspond to the boundaries of the input image. 
]]

function PGG:__init(depth,height,width,focal_length)
   parent.__init(self)
   assert(depth > 1)
   assert(height > 1)
   assert(width > 1)
   self.depth = depth
   self.height = height
   self.width = width
   local dmin = 1/(focal_length + math.sqrt(3))
   local dmax = 1/(focal_length)
   print(focal_length)
   print(dmin .. ' ' .. dmax)

   --zt = 1, xt, yt [-1, 1]
   self.baseGrid = torch.Tensor(depth, height, width, 4)
   
   for k=1,self.depth do
     for i=1,self.height do
       for j=1,self.width do
          local disf = dmin + (k-1)/(self.depth-1) * (dmax-dmin)
          --print(disf)
          self.baseGrid[k][i][j][1] = 1/disf
          self.baseGrid[k][i][j][2] = (-1 + (i-1)/(self.height-1) * 2)/disf
          self.baseGrid[k][i][j][3] = (-1 + (j-1)/(self.width-1) * 2)/disf
          self.baseGrid[k][i][j][4] = 1
       end
     end
   end

   --[[self.baseGrid:select(4,1):fill(1)
   for i=1,self.height do
      self.baseGrid:select(4,2):select(2,i):fill(-1 + (i-1)/(self.height-1) * 2)
   end
   for j=1,self.width do
      self.baseGrid:select(4,3):select(3,j):fill(-1 + (j-1)/(self.width-1) * 2)
   end
   for k=1,self.depth do
      self.baseGrid:select(4,1):select(1,k):div(dmin + (k-1)/(self.depth-1) * (dmax-dmin))
      self.baseGrid:select(4,2):select(1,k):div(dmin + (k-1)/(self.depth-1) * (dmax-dmin))
      self.baseGrid:select(4,3):select(1,k):div(dmin + (k-1)/(self.depth-1) * (dmax-dmin))
   end
   self.baseGrid:select(4,4):fill(1)]]

   self.batchGrid = torch.Tensor(1, depth, height, width, 4):copy(self.baseGrid)
end

local function addOuterDim(t)
   local sizes = t:size()
   local newsizes = torch.LongStorage(sizes:size()+1)
   newsizes[1]=1
   for i=1,sizes:size() do
      newsizes[i+1]=sizes[i]
   end
   return t:view(newsizes)
end

function PGG:updateOutput(_transformMatrix)
   local transformMatrix
   if _transformMatrix:nDimension()==2 then
      transformMatrix = addOuterDim(_transformMatrix)
   else
      transformMatrix = _transformMatrix
   end
   assert(transformMatrix:nDimension()==3
          and transformMatrix:size(2)==4
          and transformMatrix:size(3)==4
          , 'please input affine transform matrices (bx4x4)')
   local batchsize = transformMatrix:size(1)
   
   if self.batchGrid:size(1) ~= batchsize then
      self.batchGrid:resize(batchsize, self.depth, self.height, self.width, 4)
      for i=1,batchsize do
         self.batchGrid:select(1,i):copy(self.baseGrid)
      end
   end

   self.output:resize(batchsize, self.depth, self.height, self.width, 4)
   local flattenedBatchGrid = self.batchGrid:view(batchsize, self.depth*self.width*self.height, 4)
   local flattenedOutput = self.output:view(batchsize, self.depth*self.width*self.height, 4)
   torch.bmm(flattenedOutput, flattenedBatchGrid, transformMatrix:transpose(2,3)) 
   if _transformMatrix:nDimension()==2 then
      self.output = self.output:select(1,1)
   end
   --print(self.output:size())
   --[[for k=1,self.depth do
     for i=1,self.height do
       for j=1,self.width do
         print(string.format('[%d %d %d] (%.3f, %.3f, %.3f) --> (%.3f, %.3f, %.3f)',
          k,i,j, self.batchGrid[1][k][i][j][1], self.batchGrid[1][k][i][j][2], self.batchGrid[1][k][i][j][3],
          self.output[1][k][i][j][1], self.output[1][k][i][j][2], self.output[1][k][i][j][3]))
        end
      end
    end]]

   return self.output
end

function PGG:updateGradInput(_transformMatrix, _gradGrid)
   local transformMatrix, gradGrid
   if _transformMatrix:nDimension()==2 then
      transformMatrix = addOuterDim(_transformMatrix)
      gradGrid = addOuterDim(_gradGrid)
   else
      transformMatrix = _transformMatrix
      gradGrid = _gradGrid
   end

   local batchsize = transformMatrix:size(1)

   local flattenedGradGrid = gradGrid:view(batchsize, self.depth*self.width*self.height, 4)
   local flattenedBatchGrid = self.batchGrid:view(batchsize, self.depth*self.width*self.height, 4)
   self.gradInput:resizeAs(transformMatrix):zero()
   self.gradInput:baddbmm(flattenedGradGrid:transpose(2,3), flattenedBatchGrid) ---????
   -- torch.baddbmm doesn't work on cudatensors for some reason

   if _transformMatrix:nDimension()==2 then
      self.gradInput = self.gradInput:select(1,1)
   end

   return self.gradInput
end
