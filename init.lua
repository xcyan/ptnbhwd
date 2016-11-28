require 'nn'
local withCuda = pcall(require, 'cutorch')

require 'libptn'
if withCuda then
   require 'libcuptn'
end

--require('ptn.AffineTransformMatrixGenerator')
--require('ptn.AffineGridGeneratorBHWD')
require('ptn.BilinearSamplerBHWD')

require('ptn.PerspectiveGridGenerator')
require('ptn.BilinearSamplerPerspective')

--require('ptn.test')

return nn
