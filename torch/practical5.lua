-- Nice example:  http://stackoverflow.com/questions/33545325/torch-backward-through-gmodule
require 'torch'
require 'nn'
require 'nngraph'
require 'graph'

-- params for the linear layer
params = {
	x3_size1 = 10,
	x3_size2 = 30
}
-- dummy nodes to take input data as nodes in graph
x1 = nn.Identity()()
x2 = nn.Identity()()
x3 = nn.Identity()()

-- modeling output = x1 + x2 cmul linear(x3)
l3 = nn.Linear(params.x3_size1, params.x3_size2)(x3)
m23 = nn.CMulTable()({x2,l3})
add = nn.CAddTable()({x1, m23})

-- specify the inputs and outputs of the graph
m = nn.gModule({x1,x2,x3}, {add})

-- check sizes
inpx3 = torch.rand(params.x3_size1)
inpx2 = torch.rand(params.x3_size2)
inpx1 = torch.rand(params.x3_size2)

output = m:forward({inpx1, inpx2, inpx3})
print(output)

