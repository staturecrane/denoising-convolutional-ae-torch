require 'image'
gnuplot = require 'gnuplot'
require 'cutorch'
require 'xlua'
hasCudnn, cudnn = pcall(require, 'cudnn')
assert(hasCudnn)

cutorch.setHeapTracking(true)
torch.setheaptracking(true)
torch.setdefaulttensortype('torch.FloatTensor')

train_size = 100

test_images = torch.Tensor(40, 3, 96, 96):cuda()
train_images = torch.Tensor(train_size, 3, 96, 96):cuda()

function getFilename(num)
  length = #tostring(num)
  filename = '2001_96x96_jpg/2001'
  for i=1, (5 - length) do
    filename  = filename .. 0
  end
  filename = filename .. num .. '.jpg'
  return filename
end

local shuffle = torch.randperm(65000)

train_count = 1

print('Moving images to GPU ...')

idx = 1

function createSamples()
  for i = 1, train_size - 1 do
    xlua.progress(i, train_size)
    local sample = image.load(getFilename(shuffle[idx + i]))
    train_images[train_count] = sample
    train_count = train_count + 1
    idx = idx + 1
    if idx >= 65000 then idx = 1 end
  end
  train_count = 1
end

createSamples()

test_count = 1

for i = 70001, 70040 do
  local sample = image.load(getFilename(i))
  test_images[test_count] = sample
  test_count = test_count + 1
end

require 'nn'
require 'dpnn'

encoder = nn.Sequential()
encoder:add(nn.SpatialConvolution(3, 64, 3, 3, 1, 1, 1, 1))
encoder:add(nn.ReLU(true))
encoder:add(nn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1))
encoder:add(nn.ReLU(true))

decoder = nn.Sequential()
decoder:add(nn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1))
decoder:add(nn.ReLU(true))
decoder:add(nn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1))
decoder:add(nn.ReLU(true))
decoder:add(nn.SpatialConvolution(64, 3, 3, 3, 1, 1, 1, 1))
decoder:add(nn.Sigmoid(true))

autoencoder = nn.Sequential()
noiser = nn.WhiteNoise(0, 0.5) -- Add white noise to inputs during training
autoencoder:add(noiser)
autoencoder:add(encoder)
autoencoder:add(decoder)

autoencoder = autoencoder:cuda():cuda()
cudnn.convert(autoencoder, cudnn):cuda()
criterion = nn.BCECriterion():cuda()

theta, gradTheta = autoencoder:getParameters()

autoencoder:training()

optimParams = {learning_rate = 0.00001}
losses = {}

require 'optim'

batch_size = 50

function trainEpoch(index, batchSize, epoch)
    local size = math.min(index + batchSize - 1, train_size) - index
    local x = train_images:narrow(1, size, batchSize)

    function feval(params)
      if theta ~= params then
        theta:copy(params)
      end

      gradTheta:zero()

      local xHat = autoencoder:forward(x)
      local loss = criterion:forward(xHat, x)
      local gradLoss = criterion:backward(xHat, x)
      autoencoder:backward(x, gradLoss)
      return loss, gradTheta
    end

    __, loss = optim['adam'](feval, theta, optimParams)
    losses[#losses + 1] = loss[1]

    x = nil
    autoencoder:clearState()

    collectgarbage('collect')
    collectgarbage('collect')

    print('Loss at iteration ' .. index .. ': ' .. loss[1])
    print('Memory usage: ' .. collectgarbage('count'))
    if size + index < train_size - 1 then
      return trainEpoch(index + size, batchSize, epoch)
    end
    return loss[1]
end

for epoch = 1, 400 do
  if epoch > 1 then
    createSamples()
  end
  local loss = trainEpoch(1, batch_size, epoch)
  print('loss at epoch ' .. epoch .. ': ' .. loss)
  autoencoder:evaluate()
  xHat = autoencoder(test_images)
  image.save('reconstructions/2001_epoch_' .. epoch .. '.png', xHat[1])
  autoencoder:training()
end
