require 'image'
gnuplot = require 'gnuplot'
require 'cutorch'
require 'xlua'
hasCudnn, cudnn = pcall(require, 'cudnn')
assert(hasCudnn)

torch.setdefaulttensortype('torch.FloatTensor')

train_size = 100
test_size = 40
channels = 3
width = 96
height = 96

test_images = torch.Tensor(test_size, channels, height, width):cuda()
train_images = torch.Tensor(train_size, channels, height, width):cuda()

function getFilename(num)
  length = #tostring(num)
  filename = '2001_96x96_jpg/2001'
  for i=1, (5 - length) do
    filename  = filename .. 0
  end
  filename = filename .. num .. '.jpg'
  return filename
end

train_random_end = 50000
train_idx = 1


function createSamples()
  print('Moving images to GPU ...')
  for i = 1, train_size do
    xlua.progress(i, train_size)
    local sample = image.load(getFilename(torch.random(1, train_random_end)))
    train_images[train_idx] = sample
    train_idx = train_idx + 1
  end
  train_idx = 1
end

test_count = 1
test_start = 60000
test_random_end = 70000

for i = 1, test_size  do
  local sample = image.load(getFilename(torch.random(test_start, test_random_end)))
  test_images[test_count] = sample
  test_count = test_count + 1
end

require 'nn'
require 'dpnn'

feature_size = 64
kernel_size = 3
padding = 1
stride = 1

encoder = nn.Sequential()
encoder:add(nn.SpatialConvolution(channels, feature_size, kernel_size, kernel_size, stride, stride, padding, padding))
encoder:add(nn.ReLU(true))
encoder:add(nn.SpatialConvolution(feature_size, feature_size, kernel_size, kernel_size, stride, stride, padding, padding))
encoder:add(nn.ReLU(true))

decoder = nn.Sequential()
decoder:add(nn.SpatialConvolution(feature_size, feature_size, kernel_size, kernel_size, stride, stride, padding, padding))
decoder:add(nn.ReLU(true))
decoder:add(nn.SpatialConvolution(feature_size, feature_size, kernel_size, kernel_size, stride, stride, padding, padding))
decoder:add(nn.ReLU(true))
decoder:add(nn.SpatialConvolution(feature_size, channels, kernel_size, kernel_size, stride, stride, padding, padding))
decoder:add(nn.Sigmoid(true))

autoencoder = nn.Sequential()

noiser = nn.WhiteNoise(0, 0.25)
autoencoder:add(noiser)
autoencoder:add(encoder)
autoencoder:add(decoder)

autoencoder = autoencoder:cuda()
cudnn.convert(autoencoder, cudnn)
criterion = nn.BCECriterion():cuda()

theta, gradTheta = autoencoder:getParameters()

autoencoder:training()

learning_rate = 0.000001

optimParams = {learning_rate}
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

    print('Loss at iteration ' .. index .. ': ' .. loss[1])
    print('Memory usage: ' .. collectgarbage('count'))
    if size + index < train_size - 1 then
      return trainEpoch(index + size, batchSize, epoch)
    end
    return loss[1]
end

for epoch = 1, 300 do

  createSamples()

  local loss = trainEpoch(1, batch_size, epoch)
  print('loss at epoch ' .. epoch .. ': ' .. loss)

  if epoch == 1 then
    for i = 1, test_size do
      image.save('reconstructions/originals/2001_original_' .. i .. '.jpg', test_images[i])
    end
  end

  autoencoder:evaluate()
  xHat = autoencoder(test_images)

  for i = 1, test_size do
    image.save('reconstructions/' .. i .. '/2001_epoch_' .. epoch .. '.jpg', xHat[i])
  end

  autoencoder:training()

end
