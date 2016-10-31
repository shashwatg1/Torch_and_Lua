--[[
CIFAR-10 dataset training and testing using a deep learning implementation

Note: use 'th' to compile

We shall use the CIFAR-10 dataset, which has the classes:
'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'.
The images in CIFAR-10 are of size 3x32x32, i.e. 3-channel color images of 32x32 pixels in size.
The dataset has 50,000 training images and 10,000 test images in total. 

5 main steps:
    Load and normalize data
    Define Neural Network
    Define Loss function
    Train network on training data
    Test network on test data.
]]

torch.manualSeed(0)

-- 1. Load and normalize Data

require 'paths' -- check if the required dataset file is present in the working directory, else download it
if (not paths.filep("cifar10torchsmall.zip")) then
    os.execute('wget -c https://s3.amazonaws.com/torch7/data/cifar10torchsmall.zip')
    os.execute('unzip cifar10torchsmall.zip')
end

trainset = torch.load('cifar10-train.t7')
testset = torch.load('cifar10-test.t7')
classes = {'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}

print(trainset)
-- print(#trainset.data)

--itorch.image(trainset.data[100]) -- display the 100-th image in dataset
-- print(classes[trainset.label[100]])

--[[
Now, to prepare the dataset to be used with nn.StochasticGradient,
a couple of things have to be done according to it's documentation.
->    The dataset has to have a :size() function.
->    The dataset has to have a [i] index operator, so that dataset[i] returns the ith sample in the datset.
Both can be done quickly:
]]

-- setmetatable sets the index operator.
setmetatable(trainset, 
    {__index = function(t, i) 
                    return {t.data[i], t.label[i]} 
                end}
);
trainset.data = trainset.data:double() -- convert the data from a ByteTensor to a DoubleTensor.

function trainset:size() 
    return self.data:size(1) 
end

-- print(trainset:size()) -- just to test

-- print(trainset[33]) -- load sample number 33.
--itorch.image(trainset[33][1])


--  tensor indexing operator example:
-- redChannel = trainset.data[{ {}, {1}, {}, {}  }]
-- this picks {all images, 1st channel, all vertical pixels, all horizontal pixels}

-- print(#redChannel)

-- In this indexing operator, you initally start with [{ }].
-- You can pick all elements in a dimension using {} or pick a particular element using {i}
-- where i is the element index. You can also pick a range of elements using {i1, i2},
-- for example {3,5} gives us the 3,4,5 elements.


-- One of the most important things you can do in conditioning your data (in general in data-science or 
-- machine learning) is to make your data to have a mean of 0.0 and standard-deviation of 1.0.

mean = {} -- store the mean, to normalize the test set in the future
stdv  = {} -- store the standard-deviation for the future
for i=1,3 do -- over each image channel
    mean[i] = trainset.data[{ {}, {i}, {}, {}  }]:mean() -- mean estimation
    print('Channel ' .. i .. ', Mean: ' .. mean[i])
    trainset.data[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction
    
    stdv[i] = trainset.data[{ {}, {i}, {}, {}  }]:std() -- std estimation
    print('Channel ' .. i .. ', Standard Deviation: ' .. stdv[i])
    trainset.data[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
end
-- Our training data is now normalized and ready to be used.


-- 2. Define the Neural Network

require 'nn'

net = nn.Sequential()
net:add(nn.SpatialConvolution(3, 6, 5, 5)) -- 3 input image channels, 6 output channels, 5x5 convolution kernel
net:add(nn.ReLU())                       -- non-linearity 
net:add(nn.SpatialMaxPooling(2,2,2,2))  -- A max-pooling operation that looks at 2x2 windows and finds the max.
net:add(nn.SpatialConvolution(6, 16, 5, 5))
net:add(nn.ReLU())                      -- non-linearity 
net:add(nn.SpatialMaxPooling(2,2,2,2))
net:add(nn.View(16*5*5))                -- reshapes from a 3D tensor of 16x5x5 into 1D tensor of 16*5*5
net:add(nn.Linear(16*5*5, 120))     -- fully connected layer (matrix multiplication between input and weights)
net:add(nn.ReLU())                  -- non-linearity 
net:add(nn.Linear(120, 84))
net:add(nn.ReLU())                  -- non-linearity 
net:add(nn.Linear(84, 10))          -- 10 is the number of outputs of the network (in this case, 10 digits)
net:add(nn.LogSoftMax())       -- converts the output to a log-probability. Useful for classification problems


-- 3. Define the Loss Function

-- Let us use a Log-likelihood classification loss. It is well suited for most classification problems.
criterion = nn.ClassNLLCriterion()


-- 4. Train network on training data

-- This is when things start to get interesting.
-- Let us first define an nn.StochasticGradient object.
-- Then we will give our dataset to this object's :train function, and that will get the ball rolling.

trainer = nn.StochasticGradient(net, criterion)
trainer.learningRate = 0.001
trainer.maxIteration = 5 -- just do 5 epochs of training.

trainer:train(trainset)


-- 5. Test network on test data

-- print(classes[testset.label[100]])
--itorch.image(testset.data[100])

-- normalizing
testset.data = testset.data:double()   -- convert from Byte tensor to Double tensor
for i=1,3 do -- over each image channel
    testset.data[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction    
    testset.data[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
end

-- testing one example
print(classes[testset.label[100]])
--itorch.image(testset.data[100])
predicted = net:forward(testset.data[100])
-- the output of the network is Log-Probabilities. To convert them to probabilities, you have to take e^x 
print(predicted:exp())
-- You can see the network predictions. The network assigned a probability to each classes, given the image.
-- To make it clearer, let us tag each probability with it's class-name:
for i=1,predicted:size(1) do
    print(classes[i], predicted[i])
end


--Alright but how many in total seem to be correct over the test set?
correct = 0
for i=1,10000 do
    local groundtruth = testset.label[i]
    local prediction = net:forward(testset.data[i])
    local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
    if groundtruth == indices[1] then
        correct = correct + 1
    end
end

print(correct, 100*correct/10000 .. ' % ')


--Hmmm, what are the classes that performed well, and the classes that did not perform well:
class_performance = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
for i=1,10000 do
    local groundtruth = testset.label[i]
    local prediction = net:forward(testset.data[i])
    local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
    if groundtruth == indices[1] then
        class_performance[groundtruth] = class_performance[groundtruth] + 1
    end
end

for i=1,#classes do
    print(classes[i], 100*class_performance[i]/1000 .. ' %')
end
