--[[
image is a Torch package for processing images
it includes:
	saving, opening
	simple transformations, scaling, rotations
	convolutions, warping
	drawing routines
	GUI
	Coulour space conversion
	Tensor Constructions
]]

-- use 'qlua -lenv' to compile

-- the package has lena.jgp by default and can be accessed by image.lena()

require 'image'

im = image.lena()
-- im = image.load('image.jpg') -- this is an alternate
-- image.display(im) -- displays on a window named image.display

print(#im) -- dimensions ( 3, 512, 512 )
print(torch.type(im)) -- DoubleTensor

im_scale = image.scale(im, 512/2, 512/2)
print(#im_scale) -- dimensions ( 3, 256, 256 )
-- image.display(im_scale) -- displays on a window named image.display


im_green = im_scale:clone()
im_green[1]:fill(0)
im_green[3]:fill(0)
-- image.display{image=im_green, legend='Green Plane'} -- change window name to green plane

-- To club the two images under the same window:
-- image.display{image = {im_scale, im_green}, legend = 'Orginal | Green', zoom = 1.0}


-- Now to club the image and its RGB components all in a window:
im_red = torch.zeros(#im_scale)
im_red[1] = im_scale[1]
im_blue = torch.zeros(#im_scale)
im_blue[3] = im_scale[3]

-- image.display{image={im_scale, im_red, im_green, im_blue}, nrow=2, zoom=0.5, legend='Original | R | G | B'}

-- To save this image in a new image we do the following:
outputImg = image.toDisplayTensor{input={im_scale, im_red, im_green, im_blue}, nrow=2, zoom=1, legend='Original | R | G | B'}
-- image.display{image=outputImg, zoom = 1, legend = 'Output'} -- to check the results

print(#outputImg) -- 3 X 512 X 512

-- image.savePNG('lenaRGB.png', im) -- to save the image in the directory
-- image.savePNG('lena.png', outputImg) -- to save the image in the directory

cropImg = image.crop(outputImg, 0, 0, 512/2, 512/2) -- to get just the top left quarter of the image

-- image.display{image=cropImg, zoom = 1, legend = 'Cropped'} -- to check the results


-- image.display(image.drawText(image.lena(), 'hello\nworld', 10, 10)) -- to write on image
-- image.display(image.drawText(image.lena(), 'hello\nworld', 10, 20, {color = {0,255,0},size = 5}))
-- image.display(image.drawText(image.lena(), 'hello\nworld', 10, 20, {color = {0,255,0},size = 5,bg = {255,0,0}}))

-- image.display(image.drawRect(image.lena(), 200, 200, 370, 400, {lineWidth = 5, color = {0,255,0}})) -- to draw a rectangle
