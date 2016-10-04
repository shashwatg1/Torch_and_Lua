-- To plot using gnuplot do the following

require 'gnuplot'

gnuplot.plot(torch.linspace(3,8,10)) -- plots linear equispaced points

gnuplot.plot(torch.logspace(3,8,10)) -- plots on log scale

gnuplot.plot(torch.logspace(math.log10(3),math.log10(8),10)) -- same as linspace

gnuplot.hist(torch.randn(1000)) -- normal dist
gnuplot.hist(torch.randn(1e6)) -- classic bell function shape because of many samples

gnuplot.hist(torch.rand(1000))
gnuplot.hist(torch.rand(1e6)) -- uniform dist