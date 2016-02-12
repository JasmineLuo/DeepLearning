function net = proj6_part2_cnn_init()
%code for Computer Vision, Georgia Tech by James Hays

net = load('imagenet-vgg-f.mat') ;
%We'll need to make some modifications to this network. First, the network
%accepts 
f=1/100;
%This network is missing the dropout layers (because they're not needed at
%test time). It may be a good idea to reinsert dropout layers between the
%fully connected layers.

net.layers{end}={};
net.layers{end-1}={};

net.layers{end-1} = net.layers{end-2}; %%lower the layer of fc7 to insert a dropout layer before it
net.layers{end-2} = net.layers{end-3}; %%lower the layer of fc7 to insert a dropout layer before it

net.layers{end-3} = struct('type','dropout','rate',0.3);

net.layers{end} = struct('type','dropout','rate',0.3); %% the dropout between fc7 & fc8

net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{f*randn(3,3,4096,15, 'single'), zeros(1, 15, 'single')}}, ...
                           'stride', 1, ...
                           'pad', 1, ...
                           'name', 'fc8') ;  %% new fc8
                       
net.layers{end+1} = struct('type', 'softmaxloss') ; %% new softmaxloss

vl_simplenn_display(net, 'inputSize', [224 224 3 50])

% [copied from the project webpage]
% proj6_part2_cnn_init.m will start with net = load('imagenet-vgg-f.mat');
% and then edit the network rather than specifying the structure from
% scratch.

% You need to make the following edits to the network: The final two
% layers, fc8 and the softmax layer, should be removed and specified again
% using the same syntax seen in Part 1. The original fc8 had an input data
% depth of 4096 and an output data depth of 1000 (for 1000 ImageNet
% categories). We need the output depth to be 15, instead. The weights can
% be randomly initialized just like in Part 1.

% The dropout layers used to train VGG-F are missing from the pretrained
% model (probably because they're not used at test time). It's probably a
% good idea to add one or both of them back in between fc6 and fc7 and
% between fc7 and fc8.

