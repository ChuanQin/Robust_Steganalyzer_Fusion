function [Feature]=featureMerge(F)
% F is the struct of multiple block of features extracted by SRMQ1. This
% function can merge the blocks into a whole matrix of feature. 
% if you evaluate the next codes "[M N]=size(Feature)" after the feature fusion, then the M is the
% number of the instance, while the N is the dimentional of the feature.

names=fieldnames(F);% get the names of all the blocks of the features
Feature=single([]);
for i=1:length(names)
    index = names{i};
    if (isa(index, 'cell'))
        f = F.(index{:});              
    elseif ischar(index)
        
        % Return the first element, if a comma separated list is generated
        try
            f = F.(deblank(index)); % deblank field name                  
        catch exception %#ok
            tmp = cell(1,length(f));
            [tmp{:}] = deal(F.(deblank(index)));
            f = tmp{1};
        end                   
    else
        error(message('MATLAB:getfield:InvalidType'));
    end
%     single_one=getfield2(F,names(i));
    Feature=[Feature f];% merge the features
    for j=1:size(f,2)
        NewOrigin(1,j) =i;
    end
end
