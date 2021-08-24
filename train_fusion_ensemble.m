function train_fusion_ensemble(hand_stego_feature_path, deep_stego_feature_path, hand_clf, deep_clf, domain)
    if nargin == 4
        domain = 'spatial';
    % split training and testing sets
    training_names = matfile('training_names.mat'); trn_names = training_names.training_names;
    testing_names = matfile('testing_names.mat'); tst_names = testing_names.testing_names;
    full_ind = 1:20000;
    trn = cell(length(trn_names),1);
    for i=1:length(trn_names)
        if domain == 'spatial'
            name = trn_names(i,:);
        elseif domain == 'jpeg'
            pre_name = split(name,'.'); pre_name = pre_name{1};
            name = [pre_name, '.jpg'];
        end
        trn(i,1) = {name(~isspace(name))};
    end
    trn_ind = full_ind(ismember(names, trn));
    tst = cell(length(tst_names),1);
    for i=1:length(tst_names)
        if domain == 'spatial'
            name = tst_names(i,:);
        elseif domain == 'jpeg'
            pre_name = split(name,'.'); pre_name = pre_name{1};
            name = [pre_name, '.jpg'];
        end
        tst(i,1) = {name(~isspace(name))};
    end
    tst_ind = full_ind(ismember(names, tst));
    randind = [trn_ind; tst_ind];
    % train/load handcrafted feature-based steganalyzer
    train_ensemble(hand_stego_feature_path, hand_clf, randind);
    % train/load deep steganalyzer
    train_ensemble(deep_stego_feature_path, deep_clf, randind);
    % eval fusion results on adversarial stego images
    
end