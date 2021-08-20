function TRN_TST_EC_MIX(cover_dir, stego_dir, adv_dir, feature_dir, payload, ec_path, ref_trn_dir, ref_val_dir, ref_tst_dir)
    fprintf([stego_dir, '\n']);
    fprintf([feature_dir, '\n']);
    if ~exist(feature_dir,'dir'); mkdir(feature_dir); end

    %% extract features
    fprintf('feature extraction started.\n');
    % feature_extraction(cover_dir, stego_dir, feature_dir, payload);
    fprintf('feature extraction complete.\n');
    
    %% load features
    cover_fea_dir = [feature_dir, 'cover.mat'];
    % Conventional stego
    split_stego = split(stego_dir, '/'); 
    stego_fea_dir = [feature_dir, split_stego{end-2}, '_', num2str(payload), '.mat'];
    % Adversarial stego
    split_adv = split(adv_dir, '/'); 
    adv_fea_dir = [feature_dir, split_adv{end-4}, '_', split_adv{end-3}, '_', split_adv{end-2}, '_', split_adv{end-1}, '.mat'];

    cover = matfile(cover_fea_dir);
    stego = matfile(stego_fea_dir); 
    adv = matfile(adv_fea_dir);
    names = cover.names;
    names = sort(names);
    cover = cover.F;
    stego = stego.F;
    adv = adv.F;

    % split training and testing sets (with specific split seed)
    trn_items = dir(ref_trn_dir); trn_items = trn_items(3:end);
    val_items = dir(ref_val_dir); val_items = val_items(3:end);
    trn = cell(length(trn_items)+length(val_items),1);
    for i=1:length(trn_items)
        trn{i,1} = trn_items(i).name;
    end
    for i=1:length(val_items)
        trn{i+length(trn_items),1} = val_items(i).name;
    end
    tst_items = dir(ref_tst_dir); tst_items = tst_items(3:end);
    tst = cell(length(tst_items),1);
    for i=1:length(tst_items)
        tst{i,1} = tst_items(i).name;
    end
    full_ind = 1:20000;
    trn_ind = full_ind(ismember(names, trn));
    tst_ind = full_ind(ismember(names, tst));
    TRN_cover = cover(trn_ind,:); TST_cover = cover(tst_ind,:);
    TRN_stego = stego(trn_ind,:); TST_stego = stego(tst_ind,:);
    TRN_adv = adv(trn_ind,:); TST_adv = adv(tst_ind,:);
    TRN_mix = [TRN_stego(1:int32(size(TRN_stego,1)/2),:); TRN_adv(int32(size(TRN_adv,1)/2)+1:end,:)];

    % training
    fprintf([cover_fea_dir, '\n']);
    fprintf([stego_fea_dir, '\n']);
    [trained_ensemble,~] = ensemble_training(TRN_cover,TRN_mix);
    split_ec_path = split(ec_path, '/'); ec_dir = [];
    for i=1:length(split_ec_path)-1
        dir_cell = split_ec_path(i);
        ec_dir = [ec_dir, dir_cell{1}, '/'];
    end
    if ~exist(ec_dir,'dir'); mkdir(ec_dir); end
    save(ec_path, 'trained_ensemble');
    % testing
    test_results_cover = ensemble_testing(TST_cover,trained_ensemble);
    test_results_stego = ensemble_testing(TST_stego,trained_ensemble);
    test_results_adv = ensemble_testing(TST_adv,trained_ensemble);
    cover_acc = sum(test_results_cover.predictions==-1);
    stego_acc = sum(test_results_stego.predictions==+1);
    adv_acc = sum(test_results_adv.predictions==+1);

    % print
    fprintf('Cover Accuracy: %f\n', cover_acc/length(tst_ind));
    fprintf('Stego Accuracy: %f\n', stego_acc/length(tst_ind));
    fprintf('Average Classification Accuracy: %f\n', (cover_acc+stego_acc)/(2*length(tst_ind)))
    fprintf('Adversarial Examples Accuracy: %f\n', adv_acc/length(tst_ind));
end